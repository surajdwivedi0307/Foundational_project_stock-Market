# Long-Term Investor Model for Indian Market
# Implementation based on CRISP-ML(Q) Framework

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import pickle
import warnings
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
import shap
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import wandb



# ─── Inject your key directly into the environment ─────────────────────────────
os.environ["WANDB_API_KEY"] = "7fe7511aca549b00e22a6009080f21f46578f91e"


# Suppress warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
plt.style.use('ggplot')


# Helper function for converting Timestamps in a dict to strings
def convert_timestamps(obj):
    """
    Recursively convert any datetime or pandas Timestamp objects in obj to strings.
    """
    if isinstance(obj, dict):
        new_obj = {}
        for k, v in obj.items():
            if isinstance(k, (pd.Timestamp, datetime)):
                k = k.strftime('%Y-%m-%d %H:%M:%S')
            new_obj[k] = convert_timestamps(v)
        return new_obj
    elif isinstance(obj, list):
        return [convert_timestamps(item) for item in obj]
    elif isinstance(obj, (pd.Timestamp, datetime)):
        return obj.strftime('%Y-%m-%d %H:%M:%S')
    else:
        return obj
def compute_psi(expected, actual, buckets=10):
    """
    Simple Population Stability Index (PSI) implementation.
    """
    def _percentile_breaks(data, buckets):
        return np.percentile(data, np.linspace(0, 100, buckets + 1))

    # build combined breakpoints
    breaks = _percentile_breaks(np.concatenate((expected, actual)), buckets)

    # histogram percentage in each bucket
    e_perc = np.histogram(expected, bins=breaks)[0] / len(expected)
    a_perc = np.histogram(actual,   bins=breaks)[0] / len(actual)

    # avoid zero‐division
    a_perc = np.where(a_perc == 0, 1e-8, a_perc)
    e_perc = np.where(e_perc == 0, 1e-8, e_perc)

    # PSI formula
    return np.sum((e_perc - a_perc) * np.log(e_perc / a_perc))    

class LongTermInvestorModel:
    """
    Long-Term Investor Model for the Indian Market.
    This model focuses on fundamental analysis and long-term growth prediction
    for a time horizon of 1-3 years.
    """
    # In long_term_investor_model.py, alongside your other helper functions:

    def detect_data_drift(self, current_df: pd.DataFrame, top_n: int = 5) -> pd.Series:
        """
        Compare current_df against self.train_data feature‑wise and return
        the top_n PSI scores (largest drift).
        """
        psi_scores = {}
        for feat in self.feature_names:
            if feat in current_df.columns:
                psi_scores[feat] = compute_psi(
                    self.train_data[feat].values,
                    current_df[feat].values
                )
        # return top_n highest‐drift features
        return pd.Series(psi_scores, name="PSI Score").sort_values(ascending=False).head(top_n)
    
    def __init__(self, lookback_years=5, prediction_horizon=365, model_path=None):
        """
        Initialize the model with configuration parameters.
        """
        self.lookback_years = lookback_years
        self.prediction_horizon = prediction_horizon
        self.model_path = model_path
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = None
        self.shap_values = None
        self.explainer = None
        self.train_data = None
        self.test_data = None
        self.evaluation_metrics = {}
        # Define Indian stock exchanges
        self.exchanges = ['.NS', '.BO']  # NSE and BSE
        
    def load_stock_data(self, symbol, exchange='.NS'):
        """
        Load historical stock data from Yahoo Finance.
        """
        if not symbol.endswith(exchange):
            full_symbol = f"{symbol}{exchange}"
        else:
            full_symbol = symbol
            
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * self.lookback_years)
        
        try:
            data = yf.download(full_symbol, start=start_date, end=end_date)
            if data.empty or len(data) < 100:
                print(f"Insufficient data for {full_symbol}, trying alternative exchange")
                alt_exchange = '.BO' if exchange == '.NS' else '.NS'
                full_symbol = f"{symbol.replace(exchange, '')}{alt_exchange}"
                data = yf.download(full_symbol, start=start_date, end=end_date)
            if data.empty:
                raise ValueError(f"No data available for {symbol} on either NSE or BSE")
            print(f"Successfully loaded {len(data)} records for {full_symbol}")
            return data, full_symbol
        except Exception as e:
            print(f"Error loading data for {full_symbol}: {str(e)}")
            return None, None
    
    def load_fundamental_data(self, symbol):
        """
        Load fundamental data for a stock.
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            fundamentals = {
                'marketCap': info.get('marketCap', np.nan),
                'trailingPE': info.get('trailingPE', np.nan),
                'forwardPE': info.get('forwardPE', np.nan),
                'dividendYield': info.get('dividendYield', np.nan),
                'bookValue': info.get('bookValue', np.nan),
                'priceToBook': info.get('priceToBook', np.nan),
                'returnOnEquity': info.get('returnOnEquity', np.nan),
                'totalDebt': info.get('totalDebt', np.nan),
                'operatingMargins': info.get('operatingMargins', np.nan),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown')
            }
            try:
                financials = ticker.quarterly_financials
                if not financials.empty:
                    recent_quarter = financials.columns[0]
                    fundamentals['totalRevenue'] = financials.loc['Total Revenue', recent_quarter] if 'Total Revenue' in financials.index else np.nan
                    fundamentals['grossProfit'] = financials.loc['Gross Profit', recent_quarter] if 'Gross Profit' in financials.index else np.nan
                    fundamentals['netIncome'] = financials.loc['Net Income', recent_quarter] if 'Net Income' in financials.index else np.nan
            except:
                print(f"Quarterly financials not available for {symbol}")
            return fundamentals
        except Exception as e:
            print(f"Error loading fundamental data for {symbol}: {str(e)}")
            return {}
    
    def get_sector_performance(self, sectors):
        """
        Get performance data for specified Indian market sectors.
        """
        sector_indices = {
            'Finance': 'NIFTY_FIN.NS',
            'IT': 'NIFTIT.NS',
            'Pharma': 'NIFTYPHARMA.NS',
            'Auto': 'NIFTYAUTO.NS',
            'FMCG': 'NIFTYFMCG.NS',
            'Metal': 'NIFTYMETAL.NS',
            'Energy': 'NIFTYENERGY.NS',
            'Realty': 'NIFTYREALTY.NS'
        }
        relevant_indices = {k: v for k, v in sector_indices.items() if k in sectors}
        if not relevant_indices:
            return pd.DataFrame()
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        sector_data = {}
        for sector, symbol in relevant_indices.items():
            try:
                data = yf.download(symbol, start=start_date, end=end_date)
                if not data.empty:
                    ytd_return = (data['Close'].iloc[-1] / data['Close'].iloc[0] - 1) * 100
                    sector_data[sector] = ytd_return
            except:
                print(f"Could not load data for {sector} ({symbol})")
        return sector_data
    
    def prepare_features(self, data, symbol, add_fundamentals=True):
        # Use Adjusted Close if available so that dividends and splits are reflected
        if 'Adj Close' in data.columns:
            data['Close'] = data['Adj Close']
        if isinstance(data.columns, pd.MultiIndex):
            if len(data.columns.levels[1]) == 1:
                df = data.copy()
                df.columns = df.columns.droplevel(1)
            elif symbol in data.columns.get_level_values(1):
                df = data.xs(symbol, axis=1, level=1)
            else:
                first_ticker = data.columns.levels[1][0]
                print(f"Symbol {symbol} not found in MultiIndex columns; defaulting to {first_ticker}")
                df = data.xs(first_ticker, axis=1, level=1)
        else:
            df = data.copy()
        if isinstance(df['Close'], pd.DataFrame):
            df['Close'] = df['Close'].iloc[:, 0]
        if 'Volume' in df.columns and isinstance(df['Volume'], pd.DataFrame):
            df['Volume'] = df['Volume'].iloc[:, 0]
        df['MA50'] = df['Close'].rolling(window=50).mean()
        df['MA200'] = df['Close'].rolling(window=200).mean()
        df['MA50'] = df['MA50'].squeeze()
        df['MA200'] = df['MA200'].squeeze()
        df['MA50_ratio'] = df['Close'] / df['MA50']
        df['MA200_ratio'] = df['Close'] / df['MA200']
        df['Volatility_30'] = df['Close'].pct_change().rolling(window=30).std() * np.sqrt(252)
        df['Volatility_90'] = df['Close'].pct_change().rolling(window=90).std() * np.sqrt(252)
        df['Return_30'] = df['Close'].pct_change(periods=30)
        df['Return_90'] = df['Close'].pct_change(periods=90)
        df['Return_180'] = df['Close'].pct_change(periods=180)
        df['Return_365'] = df['Close'].pct_change(periods=365)
        df['Volume_ratio_30'] = df['Volume'] / df['Volume'].rolling(window=30).mean()
        df['Momentum_90'] = df['Close'] - df['Close'].shift(90)
        df['Momentum_180'] = df['Close'] - df['Close'].shift(180)
        if add_fundamentals:
            fundamentals = self.load_fundamental_data(symbol)
            # Define a fixed list of fundamental keys
            fixed_keys = [
                'marketcap', 'trailingpe', 'forwardpe', 'dividendyield',
                'bookvalue', 'pricetobook', 'returnonequity', 'totaldebt',
                'operatingmargins', 'totalrevenue', 'grossprofit', 'netincome'
            ]
            for key in fixed_keys:
                col_name = "fund_" + key.lower()
                if key in fundamentals and not pd.isna(fundamentals[key]):
                    df[col_name] = fundamentals[key]
                else:
                    df[col_name] = 0.0
            if 'sector' in fundamentals and fundamentals['sector'] != 'Unknown':
                sector_perf = self.get_sector_performance([fundamentals['sector']])
                if fundamentals['sector'] in sector_perf:
                    df['sector_performance_ytd'] = sector_perf[fundamentals['sector']]
        df = df.dropna()
        forward_period = 365
        df['Future_Return'] = df['Close'].pct_change(forward_period).shift(-forward_period)
        return df
    
    def split_data(self, df, test_ratio=0.2):
        """
        Split data into training and testing sets.
        """
        cutoff_idx = int(len(df) * (1 - test_ratio))
        train_data = df.iloc[:cutoff_idx].copy()
        test_data = df.iloc[cutoff_idx:].copy()
        train_data = train_data.dropna(subset=['Future_Return'])
        test_data = test_data.dropna(subset=['Future_Return'])
        self.train_data = train_data
        self.test_data = test_data
        feature_cols = [col for col in df.columns if col not in ['Future_Return', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Date']]
        X_train = train_data[feature_cols]
        y_train = train_data['Future_Return']
        X_test = test_data[feature_cols]
        y_test = test_data['Future_Return']
        self.feature_names = feature_cols
        return X_train, X_test, y_train, y_test

    def train_model(self, X_train, y_train):
        import os, pickle, pandas as pd, wandb
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import LinearRegression

        # ─── 1) W&B Setup ─────────────────────────────────────────────────────
        # Inject API key and login
        os.environ["WANDB_API_KEY"] = "7fe7511aca549b00e22a6009080f21f46578f91e"
        wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True)

        # Initialize a new run
        run = wandb.init(
            project="long_term_investor_model",
            name=f"training_{self.prediction_horizon}d",
            config={
                "lookback_years": self.lookback_years,
                "prediction_horizon": self.prediction_horizon,
                "model_type": "RF + LR ensemble",
                "rf_params": {"n_estimators": 100, "max_depth": 10},
            }
        )

        # ─── 2) Scale the data ─────────────────────────────────────────────────
        X_train_scaled = self.scaler.fit_transform(X_train)

        # ─── 3) Train Random Forest & Linear Regression ───────────────────────
        print("Training Random Forest model...")
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        )
        rf_model.fit(X_train_scaled, y_train)

        print("Training Linear Regression model...")
        lr_model = LinearRegression()
        lr_model.fit(X_train_scaled, y_train)

        # ─── 4) Capture & log feature importances ────────────────────────────
        fi_df = pd.DataFrame({
            "Feature": X_train.columns,
            "Importance": rf_model.feature_importances_
        }).sort_values("Importance", ascending=False)

        # Store locally and log to W&B
        self.feature_importance = fi_df
        print("Columns in feature_importance:", self.feature_importance.columns.tolist())
        run.log({"feature_importances": wandb.Table(dataframe=fi_df)})

        # ─── 5) Bundle & register pickle artifact ────────────────────────────
        artifact_path = "long_term_model.pkl"
        with open(artifact_path, "wb") as f:
            pickle.dump({
                "rf": rf_model,
                "lr": lr_model,
                "scaler": self.scaler
            }, f)
        run.log_artifact(artifact_path, name="long_term_model")

        # ─── 6) SHAP setup ───────────────────────────────────────────────────
        self.explainer = shap.TreeExplainer(rf_model)
        self.shap_values = self.explainer.shap_values(X_train_scaled)

        # ─── 7) Finalize internal state ───────────────────────────────────────
        self.model = {"rf": rf_model, "lr": lr_model, "scaler": self.scaler}
        self.feature_names = list(X_train.columns)
        self.train_data = X_train.copy()

        # ─── 8) Close the W&B run ─────────────────────────────────────────────
        run.finish()
        return self.model

    
    def save_model(self, filepath):
        if self.model is not None:
            data_to_save = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_importance': self.feature_importance,
                'shap_values': self.shap_values,
                'feature_names': self.feature_names,
                'train_data': self.train_data  # save training data for recomputing SHAP if needed
            }
            with open(filepath, 'wb') as f:
                pickle.dump(data_to_save, f)
            print(f"Model saved to {filepath}")
        else:
            print("No model to save. Please train a model first.")

    def load_model(self, filepath):
        try:
            with open(filepath, 'rb') as f:
                data_loaded = pickle.load(f)
            self.model = data_loaded.get('model', None)
            self.scaler = data_loaded.get('scaler', None)
            self.feature_importance = data_loaded.get('feature_importance', None)
            self.shap_values = data_loaded.get('shap_values', None)
            self.feature_names = data_loaded.get('feature_names', None)
            self.train_data = data_loaded.get('train_data', None)
            print(f"Model loaded from {filepath}")
            if self.model and 'rf' in self.model and self.model['rf'] is not None:
                self.explainer = shap.TreeExplainer(self.model['rf'])
            else:
                self.explainer = None
            
            # Recompute SHAP values if missing and training data is available
            if self.shap_values is None and self.explainer is not None and self.train_data is not None and self.feature_names is not None:
                X_train_scaled = self.scaler.transform(self.train_data[self.feature_names])
                self.shap_values = self.explainer.shap_values(X_train_scaled)
                print("Recomputed SHAP values after loading model.")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the model's performance.
        """
        if self.model is None:
            print("No model to evaluate. Please train or load a model first.")
            return {}
        X_test_scaled = self.scaler.transform(X_test)
        rf_pred = self.model['rf'].predict(X_test_scaled)
        lr_pred = self.model['lr'].predict(X_test_scaled)
        y_pred = (rf_pred + lr_pred) / 2
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        direction_actual = (y_test > 0).astype(int)
        direction_pred = (y_pred > 0).astype(int)
        directional_accuracy = np.mean(direction_actual == direction_pred)
        investment_returns = y_test[y_pred > 0].mean() if any(y_pred > 0) else 0
        self.evaluation_metrics = {
            'mse': mse,
            'rmse': rmse,
            'mape': mape,
            'r2': r2,
            'directional_accuracy': directional_accuracy,
            'investment_returns': investment_returns
        }
        return self.evaluation_metrics
    
    def predict(self, X):
        """
        Make predictions using the trained model.
        """
        if self.model is None:
            print("No model to make predictions. Please train or load a model first.")
            return None
        X_scaled = self.scaler.transform(X)
        rf_pred = self.model['rf'].predict(X_scaled)
        lr_pred = self.model['lr'].predict(X_scaled)
        y_pred = (rf_pred + lr_pred) / 2
        return y_pred
    
    def predict_stock(self, symbol, exchange='.NS'):
        """
        Make predictions for a specific stock.
        """
        data, full_symbol = self.load_stock_data(symbol, exchange)
        if data is None:
            return None
        df = self.prepare_features(data, full_symbol)
        latest_data = df.iloc[-1:].copy()
        if 'Future_Return' in latest_data.columns:
            latest_data = latest_data.drop(columns=['Future_Return'])
        if hasattr(self, 'feature_names') and self.feature_names is not None:
            feature_cols = self.feature_names
        else:
            feature_cols = [col for col in df.columns if col not in ['Future_Return', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Date']]
        missing = set(feature_cols) - set(latest_data.columns)
        for col in missing:
            latest_data[col] = 0.0
        X = latest_data[feature_cols]
        # Get the raw prediction (this is the 1-year (365-day) target return as learned during training)
        raw_predicted_return = float(self.predict(X)[0])
        normalized_return = raw_predicted_return / 2.0 if abs(raw_predicted_return) > 0.40 else raw_predicted_return
        
        # Retrieve the current price from the processed dataframe
        current_price = float(df['Close'].iloc[-1])
        # Scale the predicted annual return to the user-selected horizon (assumed compound returns)
        # The model's return was computed for 365 days, so if the user selects a shorter horizon, scale it:
        scaled_return = (1 + normalized_return) ** (self.prediction_horizon / 365) - 1
        predicted_price = current_price * (1 + scaled_return)
        
        X_scaled = self.scaler.transform(X)
        if self.explainer is None:
            self.explainer = shap.TreeExplainer(self.model['rf'])
        shap_values = self.explainer.shap_values(X_scaled)
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Value': X.values[0],
            'Impact': shap_values[0]
        })
        feature_importance['AbsImpact'] = np.abs(feature_importance['Impact'])
        top_factors = feature_importance.sort_values('AbsImpact', ascending=False).head(5)
        
        explanation_lines = []
        if 'Momentum_90' in df.columns:
            mom90 = df['Momentum_90'].iloc[-1]
            explanation_lines.append(f"90-day Momentum: {mom90:.2f} (indicates recent trend).")
        if 'MA200_ratio' in df.columns:
            ma200_ratio = df['MA200_ratio'].iloc[-1]
            explanation_lines.append(f"200-day MA Ratio: {ma200_ratio:.2f} (above 1 suggests strength).")
        if 'MA50_ratio' in df.columns:
            ma50_ratio = df['MA50_ratio'].iloc[-1]
            explanation_lines.append(f"50-day MA Ratio: {ma50_ratio:.2f} (reflects medium-term trend).")
        if 'fund_dividendyield' in df.columns:
            div_yield = df['fund_dividendyield'].iloc[-1]
            explanation_lines.append(f"Dividend Yield: {div_yield:.2%} (regular payouts).")
        if not top_factors.empty:
            top_factor = top_factors.iloc[0]
            explanation_lines.append(f"Dominant factor: '{top_factor['Feature']}' with impact {top_factor['Impact']:.2f}.")
        else:
            explanation_lines.append("No dominant factor was identified.")
        explanation_lines.append(f"Raw predicted return (1-year): {raw_predicted_return:.2%} normalized to {normalized_return:.2%}.")
        explanation_lines.append(f"Scaled return for {self.prediction_horizon} days: {scaled_return:.2%}.")
        explanation = "Reasons for prediction:\n" + "\n".join(f"- {line}" for line in explanation_lines)
        
        if scaled_return > 0.15:
            recommendation = 'Strong Buy'
        elif scaled_return > 0.05:
            recommendation = 'Buy'
        elif scaled_return > -0.05:
            recommendation = 'Hold'
        elif scaled_return > -0.15:
            recommendation = 'Sell'
        else:
            recommendation = 'Strong Sell'
            
        price_history = {str(k): float(v) for k, v in df['Close'].tail(252).to_dict().items()}
        results = {
            'symbol': full_symbol,
            'current_date': datetime.now().strftime('%Y-%m-%d'),
            'prediction_horizon_days': self.prediction_horizon,
            'prediction_horizon': f"{self.prediction_horizon} days",
            'current_price': current_price,
            'predicted_date': (datetime.now() + timedelta(days=self.prediction_horizon)).strftime('%Y-%m-%d'),
            'predicted_return': scaled_return,
            'predicted_price': predicted_price,
            'top_factors': top_factors.to_dict(orient='records'),
            'explanation': explanation,
            'price_history': price_history,
            'recommendation': recommendation
        }
        return results
    
    def plot_feature_importance(self):
        if self.feature_importance is None:
            print("No feature importance available. Please train a model first.")
            return None
        sorted_imp = self.feature_importance.sort_values('Importance', ascending=False).head(15)
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=sorted_imp, ax=ax)
        ax.set_title('Top 15 Feature Importance for Long-Term Model')
        plt.tight_layout()
        return fig
    def plot_shap_values(self, X=None):
        if self.explainer is None:
            print("No SHAP explainer available. Please train a model first.")
            return None
        if self.shap_values is None:
            print("No SHAP values available. Please train a model first.")
            return None
        try:
            if X is None:
                if self.train_data is None or self.feature_names is None:
                    print("No training data or feature names available.")
                    return None
                fig = plt.figure(figsize=(12, 8))
                shap.summary_plot(self.shap_values, self.train_data[self.feature_names], show=False)
                plt.title('SHAP Values for Long-Term Model')
                plt.tight_layout()
                return plt.gcf()
            else:
                X_scaled = self.scaler.transform(X)
                local_shap_vals = self.explainer.shap_values(X_scaled)
                fig = plt.figure(figsize=(12, 8))
                shap.summary_plot(local_shap_vals, X, show=False)
                plt.title('SHAP Values for Current Prediction')
                plt.tight_layout()
                return plt.gcf()
        except Exception as e:
            print("Error in plot_shap_values:", e)
            return None
    
    def create_portfolio_backtest(self, symbols, start_date=None, end_date=None, investment=100000):
        """
        Create a backtest of the model's performance on a portfolio of stocks.
        The portfolio is rebalanced every 30 business days (or on the first day).
        
        Parameters:
          symbols: List of stock symbols (e.g. ['TCS.NS', 'INFY.NS', ...])
          start_date: datetime or string; start date for the backtest.
          end_date: datetime or string; end date for the backtest.
          investment: Initial cash investment.
        
        Returns:
          dict: Contains dates, portfolio value, trades, and performance metrics.
        """
        # Set default dates if not provided
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=365)
            
        # Check if the model is trained
        if self.model is None:
            print("No model available. Please train or load a model first.")
            return None
        
        results = {
            'dates': [],
            'portfolio_value': [],
            'trades': []
        }
        cash = investment
        holdings = {}
        portfolio_values = []
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')  # business days
        
        first_day = True  # flag to force initial trade on day 0

        for i, current_date in enumerate(date_range):
            # Update the value of current holdings
            day_value = cash
            for sym, holding in list(holdings.items()):
                try:
                    price_data = yf.download(sym, start=current_date, end=current_date + timedelta(days=1), progress=False)
                    if not price_data.empty:
                        current_price = float(price_data['Close'].iloc[0])
                        holding['price'] = current_price
                        holding['value'] = holding['quantity'] * current_price
                    day_value += holding.get('value', 0)
                except Exception as e:
                    print(f"Error updating {sym} price on {current_date}: {e}")
            
            # Rebalance on day 0 and every 30 business days
            if first_day or (i % 30 == 0):
                # Sell all current holdings
                for sym, holding in holdings.items():
                    cash += holding['value']
                    results['trades'].append({
                        'date': current_date.strftime('%Y-%m-%d'),
                        'action': 'SELL',
                        'symbol': sym,
                        'quantity': holding['quantity'],
                        'price': holding['price'],
                        'value': holding['value']
                    })
                holdings = {}

                # Get predictions for all symbols
                predictions = []
                for sym in symbols:
                    try:
                        pred = self.predict_stock(sym)
                        if pred is not None:
                            predictions.append(pred)
                    except Exception as e:
                        print(f"Error predicting {sym}: {e}")
                # Sort predictions by predicted return (highest first)
                predictions.sort(key=lambda x: x['predicted_return'], reverse=True)
                # Buy top 3 stocks regardless of predicted sign (adjust if needed)
                top_stocks = predictions[:3]
                if top_stocks:
                    allocation = cash / len(top_stocks)
                    for stock in top_stocks:
                        try:
                            price_data = yf.download(stock['symbol'], start=current_date, end=current_date + timedelta(days=1), progress=False)
                            if price_data.empty:
                                continue
                            current_price = float(price_data['Close'].iloc[0])
                            quantity = int(allocation / current_price)
                            if quantity <= 0:
                                continue
                            cost = quantity * current_price
                            cash -= cost
                            holdings[stock['symbol']] = {
                                'quantity': quantity,
                                'price': current_price,
                                'value': cost
                            }
                            results['trades'].append({
                                'date': current_date.strftime('%Y-%m-%d'),
                                'action': 'BUY',
                                'symbol': stock['symbol'],
                                'quantity': quantity,
                                'price': current_price,
                                'value': cost,
                                'predicted_return': stock['predicted_return']
                            })
                        except Exception as e:
                            print(f"Error buying {stock['symbol']} on {current_date}: {e}")
                first_day = False

            results['dates'].append(current_date.strftime('%Y-%m-%d'))
            portfolio_values.append(day_value)
            results['portfolio_value'].append(day_value)

        # Calculate performance metrics
        final_value = portfolio_values[-1]
        overall_return = (final_value / investment) - 1
        num_periods = len(portfolio_values)
        annualized_return = (final_value / investment) ** (252 / num_periods) - 1 if num_periods > 1 else 0
        daily_returns = np.diff(portfolio_values) / portfolio_values[:-1] if len(portfolio_values) > 1 else np.array([])
        volatility = np.std(daily_returns) * np.sqrt(252) if len(daily_returns) > 0 else 0
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0

        # Compute maximum drawdown
        peak = portfolio_values[0]
        max_drawdown = 0
        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        metrics = {
            'initial_investment': investment,
            'final_value': final_value,
            'total_return': overall_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }
        results['metrics'] = metrics
        return results
    
    def plot_backtest_results(self, backtest_results):
        """
        Plot the results of a portfolio backtest.
        """
        if not backtest_results:
            print("No backtest results to plot.")
            return None
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Portfolio Value", "Cumulative Returns"),
            vertical_spacing=0.15,
            row_heights=[0.7, 0.3]
        )
        fig.add_trace(
            go.Scatter(
                x=backtest_results['dates'],
                y=backtest_results['portfolio_value'],
                mode='lines',
                name='Portfolio Value',
                line=dict(color='rgb(49, 130, 189)', width=2)
            ),
            row=1, col=1
        )
        baseline = [backtest_results['metrics']['initial_investment']] * len(backtest_results['dates'])
        fig.add_trace(
            go.Scatter(
                x=backtest_results['dates'],
                y=baseline,
                mode='lines',
                name='Initial Investment',
                line=dict(color='rgba(200, 200, 200, 0.7)', width=2, dash='dash')
            ),
            row=1, col=1
        )
        cumulative_returns = [pv / backtest_results['metrics']['initial_investment'] - 1 
                              for pv in backtest_results['portfolio_value']]
        fig.add_trace(
            go.Scatter(
                x=backtest_results['dates'],
                y=cumulative_returns,
                mode='lines',
                name='Cumulative Return',
                line=dict(color='rgb(214, 39, 40)', width=2)
            ),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=[backtest_results['dates'][0], backtest_results['dates'][-1]],
                y=[0, 0],
                mode='lines',
                name='Breakeven',
                line=dict(color='rgba(200, 200, 200, 0.7)', width=1, dash='dash')
            ),
            row=2, col=1
        )
        buy_dates = []
        buy_values = []
        sell_dates = []
        sell_values = []
        for trade in backtest_results['trades']:
            if trade['action'] == 'BUY':
                buy_dates.append(trade['date'])
                if trade['date'] in backtest_results['dates']:
                    idx = backtest_results['dates'].index(trade['date'])
                    buy_values.append(backtest_results['portfolio_value'][idx])
            elif trade['action'] == 'SELL':
                sell_dates.append(trade['date'])
                if trade['date'] in backtest_results['dates']:
                    idx = backtest_results['dates'].index(trade['date'])
                    sell_values.append(backtest_results['portfolio_value'][idx])
        if buy_dates:
            fig.add_trace(
                go.Scatter(
                    x=buy_dates,
                    y=buy_values,
                    mode='markers',
                    marker=dict(symbol='triangle-up', size=10, color='green'),
                    name='Buy'
                ),
                row=1, col=1
            )
        if sell_dates:
            fig.add_trace(
                go.Scatter(
                    x=sell_dates,
                    y=sell_values,
                    mode='markers',
                    marker=dict(symbol='triangle-down', size=10, color='red'),
                    name='Sell'
                ),
                row=1, col=1
            )
        metrics_text = (
            f"Total Return: {backtest_results['metrics']['total_return']:.2%}<br>"
            f"Annualized Return: {backtest_results['metrics']['annualized_return']:.2%}<br>"
            f"Sharpe Ratio: {backtest_results['metrics']['sharpe_ratio']:.2f}<br>"
            f"Max Drawdown: {backtest_results['metrics']['max_drawdown']:.2%}"
        )
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.02, y=0.95,
            text=metrics_text,
            showarrow=False,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="gray",
            borderwidth=1,
            font=dict(size=10)
        )
        fig.update_layout(
            title='Portfolio Backtest Results',
            height=800,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            template='plotly_white'
        )
        fig.update_yaxes(title_text="Portfolio Value (₹)", row=1, col=1)
        fig.update_yaxes(title_text="Return (%)", tickformat=".0%", row=2, col=1)
        return fig
    

    def plot_prediction_analysis(self, prediction_result):
        if not prediction_result:
            print("No prediction result to plot.")
            return None
        # Convert price_history keys (which might be Timestamps) to strings
        dates = [d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d) 
                 for d in prediction_result['price_history'].keys()]
        prices = list(prediction_result['price_history'].values())
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                f"Historical and Predicted Price for {prediction_result['symbol']}",
                "Top Factors Driving Prediction",
                "Price Movement Range",
                "Investment Decision Guide"
            ),
            specs=[
                [{"type": "scatter"}, {"type": "bar"}],
                [{"type": "indicator"}, {"type": "domain"}]
            ],
            row_heights=[0.6, 0.4],
            column_widths=[0.6, 0.4],
            vertical_spacing=0.1
        )
        # 1. Historical and predicted price
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=prices,
                mode='lines',
                name='Historical Price',
                line=dict(color='rgb(49,130,189)')
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=[prediction_result['current_date']],
                y=[prediction_result['current_price']],
                mode='markers',
                marker=dict(size=10, color='blue', symbol='circle'),
                name='Current Price'
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=[prediction_result['predicted_date']],
                y=[prediction_result['predicted_price']],
                mode='markers',
                marker=dict(size=12, color='red', symbol='star'),
                name='Predicted Price'
            ),
            row=1, col=1
        )
        price_lower = prediction_result['predicted_price'] * 0.85
        price_upper = prediction_result['predicted_price'] * 1.15
        fig.add_trace(
            go.Scatter(
                x=[prediction_result['predicted_date'], prediction_result['predicted_date']],
                y=[price_lower, price_upper],
                mode='lines',
                line=dict(color='rgba(255,0,0,0.3)', width=2),
                name='Prediction Range'
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=[prediction_result['current_date'], prediction_result['predicted_date']],
                y=[prediction_result['current_price'], prediction_result['predicted_price']],
                mode='lines',
                line=dict(color='rgba(255,0,0,0.5)', width=2, dash='dash'),
                name='Projected Path'
            ),
            row=1, col=1
        )
        # 2. Top factors driving prediction
        top_factors = prediction_result['top_factors']
        factor_names = [f.replace('Fund_', '').replace('_', ' ') for f in [factor['Feature'] for factor in top_factors]]
        factor_impacts = [factor['Impact'] for factor in top_factors]
        colors = ['green' if imp > 0 else 'red' for imp in factor_impacts]
        fig.add_trace(
            go.Bar(
                x=factor_impacts,
                y=factor_names,
                orientation='h',
                marker_color=colors,
                name='Factor Impact'
            ),
            row=1, col=2
        )
        # 3. Price movement indicator
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=prediction_result['predicted_price'],
                delta={'reference': prediction_result['current_price'], 'relative': True, 'valueformat': '.1%'},
                title={'text': "Predicted Change"},
                gauge={
                    'axis': {'range': [None, max(prediction_result['predicted_price'] * 1.2, prediction_result['current_price'] * 1.2)]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, prediction_result['current_price']], 'color': 'lightgray'},
                        {'range': [prediction_result['current_price'], prediction_result['predicted_price']],
                         'color': 'green' if prediction_result['predicted_return'] > 0 else 'red'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': prediction_result['current_price']
                    }
                }
            ),
            row=2, col=1
        )
        # 4. Investment decision guide with global annotations
        recommendation = prediction_result['recommendation']
        recommendation_colors = {
            'Strong Buy': '#006400',
            'Buy': '#32CD32',
            'Hold': '#FFD700',
            'Sell': '#FF6347',
            'Strong Sell': '#8B0000'
        }
        fig.add_trace(
            go.Pie(
                labels=['Strong Buy', 'Buy', 'Hold', 'Sell', 'Strong Sell'],
                values=[1, 1, 1, 1, 1],
                marker=dict(colors=[recommendation_colors[r] for r in ['Strong Buy', 'Buy', 'Hold', 'Sell', 'Strong Sell']]),
                hole=0.7,
                hoverinfo='none',
                textinfo='none',
                showlegend=False
            ),
            row=2, col=2
        )
        fig.add_annotation(
            x=0.85, y=0.15,
            xref="paper", yref="paper",
            text=recommendation,
            showarrow=False,
            font=dict(size=20, color=recommendation_colors[recommendation])
        )
        fig.add_annotation(
            x=0.85, y=0.10,
            xref="paper", yref="paper",
            text=f"Expected Return:<br>{prediction_result['predicted_return']:.2%}",
            showarrow=False,
            font=dict(size=12)
        )
        fig.update_layout(
            title=f"Stock Analysis: {prediction_result['symbol']} (Horizon: {prediction_result['prediction_horizon_days']} days)",
            height=900,
            width=1200,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            template='plotly_white'
        )
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
        fig.update_xaxes(title_text="Impact on Prediction", row=1, col=2)

        # Convert all keys (including Timestamps) to strings.
        def convert_keys(obj):
            if isinstance(obj, dict):
                new_obj = {}
                for k, v in obj.items():
                    if isinstance(k, (pd.Timestamp, datetime)):
                        new_key = k.strftime('%Y-%m-%d %H:%M:%S')
                    else:
                        new_key = str(k)
                    new_obj[new_key] = convert_keys(v)
                return new_obj
            elif isinstance(obj, list):
                return [convert_keys(item) for item in obj]
            else:
                return obj

        fig_dict = convert_keys(fig.to_dict())
        return go.Figure(fig_dict)
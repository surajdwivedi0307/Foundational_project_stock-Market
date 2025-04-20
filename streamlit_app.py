import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import yfinance as yf
import shap
import pickle
import base64
from io import BytesIO

from long_term_investor_model import LongTermInvestorModel

# --- Page Configuration ---
st.set_page_config(
    page_title="Nifty50 Stock Market Project",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS to Enhance UI ---
st.markdown("""
<style>
/* Dark background for the entire app */
.stApp {
    background-color: #121212;
    color: #e0e0e0;
}

/* Center the logo at the top */
div.logo-container {
    display: flex;
    justify-content: center;
    margin-bottom: 20px;
}

/* Style the main header */
h1.main-header {
    text-align: center;
    font-size: 2.5rem;
    color: #ff4081; /* pinkish color to match your logo */
    margin-bottom: 2rem;
    margin-top: 0rem;
}

/* Sub-headers */
h2.sub-header, h3.sub-header {
    color: #ffaaff;
}

/* Info box styling */
div.info-box {
    background-color: #1e1e1e;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #ff4081;
    margin-bottom: 1rem;
}

/* Metric card styling */
.metric-card {
    background-color: #2a2a2a;
    padding: 1rem;
    border-radius: 0.5rem;
    text-align: center;
}

/* Recommendation colors */
.recommendation-buy {
    color: #4caf50; /* green */
    font-weight: bold;
    font-size: 1.2rem;
}
.recommendation-sell {
    color: #f44336; /* red */
    font-weight: bold;
    font-size: 1.2rem;
}
.recommendation-hold {
    color: #ffeb3b; /* yellow */
    font-weight: bold;
    font-size: 1.2rem;
}

/* Table styling */
table {
    color: #e0e0e0 !important;
}
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---
def convert_timestamps(obj):
    """
    Recursively convert any datetime or Timestamp objects in obj to string.
    """
    if isinstance(obj, dict):
        new_dict = {}
        for k, v in obj.items():
            if hasattr(k, "strftime"):
                k = k.strftime('%Y-%m-%d %H:%M:%S')
            new_dict[k] = convert_timestamps(v)
        return new_dict
    elif isinstance(obj, list):
        return [convert_timestamps(item) for item in obj]
    elif hasattr(obj, "strftime"):
        return obj.strftime('%Y-%m-%d %H:%M:%S')
    else:
        return obj

def load_model(model_path):
    """Load the trained model"""
    try:
        model = LongTermInvestorModel(lookback_years=5, prediction_horizon=365)
        success = model.load_model(model_path)
        return model if success else None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def get_recommendation_class(recommendation):
    if recommendation in ['Strong Buy', 'Buy']:
        return "recommendation-buy"
    elif recommendation in ['Strong Sell', 'Sell']:
        return "recommendation-sell"
    else:
        return "recommendation-hold"

def get_nifty50_stocks():
    return [
        'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
        'HINDUNILVR.NS', 'SBIN.NS', 'BAJFINANCE.NS', 'BHARTIARTL.NS', 'KOTAKBANK.NS',
        'LT.NS', 'AXISBANK.NS', 'ASIANPAINT.NS', 'MARUTI.NS', 'SUNPHARMA.NS',
        'ITC.NS', 'HCLTECH.NS', 'TITAN.NS', 'ULTRACEMCO.NS', 'ADANIENT.NS',
        'BAJAJFINSV.NS', 'ONGC.NS', 'NTPC.NS', 'POWERGRID.NS', 'TATASTEEL.NS',
        'JSWSTEEL.NS', 'WIPRO.NS', 'ADANIPORTS.NS', 'TECHM.NS', 'GRASIM.NS',
        'INDUSINDBK.NS', 'M&M.NS', 'TATAMOTORS.NS', 'CIPLA.NS', 'NESTLEIND.NS',
        'BPCL.NS', 'APOLLOHOSP.NS', 'DIVISLAB.NS', 'DRREDDY.NS', 'COALINDIA.NS',
        'EICHERMOT.NS', 'HINDALCO.NS', 'BRITANNIA.NS', 'SBILIFE.NS', 'BAJAJ-AUTO.NS',
        'HEROMOTOCO.NS', 'UPL.NS', 'HDFCLIFE.NS', 'TATACONSUM.NS', 'LTIM.NS'
    ]

def get_stock_list():
    additional_stocks = [
        'DMART.NS', 'PIDILITIND.NS', 'HAVELLS.NS', 'BERGEPAINT.NS', 'GODREJCP.NS',
        'DABUR.NS', 'MARICO.NS', 'COLPAL.NS', 'MCDOWELL-N.NS', 'JUBLFOOD.NS',
        'ICICIPRULI.NS', 'BIOCON.NS', 'LUPIN.NS', 'GLAND.NS', 'AUROPHARMA.NS',
        'INDIGO.NS', 'HDFCAMC.NS', 'NAUKRI.NS', 'INFY.NS', 'IRCTC.NS'
    ]
    all_stocks = get_nifty50_stocks() + additional_stocks
    stock_dict = {}
    for stock in all_stocks:
        display_name = stock.replace('.NS', '')
        stock_dict[f"{display_name} (NSE)"] = stock
    return stock_dict

# --- Main Application ---
def main():
    # Top Logo (Centered)
    st.image("nifty50_logo.png", width=180)
    
    # Main Header
    st.markdown("<h1 class='main-header'>Long-Term Stock Investor for Indian Market</h1>", unsafe_allow_html=True)
    
    # Load or Train Model
    model_path = 'long_term_model.pkl'
    if not os.path.exists(model_path):
        st.error("Model not found. Please train the model first.")
        if st.button("Train Model Now"):
            with st.spinner("Training model... This may take a few minutes..."):
                model = LongTermInvestorModel(lookback_years=5, prediction_horizon=365)
                data, symbol = model.load_stock_data('RELIANCE.NS')
                df = model.prepare_features(data, symbol)
                X_train, X_test, y_train, y_test = model.split_data(df)
                model.train_model(X_train, y_train)
                metrics = model.evaluate_model(X_test, y_test)
                model.save_model(model_path)
                st.success("Model trained and saved successfully! Please refresh the page.")
        return
    else:
        model = load_model(model_path)
        if model is None:
            st.error("Failed to load the model. Please check the model file.")
            return
    
    # Sidebar Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select a page",
        ["Stock Predictor", "Portfolio Analyzer", "Model Performance", "About"]
    )
    
    # --- 1) STOCK PREDICTOR PAGE ---
    if page == "Stock Predictor":
        st.markdown("<h2 class='sub-header'>Stock Prediction</h2>", unsafe_allow_html=True)
        st.markdown("<div class='info-box'>Predict stock prices for a chosen time horizon using fundamental & technical analysis.</div>", unsafe_allow_html=True)
        stock_dict = get_stock_list()
        stock_options = list(stock_dict.keys())
        selected_stock_name = st.selectbox("Select a stock", stock_options)
        selected_stock = stock_dict[selected_stock_name]
        
        horizon_options = {"1 Year": 365, "6 Months": 180, "3 Months": 90}
        chosen_horizon = st.radio("Select prediction horizon", list(horizon_options.keys()), horizontal=True)
        model.prediction_horizon = horizon_options[chosen_horizon]
        
        if st.button("Predict"):
            with st.spinner(f"Analyzing {selected_stock_name}..."):
                result = model.predict_stock(selected_stock)
                if result is None:
                    st.error(f"Failed to fetch data for {selected_stock_name}. Try another stock.")
                else:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Current Price", f"₹{result['current_price']:.2f}")
                    with col2:
                        st.metric("Predicted Price", f"₹{result['predicted_price']:.2f}", delta=f"{result['predicted_return']:.2%}")
                    with col3:
                        rec_class = get_recommendation_class(result['recommendation'])
                        st.markdown(f"<div class='metric-card'><p>Recommendation</p><p class='{rec_class}'>{result['recommendation']}</p></div>", unsafe_allow_html=True)
                    
                    st.markdown("<h3 class='sub-header'>Prediction Analysis</h3>", unsafe_allow_html=True)
                    # Simply use the returned figure directly without extra conversion.
                    fig = model.plot_prediction_analysis(result)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("<h3 class='sub-header'>Key Factors Driving Prediction</h3>", unsafe_allow_html=True)
                    factors_df = pd.DataFrame(result['top_factors'])
                    if not factors_df.empty:
                        factors_df.columns = ['Feature','Value','Impact','AbsImpact']
                        factors_df['Feature'] = factors_df['Feature'].str.replace('Fund_','').str.replace('_',' ').str.title()
                        factors_df = factors_df[['Feature','Value','Impact']]
                        st.dataframe(factors_df)
                    else:
                        st.info("No dominant factors identified.")
    
    # --- 2) PORTFOLIO ANALYZER PAGE ---
    elif page == "Portfolio Analyzer":
        st.markdown("<h2 class='sub-header'>Portfolio Analyzer</h2>", unsafe_allow_html=True)
        st.markdown("<div class='info-box'>Analyze a portfolio of stocks to see how they would perform based on our predictions.</div>", unsafe_allow_html=True)
        stock_dict = get_stock_list()
        stock_options = list(stock_dict.keys())
        selected_stock_names = st.multiselect("Select stocks for your portfolio", stock_options, 
                                              default=stock_options[:5] if len(stock_options) >= 5 else stock_options)
        if not selected_stock_names:
            st.warning("Please select at least one stock.")
        else:
            selected_stocks = [stock_dict[name] for name in selected_stock_names]
            investment = st.number_input("Initial Investment (₹)", min_value=10000, max_value=10000000, value=100000, step=10000)
            backtest_duration = st.slider("Backtest Duration (months)", min_value=1, max_value=12, value=6)
            if st.button("Run Backtest"):
                with st.spinner("Running portfolio backtest..."):
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=30*backtest_duration)
                    backtest = model.create_portfolio_backtest(selected_stocks, start_date=start_date, end_date=end_date, investment=investment)
                    if backtest:
                        st.markdown("<h3 class='sub-header'>Backtest Results</h3>", unsafe_allow_html=True)
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Final Value", f"₹{backtest['metrics']['final_value']:.2f}", delta=f"{backtest['metrics']['total_return']:.2%}")
                        with col2:
                            st.metric("Annualized Return", f"{backtest['metrics']['annualized_return']:.2%}")
                        with col3:
                            st.metric("Sharpe Ratio", f"{backtest['metrics']['sharpe_ratio']:.2f}")
                        with col4:
                            st.metric("Max Drawdown", f"{backtest['metrics']['max_drawdown']:.2%}")
                        st.markdown("<h3 class='sub-header'>Portfolio Performance</h3>", unsafe_allow_html=True)
                        fig = model.plot_backtest_results(backtest)
                        st.plotly_chart(fig, use_container_width=True)
                        st.markdown("<h3 class='sub-header'>Trade History</h3>", unsafe_allow_html=True)
                        if backtest['trades']:
                            trades_df = pd.DataFrame(backtest['trades'])
                            trades_df = trades_df[['date','action','symbol','quantity','price','value']]
                            trades_df.columns = ['Date','Action','Symbol','Quantity','Price','Value']
                            def highlight_trades(row):
                                if row['Action'] == 'BUY':
                                    return ['background-color: rgba(0,255,0,0.1)']*len(row)
                                elif row['Action'] == 'SELL':
                                    return ['background-color: rgba(255,0,0,0.1)']*len(row)
                                return ['']*len(row)
                            st.dataframe(trades_df.style.apply(highlight_trades, axis=1))
                        else:
                            st.info("No trades were executed.")
            st.markdown("<h3 class='sub-header'>Current Predictions for Selected Stocks</h3>", unsafe_allow_html=True)
            if st.button("Get Current Predictions"):
                with st.spinner("Getting predictions..."):
                    results_df = pd.DataFrame(columns=['Symbol','Current Price','Predicted Price','Expected Return','Recommendation'])
                    progress_bar = st.progress(0)
                    for i, sym in enumerate(selected_stocks):
                        progress_bar.progress((i+1)/len(selected_stocks))
                        res = model.predict_stock(sym)
                        if res:
                            results_df.loc[len(results_df)] = [
                                res['symbol'],
                                round(res['current_price'],2),
                                round(res['predicted_price'],2),
                                f"{res['predicted_return']:.2%}",
                                res['recommendation']
                            ]
                    progress_bar.empty()
                    if not results_df.empty:
                        def highlight_recommendations(val):
                            if val in ['Strong Buy','Buy']:
                                return 'background-color: rgba(0,255,0,0.2)'
                            elif val in ['Strong Sell','Sell']:
                                return 'background-color: rgba(255,0,0,0.2)'
                            return 'background-color: rgba(255,255,0,0.2)'
                        styled_df = results_df.style.applymap(highlight_recommendations, subset=['Recommendation'])
                        st.dataframe(styled_df)
                        rec_counts = results_df['Recommendation'].value_counts().reset_index()
                        rec_counts.columns = ['Recommendation','Count']
                        fig_pie = px.pie(
                            rec_counts, 
                            values='Count', 
                            names='Recommendation', 
                            title='Recommendation Distribution',
                            color='Recommendation',
                            color_discrete_map={
                                'Strong Buy': '#006400',
                                'Buy': '#32CD32',
                                'Hold': '#FFD700',
                                'Sell': '#FF6347',
                                'Strong Sell': '#8B0000'
                            }
                        )
                        st.plotly_chart(fig_pie)
                    else:
                        st.warning("Failed to retrieve predictions for the selected stocks.")
    
    # --- 3) MODEL PERFORMANCE PAGE ---
    elif page == "Model Performance":
        st.markdown("<h2 class='sub-header'>Model Performance</h2>", unsafe_allow_html=True)
        st.markdown("<div class='info-box'>See how the model performs for long-term predictions.</div>", unsafe_allow_html=True)
    
        st.markdown("<h3 class='sub-header'>Feature Importance</h3>", unsafe_allow_html=True)
        fig_imp = model.plot_feature_importance()
        if fig_imp is not None:
            st.pyplot(fig_imp, use_container_width=True)
        else:
            st.info("No feature importance available.")
    
        if hasattr(model, 'evaluation_metrics') and model.evaluation_metrics:
            st.markdown("<h3 class='sub-header'>Model Evaluation Metrics</h3>", unsafe_allow_html=True)
            metrics = model.evaluation_metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("RMSE", f"{metrics['rmse']:.4f}")
            with col2:
                st.metric("MAPE", f"{metrics['mape']:.4f}")
            with col3:
                st.metric("R² Score", f"{metrics['r2']:.4f}")
            with col4:
                st.metric("Directional Accuracy", f"{metrics['directional_accuracy']:.2%}")
            with st.expander("Explanation of Metrics"):
                st.markdown("""
                - **RMSE (Root Mean Squared Error)**: Lower is better.  
                - **MAPE (Mean Absolute Percentage Error)**: Lower is better.  
                - **R² Score**: Closer to 1 is better.  
                - **Directional Accuracy**: Percentage of correct Up/Down predictions.
                """)
    
        st.markdown("<h3 class='sub-header'>SHAP Values Analysis</h3>", unsafe_allow_html=True)
        try:
            fig_shap = model.plot_shap_values()
            if fig_shap is not None:
                st.pyplot(fig_shap, use_container_width=True)
            else:
                st.info("No SHAP values available.")
        except Exception as e:
            st.error(f"Error generating SHAP plot: {e}")
    
        # ─── New: Data Drift Detection ─────────────────────────────────────
        st.markdown("<h3 class='sub-header'>Data Drift Detection (PSI)</h3>", unsafe_allow_html=True)
        if hasattr(model, 'detect_data_drift') and model.test_data is not None:
            # compute PSI on your test dataset vs. training distribution
            drift_series = model.detect_data_drift(model.test_data)
            st.table(drift_series.rename("PSI Score"))
        else:
            st.info("Data-drift detection not Observed.")
    
        st.markdown("<h3 class='sub-header'>Detailed Analysis of an Example Stock</h3>", unsafe_allow_html=True)
        example_symbol = "RELIANCE.NS"
        example_res = model.predict_stock(example_symbol)
        
        if example_res:
            fig_pred = model.plot_prediction_analysis(example_res)
            if fig_pred is not None:
                st.plotly_chart(fig_pred, use_container_width=True)
        else:
            st.warning(f"Could not fetch data for {example_symbol}.")       
        
    # --- 4) ABOUT PAGE ---
    else:
        ...
        st.markdown("<h2 class='sub-header'>About This Model</h2>", unsafe_allow_html=True)
        st.markdown("""
        ### Model Overview

        This application implements a long-term stock forecasting model for the Indian market, 
        focusing on Nifty 50 companies. It uses a combination of:
        - **Fundamental Analysis** (company financial metrics)
        - **Technical Indicators** (historical price patterns)
        - **Sector Performance** (relative sector strength)
        - **Explainable AI** (SHAP values to interpret predictions)

        ### Limitations
        - Does not capture sudden, unforeseen events.
        - Based on historical data patterns; no guarantee of future performance.
        - Should be used as a tool alongside other research.

        ### Project Information
        **ISB AMPBA Batch No. 22 - Summer 2025 Foundational Project**  
        **Professor:** Bharani Kumar Depuru  
        **TA:** K. Mohan  

        **Group Members:**  
        - **Kanishk Rana** — Kanishk_Rana_ampba2025S@isb.edu  
        - **Ashish Jonathan Dcruze** — Ashish_Dcruze_ampba2025S@isb.edu  
        - **Priyesh Jagtap** — Priyesh_Jagtap_ampba2025S@isb.edu  
        - **Suraj Dwivedi** — Suraj_Dwivedi_ampba2025S@isb.edu

        ### Contact
        For questions or feedback, please contact [ampba_office@isb.edu](mailto:ampba_office@isb.edu).
        """, unsafe_allow_html=True)
        st.sidebar.markdown("---")
        st.sidebar.info(f"**Version**: 1.0.0\n**Last Updated**: {datetime.now().strftime('%Y-%m-%d')}")
    
if __name__ == "__main__":
    main()
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from utils import calculate_metrics, get_recommendation

# Page config
st.set_page_config(
    page_title="Stock Analysis Dashboard",
    page_icon="📈",
    layout="wide"
)

# Load custom CSS
with open('assets/style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Title
st.title('📈 Stock Analysis Dashboard')

# Input section
col1, col2 = st.columns([2, 1])
with col1:
    symbol = st.text_input('Enter Stock Symbol (e.g., AAPL):', value='AAPL')
with col2:
    period = st.selectbox('Select Time Period:', 
                         ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y'])

# Fetch data
@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_stock_data(symbol, period):
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period, interval="1d")
        info = stock.info
        return hist, info, True
    except:
        return None, None, False

hist_data, stock_info, success = get_stock_data(symbol, period)

if not success:
    st.error('Error fetching data. Please check the stock symbol and try again.')
    st.stop()

# Current price and basic info
if stock_info and hist_data is not None:
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        current_price = hist_data['Close'].iloc[-1]
        st.metric("Current Price", 
                 f"${current_price:.2f}", 
                 f"{((current_price - hist_data['Close'].iloc[-2])/hist_data['Close'].iloc[-2]*100):.2f}%")
    
    with col2:
        st.metric("Market Cap", 
                 f"${stock_info.get('marketCap', 0)/1e9:.2f}B")
    
    with col3:
        st.metric("Volume", 
                 f"{stock_info.get('volume', 0):,}")
    
    with col4:
        st.metric("52 Week High", 
                 f"${stock_info.get('fiftyTwoWeekHigh', 0):.2f}")

    # Price Chart
    st.subheader('Price History')
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=hist_data.index,
                                open=hist_data['Open'],
                                high=hist_data['High'],
                                low=hist_data['Low'],
                                close=hist_data['Close'],
                                name='OHLC'))
    fig.update_layout(
        title=f'{symbol} Stock Price',
        yaxis_title='Stock Price (USD)',
        xaxis_title='Date',
        template='plotly_white'
    )
    st.plotly_chart(fig, use_container_width=True)

    # Metrics and Analysis
    metrics_df = calculate_metrics(hist_data)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader('Key Metrics')
        st.dataframe(metrics_df, use_container_width=True)
        
        # Download button for CSV
        csv = metrics_df.to_csv(index=True)
        st.download_button(
            label="Download Data as CSV",
            data=csv,
            file_name=f'{symbol}_metrics.csv',
            mime='text/csv',
        )
    
    with col2:
        st.subheader('Trading Recommendation')
        recommendation, confidence = get_recommendation(hist_data)
        
        # Display recommendation with color coding
        if recommendation == "Buy":
            st.markdown(f'<div class="recommendation buy">BUY ({confidence}% confidence)</div>', 
                       unsafe_allow_html=True)
        elif recommendation == "Sell":
            st.markdown(f'<div class="recommendation sell">SELL ({confidence}% confidence)</div>', 
                       unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="recommendation hold">HOLD ({confidence}% confidence)</div>', 
                       unsafe_allow_html=True)

    # Additional Information
    with st.expander("Company Information"):
        st.write(f"**Company Name:** {stock_info.get('longName', 'N/A')}")
        st.write(f"**Sector:** {stock_info.get('sector', 'N/A')}")
        st.write(f"**Industry:** {stock_info.get('industry', 'N/A')}")
        st.write(f"**Website:** {stock_info.get('website', 'N/A')}")
        st.write(f"**Business Summary:** {stock_info.get('longBusinessSummary', 'N/A')}")
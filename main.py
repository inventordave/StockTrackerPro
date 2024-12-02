import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from utils import calculate_metrics, get_recommendation
from trading import trading_service
from practice_mode import PracticePortfolio

# Initialize practice portfolio in session state if not exists
if 'practice_portfolio' not in st.session_state:
    st.session_state.practice_portfolio = PracticePortfolio()

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
    symbols_input = st.text_input('Enter Stock Symbols (comma-separated, e.g., AAPL,MSFT):', value='AAPL')
    symbols = [sym.strip().upper() for sym in symbols_input.split(',') if sym.strip()]
with col2:
    period = st.selectbox('Select Time Period:', 
                         ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y'])

# Validate input
if not symbols:
    st.error('Please enter at least one stock symbol.')
    st.stop()
if len(symbols) > 5:
    st.warning('Maximum 5 stocks can be compared at once. Only first 5 will be shown.')
    symbols = symbols[:5]

# Fetch data
@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_stock_data(symbols, period):
    data = {}
    for symbol in symbols:
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(period=period, interval="1d")
            info = stock.info
            data[symbol] = {'history': hist, 'info': info}
        except Exception as e:
            st.error(f'Error fetching data for {symbol}: {str(e)}')
            return None
    return data if data else None

stock_data = get_stock_data(symbols, period)

if stock_data is None:
    st.error('Error fetching data. Please check the stock symbols and try again.')
    st.stop()

# Current price and basic info
if stock_data:
    # Create metrics for each stock
    for symbol in symbols:
        st.subheader(f'{symbol} Metrics')
        hist_data = stock_data[symbol]['history']
        stock_info = stock_data[symbol]['info']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            current_price = hist_data['Close'].iloc[-1]
            if len(hist_data) >= 2:
                prev_price = hist_data['Close'].iloc[-2]
                price_change = ((current_price - prev_price)/prev_price*100)
                change_str = f"{price_change:.2f}%"
            else:
                change_str = "N/A"
            
            st.metric("Current Price", 
                     f"${current_price:.2f}", 
                     change_str)
        
        with col2:
            st.metric("Market Cap", 
                     f"${stock_info.get('marketCap', 0)/1e9:.2f}B")
        
        with col3:
            st.metric("Volume", 
                     f"{stock_info.get('volume', 0):,}")
        
        with col4:
            st.metric("52 Week High", 
                     f"${stock_info.get('fiftyTwoWeekHigh', 0):.2f}")

    # Comparison Charts
    st.subheader('Price Comparison')
    
    # Line chart for price comparison
    fig_comparison = go.Figure()
    for symbol in symbols:
        hist_data = stock_data[symbol]['history']
        normalized_price = (hist_data['Close'] / hist_data['Close'].iloc[0]) * 100
        fig_comparison.add_trace(go.Scatter(
            x=hist_data.index,
            y=normalized_price,
            name=symbol,
            mode='lines'
        ))
    
    fig_comparison.update_layout(
        title='Normalized Price Comparison (Base 100)',
        yaxis_title='Normalized Price',
        xaxis_title='Date',
        template='plotly_white'
    )
    st.plotly_chart(fig_comparison, use_container_width=True)
    
    # Individual stock charts
    for symbol in symbols:
        hist_data = stock_data[symbol]['history']
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=hist_data.index,
            open=hist_data['Open'],
            high=hist_data['High'],
            low=hist_data['Low'],
            close=hist_data['Close'],
            name='OHLC'
        ))
        fig.update_layout(
            title=f'{symbol} Stock Price',
            yaxis_title='Stock Price (USD)',
            xaxis_title='Date',
            template='plotly_white',
            showlegend=False
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
            
        # Trading Interface
        st.subheader("Trading Interface")
        
        # Trading platform selection (moved outside practice mode condition)
        platform = st.selectbox(
            "Select Trading Platform",
            ["Alpaca", "Interactive Brokers"],
            key="trading_platform"
        )
        
        # Practice Mode Toggle
        practice_mode = st.toggle("Practice Mode", help="Enable practice trading with virtual money")
        
        if practice_mode:
            st.info("🎓 Practice Mode Enabled - Trading with virtual portfolio")
            
            # Show practice portfolio metrics
            portfolio = st.session_state.practice_portfolio
            current_prices = {symbol: stock_data[symbol]['history']['Close'].iloc[-1] 
                            for symbol in stock_data.keys()}
            
            metrics = portfolio.get_performance_metrics(current_prices)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Portfolio Value", f"${metrics['total_value']:,.2f}")
            with col2:
                st.metric("Cash Balance", f"${metrics['cash_balance']:,.2f}")
            with col3:
                st.metric("Total Return", f"{metrics['total_return']:.2f}%")
            with col4:
                st.metric("Win Ratio", f"{metrics['win_ratio']*100:.1f}%")
            
            # Portfolio Performance Chart
            if metrics['total_trades'] > 0:
                st.subheader("Portfolio Performance")
                trade_df = pd.DataFrame([
                    {
                        'Date': trade.date,
                        'Symbol': trade.symbol,
                        'Side': trade.side,
                        'Quantity': trade.quantity,
                        'Price': trade.price,
                        'PnL': trade.pnl if trade.pnl is not None else 0
                    }
                    for trade in portfolio.trade_history
                ])
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=trade_df['Date'],
                    y=trade_df['PnL'].cumsum(),
                    mode='lines',
                    name='Cumulative P&L'
                ))
                fig.update_layout(
                    title='Portfolio P&L Over Time',
                    xaxis_title='Date',
                    yaxis_title='Cumulative P&L ($)',
                    template='plotly_white'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Show best and worst trades
                if metrics['best_trade']:
                    st.write(f"Best Trade: {metrics['best_trade'].symbol} - ${metrics['best_trade'].pnl:,.2f}")
                if metrics['worst_trade']:
                    st.write(f"Worst Trade: {metrics['worst_trade'].symbol} - ${metrics['worst_trade'].pnl:,.2f}")
        
        # API Credentials
        with st.expander("Trading Platform Credentials"):
            api_key = st.text_input("API Key", type="password", key="api_key")
            api_secret = st.text_input("API Secret", type="password", key="api_secret")
            
            if platform == "Alpaca":
                use_paper = st.checkbox("Use Paper Trading", value=True)
                base_url = "https://paper-api.alpaca.markets" if use_paper else "https://api.alpaca.markets"
            
            # Test connection button
            if st.button("Test Connection"):
                if not api_key or not api_secret:
                    st.error("Please enter both API key and secret.")
                else:
                    credentials = {
                        "api_key": api_key,
                        "api_secret": api_secret,
                        "base_url": base_url if platform == "Alpaca" else None
                    }
                    success, message = trading_service.validate_connection(platform, credentials)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
        
        # Trading controls
        col1, col2, col3 = st.columns(3)
        with col1:
            quantity = st.number_input("Quantity", min_value=1, value=1)
        with col2:
            order_type = st.selectbox("Order Type", ["Market", "Limit"])
        with col3:
            if order_type == "Limit":
                limit_price = st.number_input("Limit Price", min_value=0.01, value=float(current_price))
        
        # Trade execution section
        execute_trade = st.button("Execute Trade")
        if execute_trade:
            try:
                if practice_mode:
                    # Execute practice trade
                    price = current_price if order_type == "Market" else limit_price
                    if price <= 0:
                        st.error("Invalid price. Please check the order parameters.")
                    elif quantity <= 0:
                        st.error("Invalid quantity. Please enter a positive number.")
                    else:
                        success, message = st.session_state.practice_portfolio.execute_trade(
                            symbol=symbol,
                            quantity=quantity,
                            price=price,
                            side=recommendation.lower()
                        )
                        if success:
                            st.success(f"Practice trade executed: {recommendation} {quantity} shares of {symbol} at ${price:.2f}")
                        else:
                            st.error(f"Trade failed: {message}")
                else:
                    # Real trading execution
                    if not api_key or not api_secret:
                        st.error("Please enter API credentials first.")
                    else:
                        credentials = {
                            "api_key": api_key,
                            "api_secret": api_secret,
                            "base_url": base_url if platform == "Alpaca" else None
                        }
                        success, message = trading_service.execute_trade(
                            platform=platform,
                            credentials=credentials,
                            symbol=symbol,
                            quantity=quantity,
                            order_type=order_type,
                            side=recommendation,
                            limit_price=limit_price if order_type == "Limit" else None
                        )
                        if success:
                            st.success(message)
                        else:
                            st.error(message)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

    # Additional Information
    with st.expander("Company Information"):
        st.write(f"**Company Name:** {stock_info.get('longName', 'N/A')}")
        st.write(f"**Sector:** {stock_info.get('sector', 'N/A')}")
        st.write(f"**Industry:** {stock_info.get('industry', 'N/A')}")
        st.write(f"**Website:** {stock_info.get('website', 'N/A')}")
        st.write(f"**Business Summary:** {stock_info.get('longBusinessSummary', 'N/A')}")

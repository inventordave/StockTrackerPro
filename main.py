import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os
import os
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
import logging
import time
from functools import wraps
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure retry decorator
def with_retry(max_attempts=3, initial_wait=1):
    def decorator(func):
        @retry(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(multiplier=initial_wait, min=initial_wait, max=10),
            reraise=True
        )
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {str(e)}")
                raise
        return wrapper
    return decorator

# Error recovery and connection management
class ConnectionManager:
    def __init__(self):
        self.retry_count = 0
        self.max_retries = 3
        self.base_delay = 1
        
    def handle_connection_error(self, error):
        if self.retry_count < self.max_retries:
            delay = self.base_delay * (2 ** self.retry_count)
            logger.warning(f"Connection error: {str(error)}. Retrying in {delay} seconds...")
            time.sleep(delay)
            self.retry_count += 1
            return True
        return False

connection_manager = ConnectionManager()
from utils import calculate_metrics, get_recommendation, calculate_comparison_metrics
from trading import trading_service
from practice_mode import PracticePortfolio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config before any other Streamlit commands
st.set_page_config(
    page_title="Stock Analysis Dashboard",
    page_icon="📈",
    layout="wide"
)

# Initialize practice portfolio in session state if not exists
if 'practice_portfolio' not in st.session_state:
    st.session_state.practice_portfolio = PracticePortfolio()

# Load custom CSS with error handling
try:
    css_path = 'assets/style.css'
    if not os.path.exists(css_path):
        logger.error(f"CSS file not found: {css_path}")
        st.error("Error loading style sheet. Using default styling.")
    else:
        with open(css_path) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
except Exception as e:
    logger.error(f"Error loading CSS: {str(e)}")
    st.warning("Error loading custom styles. Using default styling.")

# Title and Description
st.title('📈 Stock Analysis Dashboard')
st.markdown("""
This dashboard allows you to compare multiple stocks and analyze their performance metrics.
Enter up to 5 stock symbols separated by commas (e.g., AAPL,MSFT,GOOGL).
""")

# Input section with improved layout
col1, col2 = st.columns([2, 1])
with col1:
    symbols_input = st.text_input(
        'Enter Stock Symbols:',
        value='AAPL',
        help='Enter up to 5 comma-separated stock symbols (e.g., AAPL,MSFT,GOOGL)'
    )
    # Improved symbol parsing with validation
    try:
        symbols = [sym.strip().upper() for sym in symbols_input.split(',') if sym.strip()]
        if not symbols:
            st.error('Please enter at least one valid stock symbol.')
            st.stop()
        elif len(symbols) > 5:
            st.warning('Maximum 5 stocks can be compared at once. Only first 5 will be shown.')
            symbols = symbols[:5]
        
        # Validate symbols format
        invalid_symbols = [sym for sym in symbols if not sym.isalpha() or len(sym) > 5]
        if invalid_symbols:
            st.error(f'Invalid symbol format: {", ".join(invalid_symbols)}. Please use valid stock symbols.')
            st.stop()
    except Exception as e:
        st.error(f'Error processing symbols: {str(e)}')
        st.stop()
        
with col2:
    period = st.selectbox(
        'Select Time Period:', 
        ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y'],
        help='Choose the time period for historical data analysis',
        key='time_period'
    )

# Validate input
if not symbols:
    st.error('Please enter at least one stock symbol.')
    st.stop()
if len(symbols) > 5:
    st.warning('Maximum 5 stocks can be compared at once. Only first 5 will be shown.')
    symbols = symbols[:5]

# Fetch data
@st.cache_data(ttl=300)  # Cache for 5 minutes
@with_retry(max_attempts=3, initial_wait=1)
def get_stock_data(symbols, period):
    """Fetch stock data with improved error handling and WebSocket management"""
    try:
        # Configure WebSocket connection and retry parameters
        st.session_state.ws_retry_count = getattr(st.session_state, 'ws_retry_count', 0)
        max_ws_retries = 3
        
        # Initialize WebSocket connection if needed
        if getattr(st.session_state, 'ws_connection', None) is None:
            try:
                st.session_state.ws_connection = True  # Placeholder for actual WebSocket connection
                logger.info("WebSocket connection initialized successfully")
            except Exception as ws_e:
                st.error(f"WebSocket connection error: {str(ws_e)}")
                logger.error(f"WebSocket error: {str(ws_e)}")
                raise
    data = {}
    errors = []
    connection_manager.retry_count = 0
    
    if not symbols:
        st.error("No symbols provided")
        return None
        
    for symbol in symbols:
        try:
            with st.spinner(f'Fetching data for {symbol}...'):
                stock = yf.Ticker(symbol)
                
                # Fetch history with timeout and error handling
                try:
                    for attempt in range(3):
                        try:
                            hist = stock.history(period=period, interval="1d", timeout=10)
                            if hist.empty:
                                if connection_manager.handle_connection_error("Empty data received"):
                                    continue
                                errors.append(f"No historical data available for {symbol}")
                                break
                            
                            if len(hist) < 20:
                                errors.append(f"Insufficient historical data for {symbol} (minimum 20 days required)")
                                break
                            
                            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                            missing_columns = [col for col in required_columns if col not in hist.columns]
                            if missing_columns:
                                errors.append(f"Missing required data columns for {symbol}: {', '.join(missing_columns)}")
                                break
                                
                            # If we got here, the data is valid
                            break
                        except Exception as e:
                            if not connection_manager.handle_connection_error(e):
                                errors.append(f"Error fetching historical data for {symbol}: {str(e)}")
                                break
                except Exception as e:
                    errors.append(f"Error fetching historical data for {symbol}: {str(e)}")
                    continue
                    
                # Fetch info with timeout and error handling
                try:
                    info = stock.info
                    if not info:
                        errors.append(f"No information available for {symbol}")
                        continue
                        
                    # Validate required info fields
                    required_info = ['longName', 'sector', 'industry']
                    for field in required_info:
                        if field not in info:
                            info[field] = 'N/A'
                            
                except Exception as e:
                    errors.append(f"Error fetching information for {symbol}: {str(e)}")
                    continue
                    
                # Process and clean the data
                hist = hist.fillna(method='ffill').fillna(method='bfill')  # Handle any NaN values
                data[symbol] = {'history': hist, 'info': info}
                
        except Exception as e:
            errors.append(f"Error processing {symbol}: {str(e)}")
    
    # Handle errors
    if errors:
        error_message = "\n".join(errors)
        if not data:  # If no valid data at all
            st.error(f"Failed to fetch any valid data:\n{error_message}")
            return None
        else:  # If some symbols were successful
            st.warning(f"Some symbols had errors:\n{error_message}")
    
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

    # Stock Comparison Analysis
    st.header('Stock Comparison Analysis')
    
    try:
        # Calculate comparison metrics with error handling
        comparison_data = calculate_comparison_metrics(stock_data)
        
        if comparison_data['correlation'].empty:
            st.error("Unable to calculate comparison metrics. Please check the input data.")
            st.stop()
        
        # Correlation Heatmap
        st.subheader('Price Correlation Matrix')
        if not comparison_data['correlation'].empty:
            fig_correlation = go.Figure(data=go.Heatmap(
                z=comparison_data['correlation'],
                x=symbols,
                y=symbols,
                colorscale='RdBu',
                zmin=-1,
                zmax=1
            ))
            fig_correlation.update_layout(
                title='Stock Price Correlation',
                template='plotly_white'
            )
            st.plotly_chart(fig_correlation, use_container_width=True)
        else:
            st.warning("Unable to generate correlation heatmap due to insufficient data.")
    
    try:
        # Normalized Price Comparison
        st.subheader('Relative Performance')
        if not comparison_data['normalized_prices'].empty:
            fig_comparison = go.Figure()
            for symbol in symbols:
                if symbol in comparison_data['normalized_prices'].columns:
                    normalized_data = comparison_data['normalized_prices'][symbol]
                    if not normalized_data.isna().all():
                        fig_comparison.add_trace(go.Scatter(
                            x=comparison_data['normalized_prices'].index,
                            y=normalized_data * 100,
                            name=symbol,
                            mode='lines'
                        ))
            
            if len(fig_comparison.data) > 0:
                fig_comparison.update_layout(
                    title='Normalized Price Comparison (Base 100)',
                    yaxis_title='Normalized Price',
                    xaxis_title='Date',
                    template='plotly_white'
                )
                st.plotly_chart(fig_comparison, use_container_width=True)
            else:
                st.warning("Unable to generate relative performance chart due to invalid data.")
        else:
            st.warning("No valid data available for relative performance comparison.")
        
        # Volume Comparison
        st.subheader('Volume Analysis')
        if not comparison_data['volume_ratio'].empty:
            fig_volume = go.Figure()
            for symbol in symbols:
                if symbol in comparison_data['volume_ratio'].columns:
                    volume_data = comparison_data['volume_ratio'][symbol]
                    if not volume_data.isna().all():
                        fig_volume.add_trace(go.Scatter(
                            x=comparison_data['volume_ratio'].index,
                            y=volume_data,
                            name=symbol,
                            mode='lines'
                        ))
            
            if len(fig_volume.data) > 0:
                fig_volume.update_layout(
                    title='Volume Ratio (Current/20-day Average)',
                    yaxis_title='Volume Ratio',
                    xaxis_title='Date',
                    template='plotly_white'
                )
                st.plotly_chart(fig_volume, use_container_width=True)
            else:
                st.warning("Unable to generate volume analysis chart due to invalid data.")
        else:
            st.warning("No valid data available for volume analysis.")
            
    except Exception as e:
        st.error(f"Error generating comparison charts: {str(e)}")
    
    # Beta Analysis
    if len(symbols) > 1:
        st.subheader('Beta Analysis')
        market_symbol = symbols[0]
        st.write(f"Beta values relative to {market_symbol}:")
        for symbol, beta in comparison_data['betas'].items():
            st.metric(f"{symbol} Beta", f"{beta:.2f}")
    
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
            
            # Show practice portfolio metrics with error handling
            try:
                portfolio = st.session_state.practice_portfolio
                current_prices = {}
                try:
                    for symbol in stock_data.keys():
                        if 'history' in stock_data[symbol] and not stock_data[symbol]['history'].empty:
                            current_prices[symbol] = stock_data[symbol]['history']['Close'].iloc[-1]
                        else:
                            logger.warning(f"Missing or empty historical data for {symbol}")
                except Exception as e:
                    logger.error(f"Error getting current prices: {str(e)}")
                    st.warning("Some price data could not be loaded. Portfolio metrics may be incomplete.")
                
                metrics = portfolio.get_performance_metrics(current_prices)
            except Exception as e:
                logger.error(f"Error calculating portfolio metrics: {str(e)}")
                st.error("Unable to calculate portfolio metrics. Please try again later.")
                metrics = {
                    'total_value': 0,
                    'cash_balance': 0,
                    'total_return': 0,
                    'win_ratio': 0,
                    'total_trades': 0,
                    'best_trade': None,
                    'worst_trade': None
                }
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Portfolio Value", f"${metrics['total_value']:,.2f}")
            with col2:
                st.metric("Cash Balance", f"${metrics['cash_balance']:,.2f}")
            with col3:
                st.metric("Total Return", f"{metrics['total_return']:.2f}%")
            with col4:
                st.metric("Win Ratio", f"{metrics['win_ratio']*100:.1f}%")
            
            # Portfolio Performance Chart with error handling
            if metrics['total_trades'] > 0:
                try:
                    st.subheader("Portfolio Performance")
                    # Create trade history dataframe with error handling
                    try:
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
                    except Exception as e:
                        logger.error(f"Error creating trade history dataframe: {str(e)}")
                        st.error("Unable to process trade history data.")
                        trade_df = pd.DataFrame()

                    if not trade_df.empty:
                        try:
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
                        except Exception as e:
                            logger.error(f"Error generating performance chart: {str(e)}")
                            st.error("Unable to generate performance chart.")
                except Exception as e:
                    logger.error(f"Error in portfolio performance visualization: {str(e)}")
                    st.error("Unable to display portfolio performance.")
                
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
                    try:
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
                    except Exception as e:
                        st.error(f"Practice trade execution error: {str(e)}")
                        logger.error(f"Practice trade error: {str(e)}")
                else:
                    try:
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
                        st.error(f"Real trade execution error: {str(e)}")
                        logger.error(f"Real trade error: {str(e)}")
            except Exception as e:
                st.error(f"Unexpected error in trade execution: {str(e)}")
                logger.error(f"Critical trade execution error: {str(e)}")

    # Additional Information
    with st.expander("Company Information"):
        st.write(f"**Company Name:** {stock_info.get('longName', 'N/A')}")
        st.write(f"**Sector:** {stock_info.get('sector', 'N/A')}")
        st.write(f"**Industry:** {stock_info.get('industry', 'N/A')}")
        st.write(f"**Website:** {stock_info.get('website', 'N/A')}")
        st.write(f"**Business Summary:** {stock_info.get('longBusinessSummary', 'N/A')}")

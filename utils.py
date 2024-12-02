import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def calculate_metrics(hist_data):
    """Calculate key financial metrics from historical data."""
    metrics = pd.DataFrame()
    
    # Basic price metrics
    metrics['Close'] = hist_data['Close']
    metrics['Daily Return'] = hist_data['Close'].pct_change()
    metrics['Rolling Volatility'] = metrics['Daily Return'].rolling(window=20).std() * np.sqrt(252)
    metrics['YTD Return'] = (hist_data['Close'] / hist_data['Close'].iloc[0] - 1) * 100
    
    # Volume analysis
    metrics['Volume'] = hist_data['Volume']
    metrics['Avg Volume (20d)'] = hist_data['Volume'].rolling(window=20).mean()
    metrics['Volume Ratio'] = hist_data['Volume'] / metrics['Avg Volume (20d)']
    
    # Moving averages
    metrics['SMA_20'] = hist_data['Close'].rolling(window=20).mean()
    metrics['SMA_50'] = hist_data['Close'].rolling(window=50).mean()
    metrics['Price vs SMA_20'] = (hist_data['Close'] / metrics['SMA_20'] - 1) * 100
    
    # Technical indicators
    # RSI
    delta = hist_data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    metrics['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = hist_data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = hist_data['Close'].ewm(span=26, adjust=False).mean()
    metrics['MACD'] = exp1 - exp2
    metrics['Signal Line'] = metrics['MACD'].ewm(span=9, adjust=False).mean()
    
    # Performance metrics
    metrics['Max Drawdown'] = (hist_data['Close'] / hist_data['Close'].expanding(min_periods=1).max() - 1) * 100
    
    # Format and clean
    metrics = metrics.round(2)
    metrics.index = metrics.index.strftime('%Y-%m-%d')
    
    return metrics

def calculate_comparison_metrics(stock_data):
    """Calculate comparison metrics between multiple stocks with input validation and error handling."""
    try:
        # Input validation
        if not stock_data or not isinstance(stock_data, dict):
            raise ValueError("Invalid input: stock_data must be a non-empty dictionary")
        
        if len(stock_data) < 2:
            raise ValueError("At least two stocks are required for comparison")
            
        comparison = {}
        closes = pd.DataFrame()
        volumes = pd.DataFrame()
        returns = pd.DataFrame()
        
        # Collect data with validation
        for symbol, data in stock_data.items():
            if 'history' not in data:
                raise KeyError(f"Missing historical data for {symbol}")
                
            hist = data['history']
            
            # Validate required columns
            required_columns = ['Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in hist.columns]
            if missing_columns:
                raise KeyError(f"Missing required columns for {symbol}: {', '.join(missing_columns)}")
            
            # Check for empty dataframes
            if hist.empty:
                raise ValueError(f"Empty historical data for {symbol}")
            
            # Check for sufficient data points (at least 20 days for meaningful analysis)
            if len(hist) < 20:
                raise ValueError(f"Insufficient data points for {symbol}. Minimum 20 days required.")
            
            # Handle missing values in Close prices
            if hist['Close'].isnull().any():
                # Forward fill, then backward fill any remaining NaNs
                hist['Close'] = hist['Close'].ffill().bfill()
                
            closes[symbol] = hist['Close']
            volumes[symbol] = hist['Volume'].fillna(0)  # Replace NaN volumes with 0
            returns[symbol] = hist['Close'].pct_change().fillna(0)
        
        # Calculate correlation matrix with error checking
        try:
            correlation_matrix = returns.corr()
            if correlation_matrix.isnull().any().any():
                # If any NaN in correlation, use spearman correlation as fallback
                correlation_matrix = returns.corr(method='spearman')
        except Exception as e:
            st.warning(f"Error in correlation calculation, using simplified method: {str(e)}")
            correlation_matrix = returns.corr(method='spearman')
        
        # Calculate relative performance with error handling
        try:
            first_day_prices = closes.iloc[0]
            if (first_day_prices == 0).any():
                raise ValueError("Zero prices detected in first day data")
            normalized_prices = closes / first_day_prices
        except Exception as e:
            st.warning(f"Error in price normalization, using alternative method: {str(e)}")
            # Alternative normalization using percentage changes
            normalized_prices = (1 + returns).cumprod()
        
        # Calculate beta with error handling
        betas = {}
        market_symbol = list(stock_data.keys())[0]
        market_returns = returns[market_symbol]
        
        if market_returns.var() == 0:
            st.warning("Market returns show no variance, beta calculations may be unreliable")
        
        for symbol in stock_data.keys():
            if symbol != market_symbol:
                try:
                    stock_returns = returns[symbol]
                    # Use rolling beta calculation for more stability
                    rolling_cov = stock_returns.rolling(window=30).cov(market_returns)
                    rolling_var = market_returns.rolling(window=30).var()
                    beta = rolling_cov.mean() / rolling_var.mean()
                    betas[symbol] = round(float(beta), 2)
                except Exception as e:
                    st.warning(f"Error calculating beta for {symbol}: {str(e)}")
                    betas[symbol] = None
        
        # Calculate volume ratio with error handling
        try:
            volume_ratio = volumes / volumes.rolling(window=20).mean()
            volume_ratio = volume_ratio.fillna(1)  # Fill NaN with 1 (neutral ratio)
        except Exception as e:
            st.warning(f"Error calculating volume ratios: {str(e)}")
            volume_ratio = pd.DataFrame(1, index=volumes.index, columns=volumes.columns)
        
        # Compile metrics
        comparison['correlation'] = correlation_matrix
        comparison['normalized_prices'] = normalized_prices
        comparison['betas'] = betas
        comparison['volume_ratio'] = volume_ratio
        
        return comparison
        
    except Exception as e:
        st.error(f"Error in comparison metrics calculation: {str(e)}")
        # Return fallback empty comparison structure
        return {
            'correlation': pd.DataFrame(),
            'normalized_prices': pd.DataFrame(),
            'betas': {},
            'volume_ratio': pd.DataFrame()
        }

def get_recommendation(hist_data):
    """Generate trading recommendation based on technical analysis."""
    # Calculate short-term and long-term momentum
    returns_short = hist_data['Close'].pct_change(periods=5)
    returns_long = hist_data['Close'].pct_change(periods=20)
    
    # Calculate RSI
    delta = hist_data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # Get latest values
    current_rsi = rsi.iloc[-1]
    short_momentum = returns_short.iloc[-1]
    long_momentum = returns_long.iloc[-1]
    
    # Decision logic
    score = 0
    max_score = 3
    
    # RSI signals
    if current_rsi < 30:
        score += 1  # Oversold - bullish
    elif current_rsi > 70:
        score -= 1  # Overbought - bearish
        
    # Momentum signals
    if short_momentum > 0 and long_momentum > 0:
        score += 1  # Positive momentum - bullish
    elif short_momentum < 0 and long_momentum < 0:
        score -= 1  # Negative momentum - bearish
        
    # Moving average signals
    sma_20 = hist_data['Close'].rolling(window=20).mean()
    sma_50 = hist_data['Close'].rolling(window=50).mean()
    
    if sma_20.iloc[-1] > sma_50.iloc[-1]:
        score += 1  # Golden cross - bullish
    else:
        score -= 1  # Death cross - bearish
        
    # Calculate confidence
    confidence = abs(score / max_score * 100)
    
    # Generate recommendation
    if score > 0:
        return "Buy", round(confidence)
    elif score < 0:
        return "Sell", round(confidence)
    else:
        return "Hold", round(confidence)

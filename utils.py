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
    """Calculate comparison metrics between multiple stocks."""
    comparison = {}
    
    # Get all closing prices in a single dataframe
    closes = pd.DataFrame()
    volumes = pd.DataFrame()
    returns = pd.DataFrame()
    
    for symbol, data in stock_data.items():
        hist = data['history']
        closes[symbol] = hist['Close']
        volumes[symbol] = hist['Volume']
        returns[symbol] = hist['Close'].pct_change()
    
    # Calculate correlation matrix
    correlation_matrix = returns.corr()
    
    # Calculate relative performance
    first_day_prices = closes.iloc[0]
    normalized_prices = closes / first_day_prices
    
    # Calculate beta (using first stock as market proxy)
    market_symbol = list(stock_data.keys())[0]
    market_returns = returns[market_symbol]
    betas = {}
    
    for symbol in stock_data.keys():
        if symbol != market_symbol:
            stock_returns = returns[symbol]
            beta = (stock_returns.cov(market_returns) / market_returns.var())
            betas[symbol] = round(beta, 2)
    
    # Compile metrics
    comparison['correlation'] = correlation_matrix
    comparison['normalized_prices'] = normalized_prices
    comparison['betas'] = betas
    comparison['volume_ratio'] = volumes / volumes.rolling(window=20).mean()
    
    return comparison

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

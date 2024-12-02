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
    
    # Moving averages
    metrics['SMA_20'] = hist_data['Close'].rolling(window=20).mean()
    metrics['SMA_50'] = hist_data['Close'].rolling(window=50).mean()
    
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
    
    # Format and clean
    metrics = metrics.round(2)
    metrics.index = metrics.index.strftime('%Y-%m-%d')
    
    return metrics

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

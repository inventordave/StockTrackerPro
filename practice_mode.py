from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd

@dataclass
class Position:
    symbol: str
    quantity: int
    entry_price: float
    entry_date: datetime

@dataclass
class Trade:
    symbol: str
    quantity: int
    price: float
    side: str  # 'buy' or 'sell'
    date: datetime
    pnl: Optional[float] = None

class PracticePortfolio:
    def __init__(self, initial_balance: float = 100000.0):
        self.initial_balance = initial_balance
        self.cash_balance = initial_balance
        self.positions: Dict[str, Position] = {}
        self.trade_history: List[Trade] = []
        
    def execute_trade(self, symbol: str, quantity: int, price: float, side: str) -> bool:
        """Execute a practice trade and update portfolio"""
        trade_value = quantity * price
        
        if side.lower() == 'buy':
            if trade_value > self.cash_balance:
                return False
            
            self.cash_balance -= trade_value
            if symbol in self.positions:
                # Update existing position
                avg_price = (
                    (self.positions[symbol].quantity * self.positions[symbol].entry_price) +
                    (quantity * price)
                ) / (self.positions[symbol].quantity + quantity)
                self.positions[symbol].quantity += quantity
                self.positions[symbol].entry_price = avg_price
            else:
                # Create new position
                self.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=quantity,
                    entry_price=price,
                    entry_date=datetime.now()
                )
        else:  # sell
            if symbol not in self.positions or self.positions[symbol].quantity < quantity:
                return False
                
            self.cash_balance += trade_value
            position = self.positions[symbol]
            pnl = (price - position.entry_price) * quantity
            
            position.quantity -= quantity
            if position.quantity == 0:
                del self.positions[symbol]
        
        # Record trade
        self.trade_history.append(Trade(
            symbol=symbol,
            quantity=quantity,
            price=price,
            side=side,
            date=datetime.now(),
            pnl=pnl if side.lower() == 'sell' else None
        ))
        
        return True
    
    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate total portfolio value including cash and positions"""
        positions_value = sum(
            pos.quantity * current_prices.get(pos.symbol, pos.entry_price)
            for pos in self.positions.values()
        )
        return self.cash_balance + positions_value
    
    def get_performance_metrics(self, current_prices: Dict[str, float]) -> dict:
        """Calculate portfolio performance metrics"""
        total_value = self.get_portfolio_value(current_prices)
        total_return = ((total_value - self.initial_balance) / self.initial_balance) * 100
        
        # Calculate win/loss ratio
        closed_trades = [t for t in self.trade_history if t.pnl is not None]
        winning_trades = len([t for t in closed_trades if t.pnl > 0])
        losing_trades = len([t for t in closed_trades if t.pnl < 0])
        win_ratio = winning_trades / len(closed_trades) if closed_trades else 0
        
        # Best and worst trades
        if closed_trades:
            best_trade = max(closed_trades, key=lambda t: t.pnl or 0)
            worst_trade = min(closed_trades, key=lambda t: t.pnl or 0)
        else:
            best_trade = worst_trade = None
            
        return {
            "total_value": total_value,
            "total_return": total_return,
            "cash_balance": self.cash_balance,
            "win_ratio": win_ratio,
            "total_trades": len(self.trade_history),
            "best_trade": best_trade,
            "worst_trade": worst_trade,
            "open_positions": len(self.positions)
        }

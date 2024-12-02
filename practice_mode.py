from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple
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
        
    def execute_trade(self, symbol: str, quantity: int, price: float, side: str) -> Tuple[bool, str]:
        """Execute a practice trade and update portfolio"""
        try:
            # Input validation
            if not symbol or not isinstance(symbol, str):
                return False, "Invalid symbol"
            if not quantity or quantity <= 0:
                return False, "Invalid quantity"
            if not price or price <= 0:
                return False, "Invalid price"
            if not side or side.lower() not in ['buy', 'sell']:
                return False, "Invalid trade side"

            trade_value = quantity * price
            pnl = None
            
            try:
                if side.lower() == 'buy':
                    if trade_value > self.cash_balance:
                        return False, "Insufficient funds for trade"
                    
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
                    if symbol not in self.positions:
                        return False, "Position not found"
                    if self.positions[symbol].quantity < quantity:
                        return False, "Insufficient shares for sale"
                    
                    position = self.positions[symbol]
                    pnl = (price - position.entry_price) * quantity
                    self.cash_balance += trade_value
                    
                    position.quantity -= quantity
                    if position.quantity == 0:
                        del self.positions[symbol]

                # Record trade
                trade = Trade(
                    symbol=symbol,
                    quantity=quantity,
                    price=price,
                    side=side,
                    date=datetime.now(),
                    pnl=pnl
                )
                self.trade_history.append(trade)
                
                return True, "Trade executed successfully"
                
            except Exception as e:
                # Rollback any changes if error occurs
                if side.lower() == 'buy':
                    self.cash_balance += trade_value
                else:
                    self.cash_balance -= trade_value
                raise Exception(f"Error executing trade: {str(e)}")
                
        except Exception as e:
            return False, str(e)
    
    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate total portfolio value including cash and positions"""
        try:
            positions_value = sum(
                pos.quantity * current_prices.get(pos.symbol, pos.entry_price)
                for pos in self.positions.values()
            )
            return self.cash_balance + positions_value
        except Exception as e:
            print(f"Error calculating portfolio value: {str(e)}")
            return self.cash_balance
    
    def get_performance_metrics(self, current_prices: Dict[str, float]) -> dict:
        """Calculate portfolio performance metrics"""
        try:
            total_value = self.get_portfolio_value(current_prices)
            total_return = ((total_value - self.initial_balance) / self.initial_balance) * 100
            
            # Calculate win/loss ratio
            closed_trades = [t for t in self.trade_history if t.pnl is not None]
            winning_trades = len([t for t in closed_trades if t.pnl > 0])
            total_closed = len(closed_trades)
            win_ratio = winning_trades / total_closed if total_closed > 0 else 0
            
            # Best and worst trades
            best_trade = max(closed_trades, key=lambda t: t.pnl or 0) if closed_trades else None
            worst_trade = min(closed_trades, key=lambda t: t.pnl or 0) if closed_trades else None
                
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
        except Exception as e:
            print(f"Error calculating performance metrics: {str(e)}")
            return {
                "total_value": self.cash_balance,
                "total_return": 0,
                "cash_balance": self.cash_balance,
                "win_ratio": 0,
                "total_trades": 0,
                "best_trade": None,
                "worst_trade": None,
                "open_positions": 0
            }

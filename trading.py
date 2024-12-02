import os
from cryptography.fernet import Fernet
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
from typing import Optional, Dict, Tuple

class TradingServiceManager:
    def __init__(self):
        self.supported_platforms = {
            'Alpaca': 'alpaca',
            'Interactive Brokers': 'ibkr'
        }
        self._load_encryption_key()
        
    def _load_encryption_key(self):
        """Initialize or load encryption key for API credentials"""
        key_path = '.key'
        if not os.path.exists(key_path):
            self.key = Fernet.generate_key()
            with open(key_path, 'wb') as key_file:
                key_file.write(self.key)
        else:
            with open(key_path, 'rb') as key_file:
                self.key = key_file.read()
        self.cipher_suite = Fernet(self.key)
        
    def encrypt_credentials(self, credentials: Dict[str, str]) -> Dict[str, bytes]:
        """Encrypt API credentials"""
        return {
            key: self.cipher_suite.encrypt(value.encode())
            for key, value in credentials.items()
        }
        
    def decrypt_credentials(self, encrypted_creds: Dict[str, bytes]) -> Dict[str, str]:
        """Decrypt API credentials"""
        return {
            key: self.cipher_suite.decrypt(value).decode()
            for key, value in encrypted_creds.items()
        }
        
    def validate_connection(self, platform: str, credentials: Dict[str, str]) -> Tuple[bool, str]:
        """Validate trading platform connection"""
        try:
            if platform == 'Alpaca':
                api = tradeapi.REST(
                    credentials['api_key'],
                    credentials['api_secret'],
                    base_url=credentials.get('base_url', 'https://paper-api.alpaca.markets')
                )
                # Test connection
                account = api.get_account()
                return True, f"Connected successfully. Account status: {account.status}"
            else:
                return False, "Platform not supported yet"
        except Exception as e:
            return False, f"Connection failed: {str(e)}"
            
    def execute_trade(self, platform: str, credentials: Dict[str, str], 
                     symbol: str, quantity: int, order_type: str, 
                     side: str, limit_price: Optional[float] = None) -> Tuple[bool, str]:
        """Execute trade on the specified platform"""
        try:
            if platform == 'Alpaca':
                api = tradeapi.REST(
                    credentials['api_key'],
                    credentials['api_secret'],
                    base_url=credentials.get('base_url', 'https://paper-api.alpaca.markets')
                )
                
                order = api.submit_order(
                    symbol=symbol,
                    qty=quantity,
                    side=side.lower(),
                    type=order_type.lower(),
                    time_in_force='gtc',
                    limit_price=limit_price if order_type.lower() == 'limit' else None
                )
                
                return True, f"Order submitted successfully. Order ID: {order.id}"
            else:
                return False, "Platform not supported yet"
        except Exception as e:
            return False, f"Trade execution failed: {str(e)}"

# Initialize trading service
trading_service = TradingServiceManager()

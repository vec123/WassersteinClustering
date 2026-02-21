import pandas as pd
import yfinance as yf
import logging

logger = logging.getLogger(__name__)

class DataFetcher:
    @staticmethod
    def get_historical_data(symbol: str, start: str, end: str) -> pd.DataFrame:
        """Fetches data from Yahoo Finance (No API Key Required)."""
        logger.info(f"Fetching {symbol} from Yahoo Finance...")
        
        # Yahoo uses ^NSEI for NIFTY 50
        ticker = "^NSEI" if symbol == "NIFTY" else symbol
        
        df = yf.download(ticker, start=start, end=end)
        
        if df.empty:
            raise ValueError(f"No data found for {symbol}. Check your internet or ticker.")
        
        # Clean column names (yfinance sometimes returns MultiIndex)
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        df.columns = df.columns.str.lower()
        
        return df
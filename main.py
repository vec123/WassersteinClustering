import warnings
import numpy as np
# Silence Numba warnings
warnings.simplefilter('ignore', category=RuntimeWarning)

from utils.data_fetcher import DataFetcher
from utils.visualizer import MarketVisualizer
from core.processing import RegimeProcessor

def main():
    # 1. Setup Parameters
    SYMBOLS = ["NIFTY", "BTC-USD", "GC=F" ]
    SYMBOL = "GC=F"
    START = "2021-01-01"
    END = "2022-01-01"

    # 2. Fetch Data (No API needed)
    df = DataFetcher.get_historical_data(SYMBOL, START, END)
    
    # 3. Calculate Log Returns
    # Using np.log(price_t / price_t-1)
    df['log_return'] = np.log(df['close'] / df['close'].shift(1)).fillna(0)

    # 4. Process Regimes
    # window_len=10, min_train=20 windows, refit every 10 bars
    processor = RegimeProcessor(window_len=10, min_train_windows=20, refit_freq=10)
    df['regime'] = processor.classify(df['log_return'])

    # 5. Visualize
    print("Regime Distribution:")
    print(df['regime'].value_counts())
    
    viz = MarketVisualizer()
    fig = viz.plot_regimes(df, SYMBOL)
    fig.show()

if __name__ == "__main__":
    main()
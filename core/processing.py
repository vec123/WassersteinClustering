import pandas as pd
import numpy as np
from .clustering import WassersteinKMeans

class RegimeProcessor:
    def __init__(self, window_len: int, min_train_windows: int, refit_freq: int):
        self.window_len = window_len
        self.min_train_windows = min_train_windows
        self.refit_freq = refit_freq
        self.model = WassersteinKMeans()

    def get_non_overlapping_windows(self, data: np.ndarray) -> list:
        return [data[i:i + self.window_len] for i in range(0, len(data) - self.window_len + 1, self.window_len)]

    def classify(self, returns: pd.Series) -> pd.Series:
        labels = np.full(len(returns), 0)
        ret_arr = returns.values
        min_bars = self.window_len * (self.min_train_windows + 1)
        
        last_refit = -1
        
        for t in range(min_bars, len(returns)):
            # Refit Logic
            if (t - last_refit) >= self.refit_freq or last_refit == -1:
                hist_ret = ret_arr[:t] # Lookback only
                train_windows = self.get_non_overlapping_windows(hist_ret)
                self.model.fit(train_windows)
                last_refit = t
            
            # Current Window Prediction
            current_win = ret_arr[t - self.window_len + 1 : t + 1]
            labels[t] = self.model.predict(current_win)
            
        return pd.Series(labels, index=returns.index)
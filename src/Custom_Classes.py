import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineer(BaseEstimator, TransformerMixin):
    
    def __init__(self, window=5):
        """
        Initialize the transformer with parameters (e.g., the MA window size).
        """
        self.window = window

    def fit(self, X, y=None):
        """
        In most feature creation cases, 'fit' does nothing, as the 
        transformation doesn't learn parameters from the data.
        """
        return self

    def transform(self, X):
        if isinstance(X, np.ndarray):
            # If the input is a NumPy array (often from a previous pipeline step), 
            # create a DataFrame for easy feature calculation.
            X_df = pd.DataFrame(X)
        else:
            X_df = X.copy()

        # Ensure X_out is initialized with the correct index/shape
        X_out = pd.DataFrame(index=X_df.index)
        
        # --- FEATURE ENGINEERING LOGIC ---
        # Calculation of exponential moving average
        name_feature = 'EMA_' + str(self.window)
        X_out[name_feature] = X_df.ewm(span=self.window, min_periods=self.window).mean().squeeze()

        # Calculation of rate of change
        name_feature = 'ROC_' + str(self.window)
        M = X_df.diff(self.window - 1).squeeze()
        N = X_df.shift(self.window - 1).squeeze()
        X_out[name_feature] = ((M / N) * 100)

        # Calculation of price momentum
        name_feature = 'MOM_' + str(self.window)
        X_out[name_feature] = X_df.diff(self.window).squeeze()

        # Calculation of relative strength index
        name_feature = 'RSI_' + str(self.window)
        delta = X_df.squeeze().diff()
        u = pd.Series(np.where(delta > 0, delta, 0), index=delta.index)
        d = pd.Series(np.where(delta < 0, -delta, 0), index=delta.index)
        avg_gain = u.ewm(com=self.window-1, adjust=False).mean()
        avg_loss = d.ewm(com=self.window-1, adjust=False).mean()
        RS = avg_gain / avg_loss
        X_out[name_feature] = 100 - (100 / (1 + RS))
        
        # Calculation of simple moving average
        name_feature = 'MA_' + str(self.window)
        X_out[name_feature] = X_df.rolling(self.window, min_periods=self.window).mean().squeeze()

        return X_out
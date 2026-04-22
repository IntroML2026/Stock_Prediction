import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PowerTransformer
from scipy.stats import skew
from gensim.models import Word2Vec



class AutoPowerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.75):
        self.threshold = threshold
        self.skewed_cols = []
        self.pt = PowerTransformer(method='yeo-johnson')

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        # PROTECTION: Only look at columns that are actually numeric
        # This prevents the step from ever seeing a categorical string
        numeric_df = X.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            return self

        # Only calculate skewness for numeric columns
        skewness = numeric_df.apply(lambda x: skew(x.dropna()))
        self.skewed_cols = skewness[abs(skewness) > self.threshold].index.tolist()
        
        if self.skewed_cols:
            self.pt.fit(X[self.skewed_cols])
        return self

    def transform(self, X):
        X_copy = X.copy()
        if not isinstance(X_copy, pd.DataFrame):
            X_copy = pd.DataFrame(X_copy)
            
        if self.skewed_cols:
            X_copy[self.skewed_cols] = self.pt.transform(X_copy[self.skewed_cols])
        return X_copy



class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, missing_threshold=0.3, corr_threshold=0.03, cardinality_threshold=0.9):
        self.missing_threshold = missing_threshold
        self.corr_threshold = corr_threshold
        self.cardinality_threshold = cardinality_threshold # Ratio of unique values to total rows
        self.features_to_keep = []

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        # 1. Missing Values Filter
        null_ratios = X.isnull().mean()
        cols_low_missing = null_ratios[null_ratios <= self.missing_threshold].index.tolist()
        X_filtered = X[cols_low_missing]

        # 2. High Cardinality Filter (Only for Categorical/Object columns)
        cat_cols = X_filtered.select_dtypes(exclude='number').columns
        cols_to_drop = []
        
        for col in cat_cols:
            uniqueness_ratio = X_filtered[col].nunique() / len(X_filtered)
            if uniqueness_ratio > self.cardinality_threshold:
                cols_to_drop.append(col)
        
        # Keep categoricals that are NOT high cardinality
        remaining_cats = [c for c in cat_cols if c not in cols_to_drop]

        # 3. Correlation Filter (Only for Numeric columns)
        numeric_X = X_filtered.select_dtypes(include='number')
        if y is not None and not numeric_X.empty:
            temp_df = numeric_X.copy()
            temp_df['target'] = y
            correlations = temp_df.corr()['target'].abs().drop('target')
            numeric_to_keep = correlations[correlations >= self.corr_threshold].index.tolist()
        else:
            numeric_to_keep = numeric_X.columns.tolist()

        self.features_to_keep = numeric_to_keep + remaining_cats
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        return X[self.features_to_keep]

class FeatureEngineer(BaseEstimator, TransformerMixin):
    
    def __init__(self, windows=[5, 10, 20]):
        """
        Initialize with a list of windows. 
        Example: FeatureEngineer(windows=[5, 14, 30])
        """
        self.windows = windows

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Handle input types
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X)
        else:
            X_df = X.copy()

        # Ensure we are working with a Series for rolling/diff operations
        # squeeze() is used if X_df is a single-column DataFrame
        data = X_df.squeeze()
        X_out = pd.DataFrame(index=X_df.index)
        
        # Iterate through each window to create multi-scale features
        for w in self.windows:
            
            # 1. Exponential Moving Average
            X_out[f'EMA_{w}'] = data.ewm(span=w, min_periods=w).mean()

            # 2. Rate of Change
            M = data.diff(w - 1)
            N = data.shift(w - 1)
            X_out[f'ROC_{w}'] = (M / N) * 100

            # 3. Price Momentum
            X_out[f'MOM_{w}'] = data.diff(w)

            # 4. Relative Strength Index (RSI)
            delta = data.diff()
            u = pd.Series(np.where(delta > 0, delta, 0), index=delta.index)
            d = pd.Series(np.where(delta < 0, -delta, 0), index=delta.index)
            avg_gain = u.ewm(com=w - 1, adjust=False).mean()
            avg_loss = d.ewm(com=w - 1, adjust=False).mean()
            rs = avg_gain / avg_loss
            X_out[f'RSI_{w}'] = 100 - (100 / (1 + rs))
            
            # 5. Simple Moving Average
            X_out[f'MA_{w}'] = data.rolling(w, min_periods=w).mean()

            # 6. Oscillators

        return X_out

class PairFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, window=60):
        self.window = window
        # Internal state
        self.last_beta_ = None
        self.last_alpha_ = None
        self.is_fitted_ = False

    def fit(self, X, y=None):
        """
        Validates that the input data is sufficient for the window size.
        In scikit-learn, fit must always return self.
        """
        if len(X) < self.window:
            raise ValueError(f"Data length {len(X)} is less than window size {self.window}")
        
        self.is_fitted_ = True
        return self

    def transform(self, X):
        """
        X: Expected to be a DataFrame or Array with 2 columns: [Price_A, Price_B]
        """
        if not self.is_fitted_:
            raise RuntimeError("Extractor must be fitted before calling transform.")

        # Convert to DataFrame if input is a numpy array
        if isinstance(X, np.ndarray):
            df = pd.DataFrame(X, columns=['price_a', 'price_b'])
        else:
            df = X.copy()
            df.columns = ['price_a', 'price_b']
        
        # 1. Compute Rolling Spread and Beta
        df[['spread', 'beta']] = self._compute_rolling_regression(df)

        # 2. Derive Statistics-based Features
        df['z_score'] = self._calculate_z_score(df['spread'])
        df['spread_std'] = df['spread'].rolling(self.window).std()
        df['beta_stability'] = df['beta'].rolling(self.window).std()

        
        return df#.dropna()

    def _compute_rolling_regression(self, df):
        spreads = np.full(len(df), np.nan)
        betas = np.full(len(df), np.nan)
        
        a_vals = df['price_a'].values
        b_vals = df['price_b'].values

        for i in range(self.window, len(df)):
            y = a_vals[i-self.window:i]
            x = b_vals[i-self.window:i]
            x_with_const = sm.add_constant(x)
            
            model = sm.OLS(y, x_with_const).fit()
            
            alpha, beta = model.params[0], model.params[1]
            betas[i] = beta
            spreads[i] = a_vals[i] - (beta * b_vals[i] + alpha)
            
            # Update state for live prediction tracking
            self.last_alpha_, self.last_beta_ = alpha, beta
            
        return pd.DataFrame({'spread': spreads, 'beta': betas}, index=df.index)

    def _calculate_z_score(self, spread_series):
        rolling_mean = spread_series.rolling(self.window).mean()
        rolling_std = spread_series.rolling(self.window).std()
        return (spread_series - rolling_mean) / rolling_std

class Word2VecTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, vector_size=100, window=5, min_count=1):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.model = None

    def fit(self, X, y=None):
        # create the word2vec model
        sentences = [str(row[0]).split() for row in X]
        self.model = Word2Vec(sentences, vector_size=self.vector_size, 
                              window=self.window, min_count=self.min_count)
        return self

    def transform(self, X):
        # Convert each headline into the average of its word vectors
        def get_mean_vector(text):
            words = str(text).split()
            # Filter words that actually exist in the Word2Vec vocabulary
            vectors = [self.model.wv[w] for w in words if w in self.model.wv]
            if not vectors:
                return np.zeros(self.vector_size)
            return np.mean(vectors, axis=0)

        return np.array([get_mean_vector(row[0]) for row in X])

# --- Usage Example ---
# extractor = PairFeatureExtractor(window=60)
# features_df = extractor.transform(data['AAPL'], data['MSFT'])






from sklearn.base import BaseEstimator, TransformerMixin

#class DropHighMissingCols(BaseEstimator, TransformerMixin):
    
#    def __init__(self, threshold=0.5):
#        self.threshold = threshold
#        self.cols_to_drop_ = []
#
#    def fit(self, X, y=None):
#        missing_ratio = X.isnull().mean()
#        self.cols_to_drop_ = missing_ratio[missing_ratio > self.threshold].index.tolist()
#        return self
#
#    def transform(self, X, y=None):
#        return X.drop(columns=self.cols_to_drop_, errors='ignore')

class DropHighMissingCols(BaseEstimator, TransformerMixin):
    
    def __init__(self, threshold=0.5, protect_cols=None):
        self.threshold     = threshold
        self.protect_cols  = protect_cols or []
        self.cols_to_drop_ = []
        self.numeric_medians_ = None
        self.cat_modes_       = None

    def fit(self, X, y=None):
        # Learn which columns to drop
        missing_ratio = X.isnull().mean()
        self.cols_to_drop_ = [
            col for col in missing_ratio[missing_ratio > self.threshold].index
            if col not in self.protect_cols
        ]

        # Learn medians and modes from training data only
        remaining = X.drop(columns=self.cols_to_drop_, errors='ignore')

        numeric_cols          = remaining.select_dtypes(include=['float64', 'int64']).columns
        self.numeric_medians_ = remaining[numeric_cols].median()

        cat_cols          = remaining.select_dtypes(include=['object']).columns
        self.cat_modes_   = remaining[cat_cols].mode().iloc[0]

        return self

    def transform(self, X, y=None):
        X = X.copy()

        # Drop high missing columns
        X = X.drop(columns=self.cols_to_drop_, errors='ignore')

        # Fill numeric with learned medians
        numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
        X[numeric_cols] = X[numeric_cols].fillna(self.numeric_medians_[numeric_cols])

        # Fill categorical with learned modes
        cat_cols = X.select_dtypes(include=['object']).columns
        X[cat_cols] = X[cat_cols].fillna(self.cat_modes_[cat_cols])

        return X


import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder

class TransactionFeatureEngineer(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.high_value_threshold_ = None
        self.addr1_frequency_      = None
        self.card1_frequency_      = None
        self.device_frequency_     = None
        self.card1_avg_amt_        = None
        self.card1_total_spend_    = None
        self.card1_max_amt_        = None
        self.card1_day_velocity_   = None
        self.label_encoders_       = {}

    def fit(self, X, y=None):
        # Learn threshold from training data only
        self.high_value_threshold_ = X['TransactionAmt'].quantile(0.75)

        # Learn frequency maps from training data only
        self.addr1_frequency_ = X['addr1'].value_counts()
        self.card1_frequency_ = X['card1'].value_counts()

        if 'DeviceType' in X.columns:
            self.device_frequency_ = X['DeviceType'].value_counts()

        # Learn group aggregates from training data only
        self.card1_avg_amt_      = X.groupby('card1')['TransactionAmt'].mean()
        self.card1_total_spend_  = X.groupby('card1')['TransactionAmt'].sum()
        self.card1_max_amt_      = X.groupby('card1')['TransactionAmt'].max()

        # Need TransactionHour for velocity — compute temporarily
        X_temp = X.copy()
        if 'TransactionDT' in X_temp.columns:
            X_temp['TransactionHour'] = (X_temp['TransactionDT'] / 3600).astype(int) % 24
        self.card1_day_velocity_ = X_temp.groupby(['card1', 'TransactionHour'])['TransactionAmt'].count()

        # Learn label encoders from training data only
        cat_cols = X.select_dtypes(include=['object']).columns
        for col in cat_cols:
            le = LabelEncoder()
            le.fit(list(X[col].astype(str).unique()) + ['unknown'])
            self.label_encoders_[col] = le

        return self

    def transform(self, X, y=None):
        X = X.copy()

        # 1. Transaction Amount Features
        X['TransactionAmt_Log']    = np.log1p(X['TransactionAmt'])
        X['HighValue_Flag']        = (X['TransactionAmt'] > self.high_value_threshold_).astype(int)
        X['IsRoundAmount']         = (X['TransactionAmt'] % 1 == 0).astype(int)

        # 2. Time Features
        if 'TransactionDT' in X.columns:
            X['TransactionHour']    = (X['TransactionDT'] / 3600).astype(int) % 24
            X['TransactionDay']     = (X['TransactionDT'] / 86400).astype(int) % 7
            X['IsNightTransaction'] = (
                (X['TransactionHour'] >= 0) & (X['TransactionHour'] <= 6)
            ).astype(int)

        # 3. Address Features
        X['Address_Match']   = (X['addr1'] == X['addr2']).astype(int)
        X['Addr1_Frequency'] = X['addr1'].map(self.addr1_frequency_).fillna(0)

        # 4. Velocity Features
        X['Card1_Frequency']       = X['card1'].map(self.card1_frequency_).fillna(0)
        X['Card1_AvgAmt']          = X['card1'].map(self.card1_avg_amt_).fillna(0)
        X['Amt_Diff_From_CardAvg'] = X['TransactionAmt'] - X['Card1_AvgAmt']
        X['Card_Total_Spend']      = X['card1'].map(self.card1_total_spend_).fillna(0)
        X['Card_Max_Amt']          = X['card1'].map(self.card1_max_amt_).fillna(0)

        velocity_map = self.card1_day_velocity_
        X['Card_Day_Velocity'] = (
            X.set_index(['card1', 'TransactionHour']).index.map(velocity_map).fillna(0).values
        )

        # 5. Device Features
        if 'DeviceType' in X.columns and self.device_frequency_ is not None:
            X['DeviceType_Frequency'] = X['DeviceType'].map(self.device_frequency_).fillna(0)

        # 6. Label Encoding
        #for col, le in self.label_encoders_.items():
         #   if col in X.columns:
          #      X[col] = le.transform(X[col].astype(str))
        # 6. Label Encoding
        for col, le in self.label_encoders_.items():
            if col in X.columns:
                # Replace unseen labels with 'unknown' before encoding
                known_labels = set(le.classes_)
                X[col] = X[col].astype(str).apply(
                    lambda x: x if x in known_labels else 'unknown'
                )
                X[col] = le.transform(X[col])

        return X

class DropHighCorrelation(BaseEstimator, TransformerMixin):
    
    def __init__(self, threshold=0.90):
        self.threshold = threshold
        self.cols_to_drop_ = []

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        corr_matrix = X.corr().abs()
        upper       = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        self.cols_to_drop_ = [
            col for col in upper.columns if any(upper[col] > self.threshold)
        ]
        return self

    def transform(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        return X.drop(columns=self.cols_to_drop_, errors='ignore')
        
# --- Usage Example ---
# extractor = PairFeatureExtractor(window=60)
# features_df = extractor.transform(data['AAPL'], data['MSFT'])        

        
# --- Usage Example ---
# extractor = PairFeatureExtractor(window=60)
# features_df = extractor.transform(data['AAPL'], data['MSFT'])

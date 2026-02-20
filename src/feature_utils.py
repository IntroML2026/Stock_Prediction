import numpy as np
import pandas as pd
import datetime
import yfinance as yf
import pandas_datareader.data as web
import requests
#from datetime import datetime, timedelta
import os
import sys

import os
import sys


# ... continue with your script ...

def extract_features():

    return_period = 5
    
    START_DATE = (datetime.date.today() - datetime.timedelta(days=365)).strftime("%Y-%m-%d")
    END_DATE = datetime.date.today().strftime("%Y-%m-%d")
    stk_tickers = ['COST', 'WMT', 'TGT'] #['MSFT', 'IBM', 'GOOGL']
    ccy_tickers = ['DEXCAUS', 'DEXJPUS']
    idx_tickers = ['SP500', 'DJIA', 'VIXCLS']
    
    stk_data = yf.download(stk_tickers, start=START_DATE, end=END_DATE, auto_adjust=False)
    #stk_data = web.DataReader(stk_tickers, 'yahoo')
    ccy_data = web.DataReader(ccy_tickers, 'fred', start=START_DATE, end=END_DATE)
    idx_data = web.DataReader(idx_tickers, 'fred', start=START_DATE, end=END_DATE)

    return_period = 5
    
    # ===== Target (5-day future log return on COST) =====
    Y = np.log(stk_data.loc[:, ('Adj Close', 'COST')]).diff(return_period).shift(-return_period)
    Y.name = 'COST_Future'
    
    
    # ===== Core Features =====
    # Correlated stocks (WMT, TGT)
    X1 = np.log(stk_data.loc[:, ('Adj Close', ['WMT', 'TGT'])]).diff(return_period)
    X1.columns = X1.columns.droplevel(0)  # drops "Adj Close" level -> ['WMT','TGT']
    
    # FX + indices
    X2 = np.log(ccy_data).diff(return_period)
    X3 = np.log(idx_data).diff(return_period)
    
    
    # ===== NEW Technical Features on COST =====
    open_ = stk_data.loc[:, ('Open', 'COST')]
    high_ = stk_data.loc[:, ('High', 'COST')]
    low_  = stk_data.loc[:, ('Low', 'COST')]
    close_ = stk_data.loc[:, ('Close', 'COST')]
    
    # 1) Range
    range_feat = ((high_ - low_) / open_).to_frame('Range')
    
    # 2) Gap
    gap_feat = ((open_ - close_.shift(1)) / close_.shift(1)).to_frame('Gap')
    
    # 3) Intraday return
    intraday_feat = ((close_ - open_) / open_).to_frame('Intraday_Return')
    
    # 4) Rolling volatility based on return_period log returns
    cost_r = np.log(stk_data.loc[:, ('Adj Close', 'COST')]).diff(return_period)
    roll_vol_feat = cost_r.rolling(10).std().to_frame('Rolling_Vol_10')
    
    X4 = pd.concat([range_feat, gap_feat, intraday_feat, roll_vol_feat], axis=1)
    
    
    # ===== Final dataset =====
    X = pd.concat([X1, X2, X3, X4], axis=1)
    
    dataset = pd.concat([Y, X], axis=1).dropna().iloc[::return_period, :]
    Y = dataset[Y.name]
    X = dataset[X.columns]
    dataset.index.name = 'Date'

    #dataset.to_csv(r"./test_data.csv")
    features = dataset.sort_index()
    features = features.reset_index(drop=True)
    features = features.iloc[:,1:]
    return features


def get_bitcoin_historical_prices(days = 60):
    
    BASE_URL = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    
    params = {
        'vs_currency': 'usd',
        'days': days,
        'interval': 'daily' # Ensure we get daily granularity
    }
    response = requests.get(BASE_URL, params=params)
    data = response.json()
    prices = data['prices']
    df = pd.DataFrame(prices, columns=['Timestamp', 'Close Price (USD)'])
    df['Date'] = pd.to_datetime(df['Timestamp'], unit='ms').dt.normalize()
    df = df[['Date', 'Close Price (USD)']].set_index('Date')
    return df





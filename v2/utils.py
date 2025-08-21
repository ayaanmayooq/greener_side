import pandas as pd
import numpy as np
from datetime import datetime


def load_yfinance_data(symbol, start, end):
    symbol = symbol.lower()
    data_dir = "data"
    file_path = f"{data_dir}/{symbol}.csv"

    df = pd.read_csv(
        file_path,
        header=None,
        skiprows=3,
        names=['Date','Close','High','Low','Open','Volume'],
        usecols=[0,1,2,3,4,5]
    )

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').set_index('Date')
    df_bt = df[['Open','High','Low','Close','Volume']].astype(float)
    return df_bt



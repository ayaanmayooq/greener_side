import pandas as pd
import backtrader as bt
from datetime import datetime

class SmaCross(bt.SignalStrategy):
    def __init__(self):
        sma1 = bt.ind.SMA(period=10)
        sma2 = bt.ind.SMA(period=30)
        x = bt.ind.CrossOver(sma1, sma2)
        self.signal_add(bt.SIGNAL_LONG, x)

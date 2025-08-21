import pandas as pd
import backtrader as bt
from datetime import datetime

class MonthlyTrend(bt.Strategy):
    params = dict(
        ema_fast=5, ema_slow=20,     # ~1M cadence
        # ema_fast=3, ema_slow=10,
        adx_period=14, adx_cut=18,   # trend quality gate
        # adx_period=14, adx_cut=12,
        hold_max=20,                 # time stop ~1M
        # hold_max=10,
        reentry_wait=1               # cool-off bars after exit
    )

    def __init__(self):
        self.ema_f = bt.ind.EMA(self.data.close, period=self.p.ema_fast)
        self.ema_s = bt.ind.EMA(self.data.close, period=self.p.ema_slow)
        self.adx   = bt.ind.ADX(self.data, period=self.p.adx_period)
        # 20d momentum (sign only)
        self.mom20 = (self.data.close / self.data.close(-20)) - 1.0

        # vote components
        self.cross_up  = bt.ind.CrossOver(self.ema_f, self.ema_s)  # +1 on bull cross
        self.slope_pos = (self.ema_s - self.ema_s(-5)) > 0         # 1-week slope > 0
        self.adx_ok    = self.adx > self.p.adx_cut
        self.mom_pos   = self.mom20 > 0

        self.bars_in_trade = 0
        self.cooldown = 0

    def bullish_vote(self):
        votes = 0
        votes += int(self.ema_f[0] > self.ema_s[0])    # fast above slow
        votes += int(self.slope_pos[0])                # slow MA rising
        votes += int(self.mom_pos[0])                  # 20d momentum +
        votes += int(self.adx_ok[0])                   # trend quality ok
        return votes

    def next(self):
        # cooldown after exit to avoid churn
        if self.cooldown > 0:
            self.cooldown -= 1

        votes = self.bullish_vote()

        if not self.position:
            if self.cooldown == 0 and votes >= 3:
                self.buy()            # size controlled by sizer outside
                self.bars_in_trade = 0
        else:
            self.bars_in_trade += 1
            # exit if signal fades or time stop reached
            if votes <= 1 or self.bars_in_trade >= self.p.hold_max:
                self.close()
                self.cooldown = self.p.reentry_wait


class MonthlyTrendLongShort(bt.Strategy):
    params = dict(
        ema_fast=5, ema_slow=20,
        adx_period=14, adx_cut=18,
        hold_max=20,          # ~1 month
        reentry_wait=3        # cool-off
    )
    def __init__(self):
        self.ema_f = bt.ind.EMA(self.data.close, period=self.p.ema_fast)
        self.ema_s = bt.ind.EMA(self.data.close, period=self.p.ema_slow)
        self.adx   = bt.ind.ADX(self.data, period=self.p.adx_period)
        self.mom20 = (self.data.close / self.data.close(-20)) - 1.0

        self.slope_pos = (self.ema_s - self.ema_s(-5)) > 0
        self.slope_neg = (self.ema_s - self.ema_s(-5)) < 0
        self.adx_ok    = self.adx > self.p.adx_cut

        self.bars_in_trade = 0
        self.cooldown = 0

    def bullish_votes(self):
        v  = int(self.ema_f[0] > self.ema_s[0])
        v += int(self.slope_pos[0])
        v += int(self.mom20[0] > 0)
        v += int(self.adx_ok[0])
        return v

    def bearish_votes(self):
        v  = int(self.ema_f[0] < self.ema_s[0])
        v += int(self.slope_neg[0])
        v += int(self.mom20[0] < 0)
        v += int(self.adx_ok[0])
        return v

    def next(self):
        if self.cooldown > 0:
            self.cooldown -= 1

        bull = self.bullish_votes()
        bear = self.bearish_votes()

        if not self.position:
            if self.cooldown == 0:
                if bull >= 3 and bull > bear:
                    self.buy()
                    self.bars_in_trade = 0
                elif bear >= 3 and bear > bull:
                    self.sell()  # opens a short
                    self.bars_in_trade = 0
        else:
            self.bars_in_trade += 1
            # exit if signal fades (≤1 vote) or time stop
            if (self.position.size > 0 and bull <= 1) or \
               (self.position.size < 0 and bear <= 1) or \
               (self.bars_in_trade >= self.p.hold_max):
                self.close()
                self.cooldown = self.p.reentry_wait


import backtrader as bt

class MonthlyTrendRiskManaged(bt.Strategy):
    params = dict(
        # --- signal ---
        ema_fast=5, ema_slow=20,
        adx_period=14, adx_cut=18,
        hold_max=20,
        reentry_wait=3,

        # --- risk & exits ---
        atr_period=14,
        stop_atr_mult=2.0,      # initial stop = entry - 2*ATR
        be_after_R=1.0,         # move stop to breakeven after +1R
        trail_after_R=1.5,      # start trailing after +1.5R
        trail_atr_mult=2.0,     # trailing stop distance = 2*ATR
        take_profit_R=None,     # e.g. 2.0 for +2R target (None disables)
        adx_exit_bars=3,
    )

    def __init__(self):
        # indicators
        self.ema_f = bt.ind.EMA(self.data.close, period=self.p.ema_fast)
        self.ema_s = bt.ind.EMA(self.data.close, period=self.p.ema_slow)
        self.adx   = bt.ind.ADX(self.data, period=self.p.adx_period)
        self.mom20 = (self.data.close / self.data.close(-20)) - 1.0

        self.slope_pos = (self.ema_s - self.ema_s(-5)) > 0
        self.adx_ok    = self.adx > self.p.adx_cut
        self.mom_pos   = self.mom20 > 0

        self.atr = bt.ind.ATR(self.data, period=self.p.atr_period)

        # state
        self.bars_in_trade = 0
        self.cooldown = 0
        self.adx_below_count = 0

        self.entry_price = None
        self.stop_level  = None        # numeric stop price we manage
        self.stop_order  = None
        self.tp_order    = None

    # ------------ helpers ------------
    def bullish_vote(self):
        votes = (
            int(self.ema_f[0] > self.ema_s[0]) +
            int(self.slope_pos[0]) +
            int(self.mom_pos[0]) +
            int(self.adx_ok[0])
        )
        return votes

    def _cancel(self, order):
        if order and order.status in [order.Submitted, order.Accepted, order.Partial, order.Created]:
            try: self.broker.cancel(order)
            except Exception: pass

    def _replace_stop(self, new_price):
        # only if we have a position and a valid price
        if not self.position: return
        if new_price is None: return
        if (self.stop_order is None) or (getattr(self.stop_order, 'created', None) is None) \
           or (new_price > self.stop_order.created.price):
            self._cancel(self.stop_order)
            self.stop_order = self.sell(exectype=bt.Order.Stop, price=new_price)
            self.stop_level = new_price

    # ------------ order/trade callbacks ------------
    def notify_order(self, order):
        # On a filled BUY: set true entry, create initial stop/TP
        if order.status == order.Completed and order.isbuy():
            self.entry_price = float(order.executed.price)
            atr_now = max(float(self.atr[0]), 1e-8)
            init_stop = self.entry_price - self.p.stop_atr_mult * atr_now
            self._replace_stop(init_stop)

            if self.p.take_profit_R is not None:
                risk_per_share = max(self.entry_price - init_stop, 1e-8)
                tgt = self.entry_price + self.p.take_profit_R * risk_per_share
                self._cancel(self.tp_order)
                self.tp_order = self.sell(exectype=bt.Order.Limit, price=tgt)

        # Clean handles if stop/TP got cancelled/rejected/expired
        if order in (self.stop_order, self.tp_order) and order.status in [order.Canceled, order.Rejected, order.Expired]:
            if order is self.stop_order: self.stop_order = None
            if order is self.tp_order:   self.tp_order   = None

    def notify_trade(self, trade):
        # When a trade fully closes, reset state and start cooldown
        if trade.isclosed:
            self.entry_price = None
            self.stop_level  = None
            self._cancel(self.stop_order); self.stop_order = None
            self._cancel(self.tp_order);   self.tp_order   = None
            self.bars_in_trade = 0
            self.cooldown = self.p.reentry_wait
            self.adx_below_count = 0

    # ------------ main loop ------------
    def next(self):
        # cooldown + adx tracking
        if self.cooldown > 0:
            self.cooldown -= 1
        self.adx_below_count = (self.adx_below_count + 1) if (self.adx[0] <= self.p.adx_cut) else 0

        votes = self.bullish_vote()

        if not self.position:
            if self.cooldown == 0 and votes >= 3:
                # ENTRY: let PercentSizer choose size; we’ll set stops on fill
                self.buy()
                self.bars_in_trade = 0
            return

        # IN POSITION
        self.bars_in_trade += 1
        price   = float(self.data.close[0])
        atr_now = max(float(self.atr[0]), 1e-8)

        # 1) Time stop
        if self.bars_in_trade >= self.p.hold_max:
            self.close(); return

        # 2) Signal fade
        if votes <= 1:
            self.close(); return

        # 3) ADX quality exit
        if self.adx_below_count >= self.p.adx_exit_bars:
            self.close(); return

        # If we don't yet know entry/stop (e.g., just filled), wait one bar
        if (self.entry_price is None) or (self.stop_level is None):
            return

        # R multiple since entry
        risk_per_share = max(self.entry_price - self.stop_level, 1e-8)
        R = (price - self.entry_price) / risk_per_share

        # 4) Move stop to breakeven after +1R
        if R >= self.p.be_after_R:
            be = max(self.entry_price, self.stop_level)
            if self.stop_level is None or be > self.stop_level:
                self._replace_stop(be)

        # 5) Trail after threshold
        if R >= self.p.trail_after_R:
            trail = price - self.p.trail_atr_mult * atr_now
            trail = max(trail, self.entry_price)  # never below BE
            if self.stop_level is None or trail > self.stop_level:
                self._replace_stop(trail)
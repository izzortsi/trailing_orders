
# %%
%matplotlib qt
import matplotlib.pyplot as plt
import numpy as np
import time
import logging
from binance.lib.utils import config_logging


config_logging(logging, logging.DEBUG)
from binance.websocket.um_futures.websocket_client import UMFuturesWebsocketClient as FuturesWebsocketClient
import pandas as pd
import pandas_ta as ta


class RingBuffer(FuturesWebsocketClient):
    def __init__(self, size):
        self.df = pd.DataFrame()
        self.size = size
        self.indicators = pd.DataFrame()
        super().__init__()
        

    def message_handler(self, message):
        try:
            # print(message)
            # print(type(message))
            if (message["e"] == "continuous_kline") and (len(self.df) < self.size):
                kline = message["k"]
                self.df = pd.concat([
                        self.df, 
                        pd.DataFrame([   
                            {
                            "s": message["ps"],
                            "date": pd.to_datetime(message["E"], unit="ms"),
                            "o": pd.to_numeric(kline["o"]),
                            "h": pd.to_numeric(kline["h"]),
                            "l": pd.to_numeric(kline["l"]),
                            "c": pd.to_numeric(kline["c"]),
                            "v": pd.to_numeric(kline["v"]),
                            },
                            ])], 
                            ignore_index = True,
                        )
            elif (message["e"] == "continuous_kline") and (len(self.df) >= self.size):
                kline = message["k"]
                self.df.drop(axis=0, index = 0, inplace=True), 
                self.df = pd.concat([
                        self.df, 
                        pd.DataFrame([   
                            {
                            "s": message["ps"],
                            "date": pd.to_datetime(message["E"], unit="ms"),
                            "o": pd.to_numeric(kline["o"]),
                            "h": pd.to_numeric(kline["h"]),
                            "l": pd.to_numeric(kline["l"]),
                            "c": pd.to_numeric(kline["c"]),
                            "v": pd.to_numeric(kline["v"]),
                            },
                            ])], 
                            ignore_index = True,
                        )
                atr_ = ta.atr(self.df.h, self.df.l, self.df.c, length=24)                        
                # print(atr_)
                closes_ema = ta.ema(self.df.c, length=24, label="ema")
                # print(closes_ema)
                closes_ema.name = "ema"
                # print(closes_ema)
                sup_band = closes_ema + 0.84*atr_
                sup_band.name = "sband"
                # print(sup_band)
                inf_band = closes_ema - 0.84*atr_
                inf_band.name = "iband"
                # indicators_row = pd.concat([inf_band, closes_ema, sup_band], axis=1)
                # print(indicators_row)
                # self.indicators_df = pd.concat([
                #                                 self.indicators_df,
                #                                 indicators_row
                #                                 ],
                #                                 ignore_index=True,
                #                             )
                self.indicators = pd.concat([inf_band, closes_ema, sup_band], axis=1)
                # print(self.indicators)
                # self.indicators_df = ta.atr(self.df.h, self.df.l, self.df.c, timeperiod=6)
                # print(self.indicators_df.iloc[-1])
                # print(self.df[["date", "c"]].iloc[0])
                # print(self.df[["date", "c"]].iloc[-1])
                # print(self.df.date.iloc[-1] - self.df.date.iloc[0])

        except Exception as e:
            print(e)

b = RingBuffer(240)
b.start()
b.continuous_kline(
    pair="btcusdt",
    id=1,
    contractType="perpetual", 
    interval='1m',
    callback=b.message_handler,
)

#%%
f, ax = plt.subplots(figsize=(12, 8))

# %%
print(b.df)
print(b.indicators)
# %%

ax.plot(b.df.date, b.df.c, label="close")


# %%

ax.plot(b.df.date, b.indicators.ema)

ax.plot(b.df.date, b.indicators.iband)
ax.plot(b.df.date, b.indicators.sband)
# %%
ax.clear()
# %%

# ax.lines[0].set_data(b.df.date, b.df.c)
# ax.lines[0].set_data(b.df.date, b.indicators.ema)
# ax.lines[2].set_data(b.df.date, b.indicators.iband)
# ax.lines[3].set_data(b.df.date, b.indicators.sband)
# #%%

# #%%
# ax.lines[0].draw(ax.plot(b.df.date, b.df.c))
# ax.lines[1].draw(ax.plot(b.df.date,b.indicators.ema))
# ax.lines[2].draw(ax.plot(b.df.date,b.indicators.iband))
# ax.lines[3].draw(ax.plot(b.df.date,b.indicators.sband))
#%%

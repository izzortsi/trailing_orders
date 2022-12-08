
# %%

import time
import logging
from binance.lib.utils import config_logging
from binance.websocket.futures.websocket_client import FuturesWebsocketClient as Client

config_logging(logging, logging.DEBUG)
from binance.websocket.futures.websocket_client import FuturesWebsocketClient
import pandas as pd

class RingBuffer(FuturesWebsocketClient):
    def __init__(self, size):
        self.df = pd.DataFrame()
        self.size = size

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
    interval='4h',
    callback=b.message_handler,
)

# %%

b.stop()

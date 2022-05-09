# %%

import time
import logging
from binance.lib.utils import config_logging
from binance.websocket.futures.websocket_client import FuturesWebsocketClient as Client

config_logging(logging, logging.DEBUG)
from binance.websocket.futures.websocket_client import FuturesWebsocketClient
import pandas as pd

class RingBuffer(FuturesWebsocketClient):
    def __init__(self, size, data=None):
        if data is None:
            self.data = {}
        else:
            self.data = data
        self.df = {}
        self.size = size

        super().__init__()
        

    def message_handler(self, message):
        try:
            if type(message) == list:
                for item in message:
                    if item["e"] == "24hrTicker":
                        sym = item["s"]
                        if sym in self.df.keys():
                            new_row = pd.DataFrame([   
                                        {
                                        "s": item["s"],
                                        "date": pd.to_datetime(item["E"], unit="ms"),
                                        "o": pd.to_numeric(item["o"]),
                                        "h": pd.to_numeric(item["h"]),
                                        "l": pd.to_numeric(item["l"]),
                                        "c": pd.to_numeric(item["c"]),
                                        "v": pd.to_numeric(item["v"]),
                                        }])                                
                            if len(self.df[sym]) < self.size:
                             
                                self.df[sym] = pd.concat([
                                        self.df[sym], 
                                        new_row], 
                                        ignore_index = True,
                                        )
                            elif len(self.df[sym]) >= self.size:
                                self.df[sym].drop(axis=0, index = 0, inplace=True)                                                                
                                self.df[sym] = pd.concat([
                                        self.df[sym], 
                                        new_row], 
                                        ignore_index = True,
                                )
                                # print(len(self.df[sym]))                                
                        else:
                            self.df[sym] = pd.DataFrame([   
                                        {
                                        "s": item["s"],
                                        "date": pd.to_datetime(item["E"], unit="ms"),
                                        "o": pd.to_numeric(item["o"]),
                                        "h": pd.to_numeric(item["h"]),
                                        "l": pd.to_numeric(item["l"]),
                                        "c": pd.to_numeric(item["c"]),
                                        "v": pd.to_numeric(item["v"]),
                                        },
                                        ]) 
        except Exception as e:
            print(e)
        # finally:
        #     print(self.data)

b = RingBuffer(10)
b.start()
b.ticker(
    id=1,
    callback=b.message_handler,
)


# %%

btc_df = b.df["BTCUSDT"]
len(b.df["BTCUSDT"])
# %%

len(b.df["BTCUSDT"])

# %%
btc_df.drop(axis=0, index = 0, inplace=True)
# %%
btc_df
# %%

len(b.df)

b.stop()
#%%

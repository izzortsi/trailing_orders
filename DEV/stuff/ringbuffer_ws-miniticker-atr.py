
# %%

import time
import logging
from binance.lib.utils import config_logging


config_logging(logging, logging.DEBUG)

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import logging
from binance.lib.utils import config_logging
from binance.websocket.um_futures.websocket_client import UMFuturesWebsocketClient as FuturesWebsocketClient
import pandas_ta as ta




class RingBuffer(FuturesWebsocketClient):
    def __init__(self, size, data=None):
        if data is None:
            self.data = []
        else:
            self.data = data
        self.df = pd.DataFrame()
        self.size = size

        super().__init__()
        

    def message_handler(self, message):
        try:
            # print(message)
            # print(type(message))
            if (message["e"] == "24hrMiniTicker") and (len(self.df) < self.size):
                self.df = pd.concat([
                        self.df, 
                        pd.DataFrame([   
                            {
                            "s": message["s"],
                            "date": pd.to_datetime(message["E"], unit="ms"),
                            "o": pd.to_numeric(message["o"]),
                            "h": pd.to_numeric(message["h"]),
                            "l": pd.to_numeric(message["l"]),
                            "c": pd.to_numeric(message["c"]),
                            "v": pd.to_numeric(message["v"]),
                            },
                            ])], 
                            ignore_index = True,
                        )
            elif (message["e"] == "24hrMiniTicker") and (len(self.df) >= self.size):
                self.df.drop(axis=0, index = 0, inplace=True), 
                self.df = pd.concat([
                        self.df, 
                        pd.DataFrame([   
                            {
                            "s": message["s"],
                            "date": pd.to_datetime(message["E"], unit="ms"),
                            "o": pd.to_numeric(message["o"]),
                            "h": pd.to_numeric(message["h"]),
                            "l": pd.to_numeric(message["l"]),
                            "c": pd.to_numeric(message["c"]),
                            "v": pd.to_numeric(message["v"]),
                            },
                            ])], 
                            ignore_index = True,
                        )
                print(self.df.iloc[[-1]])
            # item = message                  [message]["c"]  
            # self.data[item['symbol']] = item['price']
        except Exception as e:
            print(e)
        # finally:
        #     print(self.data)

b = RingBuffer(240)
b.start()
b.mini_ticker(
    id=1,
    callback=b.message_handler,
    symbol="btcusdt"
)


# %%



# %%

b.stop()
#%%

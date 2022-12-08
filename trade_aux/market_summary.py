# %%

import time
import logging
import pandas_ta as ta
from binance.lib.utils import config_logging
from binance.websocket.um_futures.websocket_client import UMFuturesWebsocketClient as FuturesWebsocketClient

config_logging(logging, logging.DEBUG)

import pandas as pd

class RingBuffer(FuturesWebsocketClient):
    def __init__(self, size, data=None):
        self.df = {}
        self.size = size

        super().__init__()
        if data is None:
            self.data = None
        else:
            self.data = data
        
    def compute_indicators(self):
        
        for sym in self.df.keys():      
            try:
                if self.data is None:
                    self.data = {sym: pd.DataFrame() for sym in self.df.keys()}
                elif sym not in self.data.keys():
                    self.data[sym] = pd.DataFrame()                    
                    self.data[sym]["diff_1"] = self.df[sym]["c"].diff(1)
                    self.data[sym]["diff_2"] = self.df[sym]["c"].diff(1) - self.df["c"].diff(2)
                    self.data[sym]["sma"] = self.df[sym]["c"].rolling(5).mean()
                    self.data[sym]["ema"] = self.df[sym]["c"].ewm(span=5, adjust=False).mean()

                else:
                    self.data[sym]["diff_1"] = self.df[sym]["c"].diff(1)
                    self.data[sym]["diff_2"] = self.df[sym]["c"].diff(1) - self.df["c"].diff(2)
                    self.data[sym]["sma"] = self.df[sym]["c"].rolling(5).mean()
                    self.data[sym]["ema"] = self.df[sym]["c"].ewm(span=5, adjust=False).mean()
            except:
                pass

    def message_handler(self, message):
        try:
            if type(message) == list:
                for item in message:
                    if item["e"] == "24hrTicker":
                        sym = item["s"]
                        # print(sym)
                        # print("USDT" == sym[-4:])
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
                                self.compute_indicators()
                            elif len(self.df[sym]) >= self.size:
                                self.df[sym].drop(axis=0, index = 0, inplace=True)                                                                
                                self.df[sym] = pd.concat([
                                        self.df[sym], 
                                        new_row], 
                                        ignore_index = True,
                                )
                                self.compute_indicators()
                                # print(len(self.df[sym]))                                
                        else:
                            if ("USDT" == sym[-4:]):
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

b = RingBuffer(100)
b.start()
b.ticker(
    id=1,
    callback=b.message_handler,
)


# %%

btc_df = b.df["BTCUSDT"]
len(b.df), len(b.df["BTCUSDT"])

# %%
print(b.df.keys())

# %%

b.stop()
#%%


# %%

import time
import logging
from binance.lib.utils import config_logging
from binance.websocket.um_futures.websocket_client import UMFuturesWebsocketClient as FuturesWebsocketClient

config_logging(logging, logging.DEBUG)

import pandas as pd
import pandas_ta as ta

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

b = RingBuffer(90)
b.start()
b.ticker(
    id=1,
    callback=b.message_handler,
)

import threading

class StreamProcesser(threading.Thread):
    
    def __init__(self, ringbuffer):
        super().__init__(daemon=True)
        self.ringbuffer = ringbuffer
        self.processed_data = {}
        self.run()
        
    def run(self):
        while True:
            time.sleep(1)
            self._process()
            print(self.processed_data)
    def _process(self):
        
        for sym in self.ringbuffer.df.keys():
            self.processed_data[sym] = pd.DataFrame()
            self.processed_data[sym]["ema"] = self.ringbuffer.df[sym]["c"].ewm(span=30).mean()
            self.processed_data[sym]["std"] = self.ringbuffer.df[sym]["c"].ewm(span=30).std()
            self.processed_data[sym]["pstd"] = self.processed_data[sym]["std"]/self.processed_data[sym]["ema"]
            # self.processed_data[sym]["pstd_diff"] = \
            #     (self.processed_data[sym]["c"] - self.processed_data[sym]["ema"])/self.processed_data[sym]["std"]
        

sproc = StreamProcesser(b)


# sproc.process()
# sproc.processed_data
time.sleep(10)

while True:
    time.sleep(1)
    # print([sproc.processed_data[k].iloc[-1] for k in sproc.processed_data.keys()])
    # print(b.df.keys())
    for sym in b.df.keys():
        print(sym, b.df[sym].iloc[-1])
    # print(b.df.iloc[-1])
    # print(sproc.processed_data["BTCUSDT"].iloc[-1][["c", "ema", "std", "pstd"]])
# %%

b.stop()
#%%

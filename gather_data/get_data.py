# %%

import time
import json
import logging
import pandas_ta as ta
from binance.lib.utils import config_logging
from binance.websocket.um_futures.websocket_client import UMFuturesWebsocketClient as FuturesWebsocketClient

config_logging(logging, logging.DEBUG)

import pandas as pd

def get_filters():
    with open("symbols_filters.json") as f:
        data = json.load(f)
    return data

SYMBOLS = get_filters()
class DataBuffer(FuturesWebsocketClient):
    def __init__(self, size):
        super().__init__()

        self.df = {sym: [] for sym in SYMBOLS.keys()}
        self.size = size

    def message_handler(self, message):
        try:
            if type(message) == list:
                for item in message:
                    if item["e"] == "24hrTicker":
                        sym = item["s"]
                        # print(sym)
                        # print("USDT" == sym[-4:])
                        del item["e"]
                        del item["s"]
                        if sym in self.df.keys():                           
                            if len(self.df[sym]) < self.size:
                                self.df[sym].append(item)

                            elif len(self.df[sym]) >= self.size:
                                
                                json_object = json.dumps(self.df[sym], indent=4)
                                from_ts = self.df[sym][0]["E"]
                                to_ts = self.df[sym][-1]["E"]

                                with open(f"{sym}_{from_ts}_{to_ts}.json", "w") as outfile:
                                    outfile.write(json_object)
                            
                                self.df = {sym: [] for sym in SYMBOLS.keys()}
                        else:
                            if ("USDT" == sym[-4:]):
                                self.df[sym].append(item) 
        except Exception as e:
            print(e)
        # finally:
        #     print(self.data)

b = DataBuffer(10)
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


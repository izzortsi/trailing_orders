# %%

import time
import os
import sys
import certifi
import win32api
import json
import logging

import pandas as pd
import pandas_ta as ta

from os.path import exists, join

from binance.lib.utils import config_logging
from binance.websocket.um_futures.websocket_client import UMFuturesWebsocketClient as FuturesWebsocketClient

config_logging(logging, logging.DEBUG)
os.environ['SSL_CERT_FILE'] = certifi.where()

cwd = os.getcwd()
data_dir = join(cwd, "gathered_data")

if not exists(data_dir):
    os.mkdir(join(cwd, "gathered_data"))

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
                        # item["date"] = str(pd.to_datetime(item["E"], unit="ms"))
                        # print(item["date"])
                        if sym in self.df.keys():                           
                            if len(self.df[sym]) < self.size:
                                self.df[sym].append(item)

                            elif len(self.df[sym]) >= self.size:
                                
                                json_object = json.dumps(self.df[sym], indent=4)
                                from_ts = self.df[sym][0]["E"]
                                to_ts = self.df[sym][-1]["E"]
                                sym_data_dir = join(data_dir, sym)

                                if not exists(sym_data_dir):
                                    os.mkdir(sym_data_dir)
                                with open(join(sym_data_dir, f"{sym}_{from_ts}_{to_ts}.json"), "w") as outfile:
                                    outfile.write(json_object)

                                pd.DataFrame(self.df[sym]).apply(pd.to_numeric).to_csv(join(sym_data_dir, f"{sym}_{from_ts}_{to_ts}.csv"))

                                self.df = {sym: [] for sym in SYMBOLS.keys()}
                        else:
                            if ("USDT" == sym[-4:]):
                                print(sym)
                                self.df[sym].append(item) 
        except Exception as e:
            print(e)
        # finally:
        #     print(self.data)

if __name__ == "__main__":
    if len(sys.argv) == 2:
        size = int(sys.argv[1])
        b = DataBuffer(size)
        b.start()
        b.ticker(id=1, callback=b.message_handler)
    else:            
        b = DataBuffer(500)
        b.start()
        b.ticker(id=1, callback=b.message_handler)
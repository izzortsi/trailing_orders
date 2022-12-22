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
        self._len_list = [0 for _ in SYMBOLS.keys()]

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
                            # print(sym)                           
                            if len(self.df[sym]) < self.size:
                                self.df[sym].append(item)

                            elif len(self.df[sym]) >= self.size:
                                print(f"saving data for {sym}")
                                json_object = json.dumps(self.df[sym], indent=4)
                                from_ts = self.df[sym][0]["E"]
                                to_ts = self.df[sym][-1]["E"]
                                sym_data_dir = join(data_dir, sym + f"_{self.size}")

                                if not exists(sym_data_dir):
                                    os.mkdir(sym_data_dir)

                                path_to_file = join(sym_data_dir, f"{sym}_{from_ts}_{to_ts}.json")    
                                with open(path_to_file, "w") as outfile:
                                    outfile.write(json_object)

                                pd.DataFrame(self.df[sym]).apply(pd.to_numeric).to_csv(path_to_file.replace(".json", ".csv"))

                                self.df[sym] = []
                        else:
                            # raise Exception(f"Symbol {sym} not in SYMBOLS")
                            if ("USDT" == sym[-4:]):
                                print(sym)
                                self.df[sym].append(item) 
        except Exception as e:
            print(e)
        # finally:
        #     print(self.data)

    @property
    def len_list(self):
        self._len_list = [len(l) for l in self.df.values()]
        return self._len_list
    # @len_list.setter
    # def len_list(self, value):
    #     assert type(value) == list
    #     self._len_list = value
    # @len_list.getter
    # def len_list(self):
    #     return self._len_list

if __name__ == "__main__":
    if len(sys.argv) == 2:
        size = int(sys.argv[1])
        b = DataBuffer(size)
        b.start()
        b.ticker(id=1, callback=b.message_handler)
    else:            
        b = DataBuffer(1000)
        b.start()
        b.ticker(id=1, callback=b.message_handler)
# %%

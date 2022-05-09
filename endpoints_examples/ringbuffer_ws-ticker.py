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
            # print(message)
            print(type(message))
            if type(message) == list:
                # print(message)
                for item in message:
                    if item["e"] == "24hrTicker":
                        # print(item["s"])
                        sym = item["s"]
                        print(sym)
                        self.data[sym] = self.data[sym] + \
                            [pd.DataFrame([   
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
                            ]
                        self.df[sym] = pd.concat([
                                self.df[sym], 
                                pd.DataFrame([   
                                    {
                                    "s": item["s"],
                                    "date": pd.to_datetime(item["E"], unit="ms"),
                                    "o": pd.to_numeric(item["o"]),
                                    "h": pd.to_numeric(item["h"]),
                                    "l": pd.to_numeric(item["l"]),
                                    "c": pd.to_numeric(item["c"]),
                                    "v": pd.to_numeric(item["v"]),
                                    },
                                    ])], 
                                    ignore_index = True,
                                )
                        print(self.data[sym])

            # item = message                  [message]["c"]  
            # self.data[item['symbol']] = item['price']
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
len(b.data)

# %%

len(b.df)

# %%

b.stop()
#%%
b.data
# %%

pd.concat(b.data, ignore_index = True)
# %%

b.df
# %%


# %%


#%%

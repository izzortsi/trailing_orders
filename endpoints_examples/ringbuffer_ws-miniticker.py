
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
            self.data = []
        else:
            self.data = data
        self.df = pd.DataFrame()
        self.size = size

        super().__init__()
        

    def message_handler(self, message):
        try:
            print(message)
            print(type(message))
            if message["e"] == "24hrMiniTicker":
                self.data = self.data + \
                    [pd.DataFrame([   
                        {
                        "s": message["s"],
                        "date": pd.to_datetime(message["E"], unit="ms"),
                        "o": pd.to_numeric(message["o"]),
                        "h": pd.to_numeric(message["h"]),
                        "l": pd.to_numeric(message["l"]),
                        "c": pd.to_numeric(message["c"]),
                        "v": pd.to_numeric(message["v"]),
                        },
                    ])
                    ]
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

            # item = message                  [message]["c"]  
            # self.data[item['symbol']] = item['price']
        except Exception as e:
            print(e)
        # finally:
        #     print(self.data)

b = RingBuffer(10)
b.start()
b.mini_ticker(
    id=1,
    callback=b.message_handler,
    symbol="btcusdt"
)


# %%
len(b.data)


# %%

b.stop()
#%%

# %%

pd.concat(b.data, ignore_index = True)
# %%

b.df
# %%


# %%


#%%

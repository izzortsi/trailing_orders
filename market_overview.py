# %%

import time
import logging
import numpy as np
from binance.lib.utils import config_logging
from binance.websocket.futures.websocket_client import FuturesWebsocketClient as Client
# from multiprocessing import Process, Queue

config_logging(logging, logging.DEBUG)




#%%
raw_data = []
out_data = {}

# %%

def write_data(m, raw_data):
    raw_data.append(m)
    time.sleep(1)

client = Client()
client.start()
client.ticker(
        id=13,
        callback= lambda m: write_data(m, raw_data),
        )


#%%
def process_data(raw_data, out_data):
    if len(raw_data[-1]) > 0:
        for event in raw_data[-1]:
            try:
                out_data[event['s']] = event['c']
            except:
                pass


# %%

stream_processer = Process(target=process_data, args=(raw_data, out_data, ))
stream_processer.start()
#%%
raw_data
#%%

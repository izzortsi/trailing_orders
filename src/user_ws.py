
# %%

import time
import logging
import os
from binance.lib.utils import config_logging
from binance.um_futures import UMFutures as Client
from binance.websocket.um_futures.websocket_client import UMFuturesWebsocketClient

import os
import sys
import certifi
import win32api

os.environ['SSL_CERT_FILE'] = certifi.where()

config_logging(logging, logging.DEBUG)

def message_handler(message):
    try:
        if "e" in message.keys():
            print(message["e"])
            if message["e"] == "ORDER_TRADE_UPDATE":
                order_data = message["o"]
                # print(message)
                print(order_data["s"])
                print(order_data["X"])
                
    except Exception as e:
        print(e)


key = os.environ.get('API_KEY'); print(key)
sec = os.environ.get('API_SECRET'); print(sec)


client = Client(key=key, secret=sec)
response = client.new_listen_key()
print(client.account())

logging.info("Receving listen key : {}".format(response["listenKey"]))
#%%


ws_client = UMFuturesWebsocketClient()
ws_client.start()
ws_client.user_data(
    listen_key=response["listenKey"],
    id=1,
    callback=message_handler,
)



logging.debug("closing ws connection")
# ws_client.stop()
#%%

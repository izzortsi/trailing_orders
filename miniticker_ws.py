
# %%

import time
import logging
from binance.lib.utils import config_logging
from binance.websocket.futures.websocket_client import FuturesWebsocketClient as Client

config_logging(logging, logging.DEBUG)


def message_handler(message):
    print(message)


my_client = Client()
my_client.start()

my_client.mini_ticker(
    id=1,
    callback=message_handler,
    symbol="btcusdt"
)

time.sleep(5)

logging.debug("closing ws connection")
my_client.stop()
#%%

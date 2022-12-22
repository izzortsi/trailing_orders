#%%


import time
from binance.websocket.cm_futures.websocket_client import CMFuturesWebsocketClient

def message_handler(message):
    print(message)

ws_client = CMFuturesWebsocketClient()
ws_client.start()

ws_client.mini_ticker(
    symbol='bnbusdt',
    id=1,
    callback=message_handler,
)

# Combine selected streams
ws_client.instant_subscribe(
    stream=['bnbusdt@bookTicker', 'ethusdt@bookTicker'],
    callback=message_handler,
)

time.sleep(10)

print("closing ws connection")
ws_client.stop()

# %%

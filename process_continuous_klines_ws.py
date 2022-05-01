# %%
import time
import logging
from binance.lib.utils import config_logging
from binance.websocket.futures.websocket_client import FuturesWebsocketClient as Client
import pandas as pd
import matplotlib
import numpy as np
config_logging(logging, logging.DEBUG)

def message_handler(message, df):
    try:
      row = message["k"]
      row["time"] = message["E"]
      if len(df) < 52:      
        df.append(row)
      else:
        df.pop(0)
        df.append(row)
    except:
      print(message)

    # print(dfs)

df = []

my_client = Client()
my_client.start()


my_client.continuous_kline(
    pair="btcusdt",
    id=1,
    contractType="perpetual", 
    interval='1d',
    callback=lambda m: message_handler(m, df),
)

time.sleep(10)

logging.debug("closing ws connection")
my_client.stop()

#%%
def process_ws_klines(kl_list):

  df = pd.DataFrame(kl_list)
  df.drop(labels=["i", "f", "L", "B"], axis=1, inplace=True)
  
  reordered_labels = ['time', 't', 'o', 'h', 'l', 'c', 'v', 'T', 'q', 'n', 'V', 'Q']
  
  df = df.reindex(columns=reordered_labels)
  df.columns = ['time', 'open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'trades', 'taker_buy_volume', 'taker_buy_quote_asset_volume']
  df['time'] = pd.to_datetime(df['time'], unit='ms')
  df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
  df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
  df['open'] = pd.to_numeric(df['open'])
  df['high'] = pd.to_numeric(df['high'])
  df['low'] = pd.to_numeric(df['low'])
  df['close'] = pd.to_numeric(df['close'])
  df['volume'] = pd.to_numeric(df['volume'])
  df['quote_asset_volume'] = pd.to_numeric(df['quote_asset_volume'])
  df['trades'] = pd.to_numeric(df['trades'])
  df['taker_buy_volume'] = pd.to_numeric(df['taker_buy_volume'])
  df['taker_buy_quote_asset_volume'] = pd.to_numeric(df['taker_buy_quote_asset_volume'])

  return df


#%%

df = process_ws_klines(kl_list=df)
#%%
df
#%%

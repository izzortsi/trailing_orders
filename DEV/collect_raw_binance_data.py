
# %%

import dateparser
import json
import os
from datetime import datetime
from binance.um_futures import UMFutures as Client
import pandas as pd
from config import *
from auxs_funcs import *



# symbol = "BNBUSDT"
#PARAMETERS


ROLLING_WINDOW_LENGTH = 12 #this is correlated with the chosen TIMEFRAME and as such should be a function of that; something about the frequency of local minima and maxima
CALLBACKRATE_FACTOR = 10
PAIR = "ETHUSDT"
TIMEFRAME = "6h"

FROM_DATE = "2018-01-17"
TO_DATE = "2019-01-17"

BLOCK = 512
DATA_WINDOW_LENGTH = BLOCK



# %%

from_date = dateparser.parse(FROM_DATE)
print(from_date)
from_ts = datetime.timestamp(from_date)
print(from_ts)
# %%
pd.to_timedelta(TIMEFRAME)
# pd.to_datetime(TIMEFRAME)

# %%


path = os.path.join(DATA_DIR, PAIR)
if not os.path.exists(path):
    os.mkdir(path)

tk1 = os.path.join(PAIR, f"{BLOCK}_{FROM_DATE};{TIMEFRAME};{TO_DATE}")

path = os.path.join(DATA_DIR, tk1)
if not os.path.exists(path):
    os.mkdir(path)


# %%

def collect_data(PAIR, from_date, to_date, BLOCK, multiplier=7, timeframe ="1m"):
    
    from_date = dateparser.parse(from_date)
    to_date = dateparser.parse(to_date)
    from_ts = round(datetime.timestamp(from_date)*1000)
    to_ts = round(datetime.timestamp(to_date)*1000)

    client = Client()
    klines_gen = client.continuous_klines(pair=PAIR, contractType="PERPETUAL", interval=timeframe, startTime=from_ts, endTime=to_ts, limit=BLOCK)
    # get_historical_klines_generator(
    #                 PAIR, TIMEFRAME, from_ts, end_str=to_ts
    #                 )
    
    tk1 = os.path.join(PAIR, f"{BLOCK}_{FROM_DATE};{TIMEFRAME};{to}")
    path = os.path.join(DATA_DIR, tk1)
    
    data=[[]]
    c = 1
    for i, kline in enumerate(klines_gen):
        data[-1].append(list(map(float, kline)))
        # print(data[-1])
        if i % BLOCK == 1:
            dump_data(data[-1], path, c)
            data.append([])
            c += 1
    return data

def dump_data(data, path, i):


    file = os.path.join(path, f"{i}.json")
    with open(file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def dump_raw_data(data, path):

    for i, BLOCK in enumerate(data):
        file = os.path.join(path, f"{i}.json")
        with open(file, 'w', encoding='utf-8') as f:
            json.dump(BLOCK, f, ensure_ascii=False, indent=4)

# %%


if __name__=="__main__":
    data = collect_data(PAIR, FROM_DATE, to, BLOCK, timeframe=TIMEFRAME)
    # dump_raw_data(data, block, fromt, timespan)
#%%
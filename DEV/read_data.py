
# %%
from collect_raw_binance_data import *
from config import *
from auxs_funcs import *
import pandas as pd

# %%

# symbol = "BNBUSDT"
# fromt = "2017-01-17"
# to = "2021-01-17"
# block = 512
# timespan = "1m"
# f"{block}_{fromt};{timespan};{to}"

data_path = os.path.join(DATA_DIR, "ETHUSDT", "512_2018-01-17;1m;2021-01-17")

#%%
# C:\Users\igor-\Dropbox\PC\Documents\GitHub\botkz\botkz\datasets\BNBUSDT\512_2017-01-17;1m;2021-01-17\3242.json
def compose_data(symbol, fromt, to, block, timespan, path):
    """
    Compose data from binance.
    """
    # data_path = path
    # data_name = f"{block}_{fromt};{timespan};{to}"
    struct_data = []
    flat_data = []
    json_files = [pos_json for pos_json in os.listdir(path) if pos_json.endswith('.json')]
    print(json_files)
    for i, J in enumerate(json_files):
        
        with open(os.path.join(path, J)) as json_file:
            # data = json.load(json_file)
            file_data = json.load(json_file)
            flat_data += file_data
            struct_data.append(file_data)
            print(struct_data[-1])
    return struct_data, flat_data

#%%
# struct_data, flat_data = compose_data(symbol, fromt, to, block, timespan, data_path)


def to_lhocv(symbol, fromt, to, block, timespan, data_path):
    _, flat_data = compose_data(symbol, fromt, to, block, timespan, data_path)
    lhocv = [[data[0], float(data[3]), float(data[2]), float(data[1]), float(data[4]), float(data[5])] for data in flat_data]
    labels = ["date", "low", "high", "open", "close", "volume"]
    return lhocv, labels

def to_ohlcv(symbol, fromt, to, block, timespan, data_path):
    _, flat_data = compose_data(symbol, fromt, to, block, timespan, data_path)
    lhocv = [[data[0], float(data[1]), float(data[2]), float(data[3]), float(data[4]), float(data[5])] for data in flat_data]
    labels = ["date", "open", "high", "low", "close", "volume"]
    return lhocv, labels

# datetime.fromtimestamp(lhocv[0][0]/1000)
#%%

to_ohlcv(symbol, fromt, to, block, timespan, data_path)
#%%

lhocv, labels = to_lhocv(symbol, fromt, to, block, timespan, data_path)

#%%
len(lhocv[0])
#%%
#READ DATA
data_path = os.path.join(NEW_DATA_DIR, f"{PAIR}_{TIMEFRAME}")
print(data_path)       
labels = pd.read_csv(data_path + "_labels.csv")
labels.drop(['Unnamed: 0'], axis=1, inplace=True)
features = pd.read_csv(data_path + "_features.csv")
features.drop(['Unnamed: 0'], axis=1, inplace=True)
print(labels)
print(features)
import os

akey = os.environ.get("API_KEY")
asec = os.environ.get("API_SECRET")

PREFIX = os.getcwd()
SYMBOLS = ["BTCUSD, ETHUSD, SOLUSD, BNBUSD, ADAUSD"]
BLOCK = 512
DUMP = '.csv'
SYMBOL = "ETHUSDT"
DATA_DIR=os.path.join(PREFIX, "datasets")

if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)

# SYMBOL = "ETHUSD"
# FROM = "2021-01-01"
# TO = "2021-05-01"
# BLOCK = 64
# INTERVAL = "minute"
# AGG = 7

# symbol = "ETHUSDT"
# fromt = "2018-01-20"
# to = "2021-11-19"
# timespan = "1m"



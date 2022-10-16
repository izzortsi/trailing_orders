import os


PREFIX = os.getcwd()
SYMBOLS = ["BTCUSD, ETHUSD, SOLUSD, BNBUSD, ADAUSD"]
# SYMBOL = "ETHUSD"
# FROM = "2021-01-01"
# TO = "2021-05-01"
# BLOCK = 64
# INTERVAL = "minute"
# AGG = 7

symbol = "ETHUSDT"
fromt = "2018-01-20"
to = "2021-11-19"
block = 512
timespan = "1m"

DATA_DIR=os.path.join(PREFIX, "datasets")
if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)


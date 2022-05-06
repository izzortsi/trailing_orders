
# %%

import websocket, json
from datetime import datetime
import time as tm
import telebot
#from pandas.errors import EmptyDataError
import xlwings as xw
#from binance.client import Client as Clientas
import logging
from binance.lib.utils import config_logging
from binance.websocket.futures.websocket_client import FuturesWebsocketClient as Client
import datetime # wasn't imported
import time # wasn't imported
config_logging(logging, logging.DEBUG)

# %%

def calc(templtp,openn,close,high,low,vol):
    try:
        templtp,tempopen,templtp,temphigh,templow,tempvol = templtp,openn,close,high,low,vol
    except Exception as e:
        print("Exception occured: ",e)
        pass
    return long,longtp1,longtp2,longtp3,longsl,short,shorttp1,shorttp2,shorttp3,shortsl      

global long,longtp1,longtp2,longtp3,longsl,short,shorttp1,shorttp2,shorttp3,shortsl,date,price
longfl = 0
shortfl = 0
securelongtp = 0
secureshortp = 0
securelongsl = 0
lot=0
secureshorsl = 0
long_repeat = 0
short_repeat = 0
tempprev=0
long=0
longtp1=0
longtp2=0
longtp3=0
longsl=0
short=0
shorttp1=0
shorttp2=0
shorttp3=0
shortsl=0
date=0
price=0
#Client.API_URL = "https://testnet.binance.vision/api"
api_key = ''
api_secret = ''


def message_handler(message):
    global longfl,shortfl,securelongtp,secureshortp,securelongsl,secureshorsl,short_repeat,long_repeat,tempprev,lot
    global long,longtp1,longtp2,longtp3,longsl,short,shorttp1,shorttp2,shorttp3,shortsl,date,price

    try:
        candle = message['k']
        is_candle_closed = candle['x']
        openn = candle['o']
        openn = float(openn)
        close = candle['c']
        close = float(close)
        high = candle['h']
        high =float(high)
        low = candle['l']
        low = float(low)
        vol = candle['v'] 
        vol = float(vol)
        time = (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        priceopen = openn
        balance = 0
        from binance.client import Client

        client = Client(api_key,api_secret)
        print("logged in")

        if longfl == 0 and shortfl == 0 and lot ==0:
            arrr = client.aggregate_trade_iter(symbol='BTCUSDT', start_str='2 minutes ago GMT+5:30')
            templtp = next(arrr)['p']
            templtp = float(templtp) 
            long,longtp1,longtp2,longtp3,longsl,short,shorttp1,shorttp2,shorttp3,shortsl=calc(templtp,openn,close,high,low,vol)
            print(long,longtp1,longtp2,longtp3,longsl,short,shorttp1,shorttp2,shorttp3,shortsl)
            lot=1
            dateopen = (datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        #long entry
        if ((priceopen > long) and (priceopen < (long + 15)) and shortfl == 0 and longfl == 0) and (lot==1):
            print(f"price: {priceopen}")
            dateopen = (datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            print(f"date: {dateopen}")
            securelongtp = longtp1;
            securelongsl = longsl;
            balance = 49.9925 / close
            print("balance bought: ",balance)
            print(f'Entered long at --> Price : ${priceopen}, TP : ${securelongtp},SL: ${securelongsl} ',dateopen)
            longfl = 1
            lot=0

        if longfl == 1 and priceopen >= securelongtp and (lot==1):
            print(f"price: {priceopen}")
            dateopen = (datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            print(f"date: {dateopen}")
            print(f'Trade Exit Long at TP --> ${priceopen} ', dateopen)
            longfl = 0
            lot=0

        if longfl == 1 and priceopen < securelongsl and (lot==1):
            print(f"price: {priceopen}")
            dateopen = (datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            print(f"date: {dateopen}")
            print(f'Trade Exit Long at SL--> ${priceopen}', dateopen)
            longfl = 0
            lot=0

        #short entery 
        if short_repeat != short and priceopen < short and priceopen > (short - 15) and shortfl == 0 and longfl == 0 and (lot==1):
            print(f"price: {priceopen}")
            dateopen = (datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            print(f"date: {dateopen}")
            short_repeat = short
            secureshortp = shorttp1
            secureshorsl = shortsl
            shortfl = 1
            print(f'Entered short at --> Price : ${priceopen},TP : ${secureshortp},SL: ${secureshorsl}', dateopen)
            lot=0

        if shortfl == 1 and priceopen <= secureshortp and (lot==1):
            print(f"price: {priceopen}")
            dateopen = (datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            print(f"date: {dateopen}")
            print(f'Trade Exit Short at TP--> ${priceopen} ',dateopen)
            shortfl = 0
            lot=0

        if shortfl == 1 and priceopen > secureshorsl and (lot==1):
            print(f"price: {priceopen}")
            dateopen = (datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            print(f"date: {dateopen}")
            print(f'Trade Exit Short at SL--> ${priceopen}',dateopen)
            shortfl = 0
            lot=0

        if (priceopen < shorttp3) and(lot==0):
            print(candle)
            arrr = client.aggregate_trade_iter(symbol='BTCUSDT', start_str='2 minutes ago GMT+5:30')
            print(arrr)
            templtp = next(arrr)['p']
            templtp = float(templtp)            
            openn = candle['o']
            openn = float(openn)
            close = candle['c']
            close = float(close)
            high = candle['h']
            high =float(high)
            low = candle['l']
            low = float(low)
            vol = candle['v'] 
            vol = float(vol)
            tempdate = time
            long,longtp1,longtp2,longtp3,longsl,short,shorttp1,shorttp2,shorttp3,shortsl=calc(templtp,openn,close,high,low,vol)
            print(long,longtp1,longtp2,longtp3,longsl,short,shorttp1,shorttp2,shorttp3,shortsl)
            print("bcz priceopen < shorttp3")
            dateopen = (datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        if (priceopen > longtp3) and lot==0:
            print(candle)
            arrr = client.aggregate_trade_iter(symbol='BTCUSDT', start_str='2 minutes ago GMT+5:30')
            print(arrr)
            templtp = next(arrr)['p']
            templtp = float(templtp)            
            openn = candle['o']
            openn = float(openn)
            close = candle['c']
            close = float(close)
            high = candle['h']
            high =float(high)
            low = candle['l']
            low = float(low)
            vol = candle['v'] 
            vol = float(vol)
            tempdate = time
            long,longtp1,longtp2,longtp3,longsl,short,shorttp1,shorttp2,shorttp3,shortsl=calc(templtp,openn,close,high,low,vol)
            print(long,longtp1,longtp2,longtp3,longsl,short,shorttp1,shorttp2,shorttp3,shortsl)
            print("bcz priceopen > longtp3")
        print(priceopen)
    except Exception as e:
        print("Exception occured: ",e)
        pass

my_client = Client()
my_client.start()

my_client.continuous_kline(
    pair="btcusdt",
    id=1,
    contractType="perpetual", 
    interval='1m',
    callback=message_handler,
)
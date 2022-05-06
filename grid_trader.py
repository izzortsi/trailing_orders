#%%
from binance.futures import Futures as Client
from binance.lib.utils import config_logging
from binance.error import ClientError
import pandas_ta as ta

import plot_functions as pf
import argparse
import logging
import pandas as pd
import numpy as np
import json
# import matplotlib.pyplot as plt
# from plotly.subplots import make_subplots
# import plotly.graph_objects as go
import os

# %%
parser = argparse.ArgumentParser(description='Sends a trailing stop order for given parameters.')


parser.add_argument('-s', '--symbol', type= str, help='e.g., btc', default="btc")
parser.add_argument('-tf', '--timeframe', type= str, help='one of: 15m, 1h, 4h, 1d', default="4h")
parser.add_argument('-tp', '--take_profit', type= float, help='take profit, in percentage (should be above 0.1% to cover trading fees)', default=5.0)
parser.add_argument('-ap', '--activation_price', type= float, help='directly uses the given activation price to set the exit point', default=0.0)
parser.add_argument('-sl', '--stop_loss', type= float, help='not implemented yet; ignore', default=0.0)
parser.add_argument('-dwl', '--data_window_length', type= int, help='how many candles to query from binance`s API, up to 500', default=50)
parser.add_argument('-rwl', '--rolling_window_length', type=int, help='lenght of the rolling window to compute means and standard deviations', default=4)
parser.add_argument('-crf', '--callback_rate_factor', type=int, help='to explain', default=10)
parser.add_argument('-d', '--position_direction', type=str, help='position direction: LONG or SHORT', default="LONG")
parser.add_argument('-plt', '--plot_stuff', type=bool, help='plot queried data?', default=False)
parser.add_argument('-so', '--send_orders', type=bool, help='actually send the order?', default=False)

args = parser.parse_args()
#%%
#PARAMETERS


PAIR = (args.symbol + "USDT").upper()
TIMEFRAME = args.timeframe
TAKE_PROFIT = args.take_profit
ACTIVATION_PRICE = args.activation_price
DATA_WINDOW_LENGTH = args.data_window_length
ROLLING_WINDOW_LENGTH = args.rolling_window_length
CALLBACKRATE_FACTOR = args.callback_rate_factor
POSITION_DIRECTION = args.position_direction
PLOT_STUFF = args.plot_stuff
SEND_ORDERS = args.send_orders

if POSITION_DIRECTION == "LONG":
    SIDE = "SELL" 
elif POSITION_DIRECTION == "SHORT":
    SIDE = "BUY"
else:
    raise Exception(f"POSITION_DIRECTION must be either 'LONG' or 'SHORT', and is {POSITION_DIRECTION}")

config_logging(logging, logging.WARNING)

def process_klines(klines):

    df = pd.DataFrame(klines)
    df.columns = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'trades', 'taker_buy_volume', 'taker_buy_quote_asset_volume', 'ignore']
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
    df['ignore'] = pd.to_numeric(df['ignore'])
    df.drop(['ignore'], axis=1, inplace=True)
    return df

def compute_indicators(klines, coefs=np.array([0.618, 1.0, 1.618]), w1=12, w2=26, w3=8, w_atr=ROLLING_WINDOW_LENGTH, step=0.0):
    # compute macd
    macd = pd.Series(
        klines["close"].ewm(span=w1, min_periods=w1).mean()
        - klines["close"].ewm(span=w2, min_periods=w2).mean()
    )
    macd_signal = macd.ewm(span=w3, min_periods=w3).mean()
    macd_hist = macd - macd_signal
    # compute atr bands

    atr = ta.atr(klines["high"], klines["low"], klines["close"], length=w_atr)

    sup_grid_coefs = coefs
    inf_grid_coefs = -1.0 * coefs

    hmean = klines.high.ewm(span=w_atr).mean()
    lmean = klines.low.ewm(span=w_atr).mean()
    global_volatility = (((hmean/lmean).mean()-1)*100)
    
    # closes_mean = klines["close"].ewm(span=w_atr, min_periods=w_atr).mean()
    # closes_std = klines["close"].ewm(span=w_atr, min_periods=w_atr).std()
    
    closes_mean = klines.close.ewm(halflife=pd.Timedelta(TIMEFRAME)/4, ignore_na=True, min_periods=ROLLING_WINDOW_LENGTH, times=klines.open_time).mean()
    closes_std = klines.close.ewm(halflife=pd.Timedelta(TIMEFRAME)/4, ignore_na=True, min_periods=ROLLING_WINDOW_LENGTH, times=klines.open_time).std()

    local_volatility = (closes_std/closes_mean).mean()*100
    
    grid_coefs = np.concatenate((np.sort(inf_grid_coefs), sup_grid_coefs))
    atr_grid = [closes_mean + atr * coef for coef in grid_coefs]

    grid_coefs = sup_grid_coefs

    inf_grid = [closes_mean - atr * coef for coef in grid_coefs]
    sup_grid = [closes_mean + atr * coef for coef in grid_coefs]

    return macd_hist, atr, inf_grid, sup_grid, closes_mean, closes_std, atr_grid, local_volatility, global_volatility

def get_open_positions(positions):
    
    open_positions = {}
    for position in positions:
        if float(position["positionAmt"]) != 0.0:
            open_positions[position['symbol']] = {
                # 'direction': position['positionSide'],
                'entry_price': float(position['entryPrice']),
                'upnl': float(position['unrealizedProfit']), 
                'pos_amt': float(position['positionAmt']),
                'leverage': int(position['leverage']),
                }
            print(f"{open_positions[position['symbol']]}");
    return open_positions

def compute_exit_tp(entry_price, target_profit, side, entry_fee=0.04, exit_fee=0.04):
    if side == "LONG":
        exit_price = (
            entry_price
            * (1 + target_profit / 100 + entry_fee / 100)
            / (1 - exit_fee / 100)
        )
    elif side == "SHORT":
        exit_price = (
            entry_price
            * (1 - target_profit / 100 - entry_fee / 100)
            / (1 + exit_fee / 100)
        )
    return exit_price

def send_grid_orders():
    pass


if __name__ == "__main__":

    akey = os.environ.get("API_KEY")
    asec = os.environ.get("API_SECRET")

    futures_client = Client(key = akey, secret= asec)
    
    acc_info = futures_client.account();
    positions = acc_info["positions"];
    open_positions = get_open_positions(positions)
    
    klines = futures_client.continuous_klines(PAIR, 'PERPETUAL', TIMEFRAME, limit=DATA_WINDOW_LENGTH);
    df = process_klines(klines)
    
    macd_hist, atr, inf_grid, sup_grid, closes_mean, closes_std, atr_grid, local_volatility, global_volatility = compute_indicators(df)
    
    avg_pdev = (closes_std/closes_mean).mean() # average percentual deviation from the EMA
    print("avgpdev", avg_pdev)
    
    callback_rate = max(0.1, round(avg_pdev*100/CALLBACKRATE_FACTOR, ndigits=1)) #the callback rate is a fraction of the percentual stdev
    print("cbr:", callback_rate)
    
    if PLOT_STUFF:
        # ax1 = plt.axis(); plt.plot(closes_mean)
        # ax2 = plt.axis(); plt.plot(closes_mean + closes_std)
        # ax3 = plt.axis(); plt.plot(closes_mean - closes_std)
        # plt.show()
        fig = pf.plot_single_atr_grid(df, atr, atr_grid, closes_mean, macd_hist)
        fig.show()

    print("upnl", open_positions[PAIR]["upnl"])
    
    entry_price = open_positions[PAIR]["entry_price"];
    leverage = open_positions[PAIR]["leverage"];
    
    if ACTIVATION_PRICE != 0.0:
        actv_price = ACTIVATION_PRICE
    else:
        actv_price = entry_price*(1+TAKE_PROFIT/(100*leverage))
    print(f"actv_price: {actv_price}")
    quantity = abs(open_positions[PAIR]["pos_amt"])
    print(f"quantity: {quantity}")

    if SEND_ORDERS:
        try:
            response = response = futures_client.new_order(symbol=PAIR, side = SIDE, type= "TRAILING_STOP_MARKET", quantity= quantity, reduceOnly = True, timeInForce="GTC", activationPrice= actv_price, callbackRate=callback_rate)
            # logging.info(response)
        except ClientError as error:
            logging.error(
                "Found error. status: {}, error code: {}, error message: {}".format(
                    error.status_code, error.error_code, error.error_message
                )
            )

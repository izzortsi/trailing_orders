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

def compute_exit(entry_price, target_profit, side, entry_fee=0.04, exit_fee=0.04):
    if side == "BUY":
        exit_price = (
            entry_price
            * (1 + target_profit / 100 + entry_fee / 100)
            / (1 - exit_fee / 100)
        )
    elif side == "SELL":
        exit_price = (
            entry_price
            * (1 - target_profit / 100 - entry_fee / 100)
            / (1 + exit_fee / 100)
        )
    return exit_price

###

def to_datetime_tz(arg, timedelta=-pd.Timedelta("03:00:00"), unit="ms", **kwargs):
    """
    to_datetime_tz(arg, timedelta=-pd.Timedelta("03:00:00"), unit="ms", **kwargs)

    Args:
        arg (float): epochtime
        timedelta (pd.Timedelta): timezone correction
        unit (string): unit in which `arg` is
        **kwargs: pd.to_datetime remaining kwargs
    Returns:
    pd.Timestamp: a timestamp corrected by the given timedelta
    """
    ts = pd.to_datetime(arg, unit=unit)
    return ts + timedelta
    
def get_time_offset(client):
    
    serverTimeSecs = client.get_server_time()["serverTime"]
    localTimeSecs = time.time()*1000
    
    # print(serverTimeSecs) 
    # print(localTimeSecs)
    print("offset:", serverTimeSecs - localTimeSecs)

def get_ping_avg(client, iters):
    
    pingavg = []
    
    for i in range(iters):
        t1 = time.time()
        client.futures_ping()
        t2 = time.time()
        pingavg.append((t2-t1)*1000)
        print((t2-t1)*1000, "ms")
    print("ping average (ms):", sum(pingavg)/len(pingavg))    
    return sum(pingavg)/len(pingavg)



##PRICE/QUANTITY FORMATTING

qty_formatter = lambda ordersize, qty_precision: f"{float(ordersize):.{qty_precision}f}"
price_formatter = lambda price, price_precision: f"{float(price):.{price_precision}f}"

def get_filters():
    with open("symbols_filters.json") as f:
        data = json

def apply_symbol_filters(filters, base_price, qty=1.2):
    price_precision = int(filters["pricePrecision"])    
    qty_precision = int(filters["quantityPrecision"])
    min_qty = float(filters["minQty"])
    step_size = float(filters["tickSize"])
    # print("price_precision", price_precision, "qty_precision", qty_precision, "min_qty", min_qty, "step_size", step_size)
    minNotional = 7
    min_qty = max(minNotional/base_price, min_qty)
    # print("minqty:", min_qty)
    order_size = qty * min_qty
    # print("ordersize", order_size)

    return price_precision, qty_precision, min_qty, order_size, step_size
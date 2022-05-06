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

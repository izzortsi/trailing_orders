from datetime import datetime



print_pretty = lambda dt, result: print(f"{dt}\n\tO: {result['o']}\n\tH: {result['h']}\n\tL: {result['l']}\n\tC: {result['c']}")

def ts_to_datetime(ts) -> str:
    return datetime.fromtimestamp(ts / 1000.0).strftime('%Y-%m-%d %H:%M')

def ts_to_datetime(ts) -> str:
    return datetime.fromtimestamp(ts / 1000.0).strftime('%Y-%m-%d %H:%M')

def tdelta_timespan(tdelta, timespan, off=0):
    days, secs, msecs = tdelta.days, tdelta.seconds, tdelta.microseconds

    if timespan == "minute":
        return days*(24*60*60) + secs//60        
    if timespan == "hour":
        return days*24 + secs//3600
        

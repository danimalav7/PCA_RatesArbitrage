from ib_insync import *
import pandas as pd

ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)

# Test with 10Yr Treasury futures as a proxy
contract = Bond(symbol='912810TM0', exchange='SMART', currency='USD')
ib.qualifyContracts(contract)

bars = ib.reqHistoricalData(
    contract,
    endDateTime='',
    durationStr='10 Y',
    barSizeSetting='1 day',
    whatToShow='MIDPOINT',
    useRTH=True
)

df = util.df(bars)
print(f"Data range: {df['date'].min()} → {df['date'].max()}")
print(f"Total bars: {len(df)}")

ib.disconnect()
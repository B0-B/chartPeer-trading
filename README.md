# chartPeer---tradingview-monitoring-panel
To declare an arbitrary amount of assets/instruments from your watch list use tradingview-compliant
symbols in the configuration section in index.py. Also you can customize the colors and 'epoch' (total time - 12 months is default).
- No dependencies
- Quick and Useful for day traders

```python
#!/usr/bin/env python3

#======= Configurations =======#
config = {
    "assets": [
        "KRAKEN:XBTEUR",
        "NASDAQ:AAPL",
        "NASDAQ:TSLA",
        "AMEX:SPY",
        "KRAKEN:DOTEUR",
        "KRAKEN:XRPEUR",
        "KRAKEN:AAVEEUR",
        "KRAKEN:LSKEUR",
        "KRAKEN:ICXEUR"
    ],
    "color": "#37a6ef",
    "colorShade": "rgba(255, 255, 0, 0.15)",
    "epoch": "12M",
    "background": "#0d0d0d"
}
#==============================#
...
```
To launch the browser app prompt in the root directory
```python 
python index.py
```

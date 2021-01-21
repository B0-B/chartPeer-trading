# chartPeer :chart:
## A decent & scalable chart panel monitor
To declare an arbitrary amount of assets/instruments (watch list) insert tradingview-compliant (MARKET:SYMBOL e.g. "NASDAQ:AAPL")
symbols in the configuration section in index.py. Also you can customize the colors and 'epoch' (total time - 12 months is default).


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
## < 7kB 
:heavy_check_mark: Chrome/Firefox <br>
:heavy_check_mark: No Dependencies <br>
:heavy_check_mark: Frame Embedment <br>
:heavy_check_mark: Customizable <br>
:heavy_check_mark: Quick and Useful for day trading <br>

## Usage

Long story short:
```bash 
~$ python chartPeer.py
```

# ChartPeer-Tradingview-Monitoring-Panel   



<img src="demo.gif" width="780" height="440" />

<br>

## A decent & scalable chart panel monitor
To declare an arbitrary amount of assets/instruments (watch list) insert tradingview-compliant (MARKET:SYMBOL e.g. "NASDAQ:AAPL")
symbols in the configuration section in `config.yml`. Customize the colors and 'epoch' (total time - 12 months is default) by your taste.


```yaml
assets:
  # put any asset on your watch list
  # search for assets here: https://www.tradingview.com/widget/technical-analysis/
  - KRAKEN:XBTEUR
  - KRAKEN:TRXEUR
  - KRAKEN:DOTEUR
  - KRAKEN:XRPEUR
  - KRAKEN:AAVEEUR
  - KRAKEN:LSKEUR
  - KRAKEN:ICXEUR
  - KRAKEN:MLNEUR
  - KRAKEN:XLMEUR
  - KRAKEN:SCEUR
  - KRAKEN:NANOEUR
  - KRAKEN:ADAEUR
  - KRAKEN:MLNEUR
  - KRAKEN:XMREUR
  - KRAKEN:GRTEUR
  - KRAKEN:KNCEUR

background: rgba(0, 0, 0, 1)  

color: rgba(3, 177, 252, 1)

colorShade: rgba(3, 177, 252, 0.15)

epoch: 1D # possible epochs: 1D, 1M, 3M, 6M, 1Y, 5Y, ALL
```
## features
✔️ Responsive <br>
✔️ No Dependencies <br>
✔️ Frame Embedment <br>
✔️ Customizable <br>
✔️ Quick and Useful for day trading <br>
✔️ tested in Chrome & Firefox <br>

## Usage
```bash 
~$ python chartPeer.py
```


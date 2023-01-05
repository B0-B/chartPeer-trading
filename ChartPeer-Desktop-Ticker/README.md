# ChartPeer-Desktop-Ticker


Specify the instrument symbol in the `config.json` and set all other parameters to suit the purpose.

```json
{   
    "data": {
        "asset_type": "crypto",         
        "symbol": "BTC",
        "interval": "5min",
        "currency": "USD",
        "unit_symbol": "Bitcoin",
        "refresh_in_min": 0.5,
        "aggregation_length": 1440,
        "alpha_vantage_api_key": "API_KEY_HERE",
        "json_api": true
    },
    "interface": {
        "color_background": "#000",
        "color_bull": "#2cc767",
        "color_bear": "#c72c55",
        "color_neutral": "#fff",
        "dark_frame": true,
        "resolution_dpi": 72,
        "transparency": 0.2
    }
    
}
```
The ticker aggregates data of max spec. lenght for analysis purposes.

<img src='demo.png'>

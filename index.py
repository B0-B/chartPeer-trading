#!/usr/bin/env python3

#======= Configurations =======#
config = {
    "assets": [
        "KRAKEN:XBTEUR",
        "KRAKEN:TRXEUR",
        "KRAKEN:DOTEUR",
        "KRAKEN:XRPEUR",
        "KRAKEN:AAVEEUR",
        "KRAKEN:LSKEUR",
        "KRAKEN:ICXEUR",
        "KRAKEN:MLNEUR",
        "KRAKEN:XLMEUR",
        "KRAKEN:SCEUR",
        "KRAKEN:NANOEUR",
        "KRAKEN:ADAEUR",
        "KRAKEN:MLNEUR",
        "KRAKEN:XMREUR",
        "KRAKEN:GRTEUR",
        "KRAKEN:KNCEUR",
    ],
    "color": "#03b1fc",
    "colorShade": "rgba(3, 177, 252, 0.15)",
    "epoch": "1D", # 1D, 1M, 3M, 6M, 1Y, 5Y, ALL
    "background": "#0d0d0d"
}
#==============================#

def widgetInject(symbol, epoch, color, underLineColor, base):
    
    return f'''\
        <div style="position:relative;display:inline-block;height:{int(100/base+10)}%;width:{int(100/base-0.1)}%">
            <div class="tradingview-widget-container">
            <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-mini-symbol-overview.js" async>{"{"} 
            "symbol": "{symbol}",
            "width": "100%",
            "height": "100%",
            "locale": "de_DE",
            "dateRange": "{epoch}",
            "colorTheme": "dark",
            "trendLineColor": "{color}",
            "underLineColor": "{underLineColor}",
            "isTransparent": true,
            "autosize": true,
            "largeChartUrl": ""
            {"}"} 
            </script>
            </div>
        </div>
    '''

def buildPage(params): # widgets must be 

    widgets = len(params["assets"])
    base = int(widgets**(0.5))
    
    # build widgets and add them to dummy
    injection = ""
    for i in range(widgets):
        injection += widgetInject(params["assets"][i], params["epoch"], params["color"], params["colorShade"], base)

    # build injectable dummy
    htmlDummy = f'''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <link rel="shortcut icon" type="image/x-icon" href="logo.svg">
        <title>dashview</title>
    </head>
    <body style="height: 100%; width: 100%; background: {params["background"]}; margin: 0">
        {injection}
    <body>
    '''

    return htmlDummy

if __name__ == "__main__":
    import os, time, webbrowser as browser
    html = buildPage(config)
    path = os.path.abspath('temp.html')
    url = 'file://' + path
    try:
        with open(path, 'w') as f:
            f.write(html)
        browser.open(url)
        print("dashview: browser started ...")
        time.sleep(5)
    except:
        pass
    finally: 
        os.remove(path)
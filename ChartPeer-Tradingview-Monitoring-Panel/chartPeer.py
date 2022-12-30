#!/usr/bin/env python3

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

def favicon(path):
    # favicon
    fav = '''\
    <svg id="Capa_1" enable-background="new 0 0 512.279 512.279" height="512" viewBox="0 0 512.279 512.279" width="512" xmlns="http://www.w3.org/2000/svg">
    <g><path d="m342.009 472.028-14.573-16.978h-141.801l-15.366 16.978-28.24 32.75h228.22z" fill="#efedef"/><path d="m261.435 455.05h-75.8l-15.366 16.978-28.24 32.75h75.8l28.24-32.75z" fill="#e5e1e5"/>
    <path d="m170.269 360.018h171.74v112.01h-171.74z" fill="#c9bfc8"/><path d="m342.009 360.018h-96.145-75.595v112.01h75.595v-3.763c0-10.856 8.801-19.657 19.657-19.657h76.488z" fill="#baafb9"/>
    <path d="m484.748 403.513 19.111-20.095v-264.19c0-22.19-17.99-40.18-40.18-40.18h-415.08c-22.19 0-40.18 17.99-40.18 40.18v264.19l34.877 20.095z" fill="#99e6fc"/><g><g id="XMLID_1052_">
    <path d="m175.073 383.418v-264.19c0-22.19 17.99-40.18 40.18-40.18h-166.654c-22.19 0-40.18 17.99-40.18 40.18v264.19l34.877 20.095h137.164c-3.42-5.913-5.387-12.772-5.387-20.095z" fill="#62dbfb"/>
    <g><path d="m8.419 383.418h495.44c0 22.2-17.99 40.19-40.18 40.19h-415.08c-22.19 0-40.18-17.99-40.18-40.19z" fill="#efedef"/></g>
    <path d="m175.073 383.418h-166.654c0 22.2 17.99 40.19 40.18 40.19h166.651c-22.189-.002-40.177-17.991-40.177-40.19z" fill="#e5e1e5"/>
    </g><g><path d="m99.122 306.265c-4.142 0-7.5-3.358-7.5-7.5v-168.74c0-4.142 3.358-7.5 7.5-7.5s7.5 3.358 7.5 7.5v168.74c0 4.142-3.358 7.5-7.5 7.5z" fill="#9cf8d2"/>
    </g><g><path d="m187.573 139.18c-4.142 0-7.5-3.358-7.5-7.5v-63.842c0-4.142 3.358-7.5 7.5-7.5s7.5 3.358 7.5 7.5v63.841c0 4.143-3.358 7.501-7.5 7.501z" fill="#f07281"/>
    </g><g><path d="m187.573 291.755c-4.142 0-7.5-3.358-7.5-7.5v-94.773c0-4.142 3.358-7.5 7.5-7.5s7.5 3.358 7.5 7.5v94.773c0 4.142-3.358 7.5-7.5 7.5z" fill="#f07281"/></g><g>
    <path d="m276.023 91.849c-4.143 0-7.5-3.358-7.5-7.5v-76.849c0-4.142 3.357-7.5 7.5-7.5s7.5 3.358 7.5 7.5v76.849c0 4.142-3.357 7.5-7.5 7.5z" fill="#9cf8d2"/>
    </g><g><path d="m276.023 250.582c-4.143 0-7.5-3.358-7.5-7.5v-109.228c0-4.142 3.357-7.5 7.5-7.5s7.5 3.358 7.5 7.5v109.228c0 4.142-3.357 7.5-7.5 7.5z" fill="#9cf8d2"/>
    </g><g><path d="m364.473 171.646c-4.143 0-7.5-3.358-7.5-7.5v-122.705c0-4.142 3.357-7.5 7.5-7.5s7.5 3.358 7.5 7.5v122.704c0 4.143-3.357 7.501-7.5 7.501z" fill="#f07281"/>
    </g><g><path d="m364.473 302.105c-4.143 0-7.5-3.358-7.5-7.5v-101.914c0-4.142 3.357-7.5 7.5-7.5s7.5 3.358 7.5 7.5v101.914c0 4.142-3.357 7.5-7.5 7.5z" fill="#f07281"/>
    </g><g fill="#e53950"><path d="m324.013 242.247-152.364 82.15-8.531-22.353c-.728-1.907-2.201-3.435-4.08-4.231-1.879-.797-4.001-.792-5.878.01l-103.953 44.49c-3.808 1.63-5.574 6.038-3.944 9.846 1.218 2.846 3.987 4.551 6.898 4.551.984 0 1.985-.195 2.948-.607l96.778-41.419 8.795 23.043c.77 2.017 2.371 3.604 4.394 4.356 2.022.751 4.272.596 6.172-.429l159.884-86.205c3.646-1.966 5.008-6.515 3.042-10.161-1.964-3.643-6.512-5.007-10.161-3.041z"/>
    <path d="m468.101 170.05-23.527-8.963c-3.87-1.474-8.204.468-9.679 4.339s.468 8.204 4.339 9.679l5.455 2.078-33.892 18.274c-3.646 1.966-5.008 6.515-3.042 10.161 1.355 2.514 3.939 3.942 6.608 3.942 1.201 0 2.421-.29 3.553-.9l34.547-18.627-2.485 6.522c-1.475 3.871.468 8.204 4.339 9.679.879.334 1.781.493 2.669.493 3.021 0 5.87-1.84 7.01-4.832l8.444-22.166c1.475-3.871-.468-8.205-4.339-9.679z"/></g></g><g>
    <path d="m382.586 512.279h-253.584c-4.142 0-7.5-3.358-7.5-7.5s3.358-7.5 7.5-7.5h253.583c4.143 0 7.5 3.358 7.5 7.5s-3.357 7.5-7.499 7.5z" fill="#c9bfc8"/>
    </g><path d="m377.602 227.604h-26.258c-3.73 0-6.754-3.024-6.754-6.754v-88.294c0-3.73 3.024-6.754 6.754-6.754h26.258c3.73 0 6.754 3.024 6.754 6.754v88.294c0 3.73-3.024 6.754-6.754 6.754z" fill="#eb5569"/>
    <path d="m289.152 195.641h-26.258c-3.73 0-6.754-3.024-6.754-6.754v-137.294c0-3.73 3.024-6.754 6.754-6.754h26.258c3.73 0 6.754 3.024 6.754 6.754v137.294c0 3.73-3.024 6.754-6.754 6.754z" fill="#6cf5c2"/>
    <path d="m200.701 210.482h-26.258c-3.73 0-6.754-3.024-6.754-6.754v-88.294c0-3.73 3.024-6.754 6.754-6.754h26.258c3.73 0 6.754 3.024 6.754 6.754v88.294c.001 3.73-3.023 6.754-6.754 6.754z" fill="#eb5569"/>
    <path d="m112.251 251.324h-26.258c-3.73 0-6.754-3.024-6.754-6.754v-88.294c0-3.73 3.024-6.754 6.754-6.754h26.258c3.73 0 6.754 3.024 6.754 6.754v88.294c.001 3.73-3.023 6.754-6.754 6.754z" fill="#6cf5c2"/>
    </g></svg>
    '''

    with open("logo.svg", "w") as l:
        l.write(fav)

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
        <title>chartPeer [Monitor]</title>
    </head>
    <body style="height: 100%; width: 100%; background: {params["background"]}; margin: 0">
        {injection}
    <body>
    '''

    return htmlDummy

if __name__ == "__main__":
    import os, time, webbrowser as browser
    with open('config.yml') as f:
        text = f.read()
        lines = text.split('\n')
        config = {}
        keys = []
        values = []
        for l in range(len(lines)):
            line = lines[l]
            if '#' in line:
                pass
            else:
                if ':' in line and '-' not in line:
                    key = line.split(':')[0]
                    if line.split(':')[1] == '':
                        arr = []
                        while (True):
                            l += 1
                            line = lines[l]
                            if '-' in line:
                                arr.append(line.replace('-','').replace(' ', ''))
                            elif '#' in line:
                                pass
                            else:
                                l -= 1
                                break
                        value = arr
                    else:
                        value = line.split(':')[1].replace('-','').replace(' ', '')
                    config[key] = value
    path1 = os.path.abspath('logo.svg')
    path2 = os.path.abspath('temp.html')
    favicon(path1)
    html = buildPage(config)
    url = 'file://' + path2
    try:
        with open(path2, 'w') as f:
            f.write(html)
        browser.open(url)
        print("chartPeer: launching browser ...")
        time.sleep(5)
    except:
        pass
    finally:
        if not eval(config['keepHTMLinLocal']):
            for path in [path1, path2]:
                os.remove(path)
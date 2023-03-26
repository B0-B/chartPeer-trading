#!/usr/bin/env python3

import json
import csv
import numpy as np
from pathlib import Path
from traceback import print_exc
import requests
from datetime import datetime
# from urllib.error import HTTPError
# from urllib.request import Request
# import urllib.request as urllib2
# import urllib.parse as urllib
# import urllib


class load:

    '''
    Loads price data from data files into desired formats.
    '''

    root = Path(__file__).resolve().parent
    dataPath = root.parent/"data"

    krakenSymbol = {
        'BTC': 'XXBTZ',
    }

    def ohlcFromFile (filename, delimiter=','):

        '''
        Loads CSV file and splits all rows to list objects 
        and turns elements into proper number type.
        Returns list of [timestamp, open, high, low, close, avg, volume] tuples.
        '''
        data = []

        with open(load.dataPath/filename, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=delimiter, quotechar='|')
            for row in spamreader:
                dataPoint = [
                    int(row[0]),
                    float(row[1]),
                    float(row[2]),
                    float(row[3]),
                    float(row[4]),
                    float(row[5]),
                    int(row[6]),
                ]
                data.append(dataPoint)
        
        return data
    
    def closedFromFile (filename, delimiter=','):

        '''
        Returns 1D numpy array of closed prices from file.
        '''

        out = []
        for p in load.ohlcFromFile(filename, delimiter):
            out.append(p[4])
        return np.array(out)

    def closedFromOhlc (ohlc):

        '''
        Returns 1D numpy array of closed prices from previously determined OHLC data.
        If the OHLC is known this method is more efficient than load.closedFromFile.
        '''

        out = []
        for p in ohlc:
            out.append(p[4])
        return np.array(out)

class krakenApi:

    '''
    Loads crypto price data from kraken rest API.
    Returns list of candle lists (time,o,h,l,c,avg,volume).
    '''

    krakenUrl = 'https://futures.kraken.com'

    krakenSymbol = {
        'BTC': 'XXBTZ',
    }


    def ohlc(symbol, interval, ref='USD', timeFormat='%Y-%m-%d %H:%M'):

        '''
        Requests OHLC timeseries data 720 points of chosen time intervals in minutes.
        '''

        if symbol.upper() in krakenApi.krakenSymbol:
            symbol = krakenApi.krakenSymbol[symbol.upper()]
        else:
            symbol = symbol.upper()

        pair = f'{symbol}{ref}'
        response = requests.get(f'https://api.kraken.com/0/public/OHLC?pair={pair}&interval={interval}', headers={'Accept': 'application/json'})
        pkg = response.json()

        if len(pkg['error']) > 0:
            raise ValueError(pkg['error'][0])

        # unpack
        ohlcData = list(pkg['result'].values())[0]
        for i in range(len(ohlcData)):
            ohlcData[i][0] = datetime.fromtimestamp(ohlcData[i][0]).strftime(timeFormat)
            for j in range(1,7):
                ohlcData[i][j] = float(ohlcData[i][j])
        
        return ohlcData

    def price(symbol, ref='USD'):

        '''
        Requests latest price for symbol.
        '''

        if symbol.upper() in krakenApi.krakenSymbol:
            symbol = krakenApi.krakenSymbol[symbol.upper()]
        else:
            symbol = symbol.upper()

        pair = f'{symbol}{ref}'
        response = requests.get(f'https://api.kraken.com/0/public/Ticker?pair={pair}', headers={'Accept': 'application/json'})
        pkg = response.json()

        if len(pkg['error']) > 0:
            raise ValueError(pkg['error'][0])

        # unpack
        price = float(list(pkg['result'].values())[0]['c'][0])
        
        return price


if __name__ == '__main__':
    # print(load.loadOhlcFromFile('XBTUSD_1440.csv'))
    api = krakenApi.price('xbt')
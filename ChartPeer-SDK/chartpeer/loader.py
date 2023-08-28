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

    def ohlcFromFile (filename:str, delimiter:str=',') -> list:

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
    
    def closedFromFile (filename:str, delimiter:str=',') -> list:

        '''
        Returns 1D numpy array of closed prices from file.
        '''

        out = []
        for p in load.ohlcFromFile(filename, delimiter):
            out.append(p[4])
        return np.array(out)

    def closedFromOhlc (ohlc:list) -> list:

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


    def ohlc (symbol:str, interval:int, ref:str='USD', timeFormat:str='%Y-%m-%d %H:%M') -> list:

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

    def price(symbol:str, ref:str='USD') -> float:

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

class alphaVantage:

    '''
    General stock data API. 
    '''

    url = 'https://www.alphavantage.co'

    def __init__ (self, api_key:str|None=None) -> None:

        if not api_key:
            raise ValueError('Please claim an alphaVantage API key here https://www.alphavantage.co/support/#api-key')
        self.avKey = 'V1UUOJF8DY7G75OO'
        self.data = []
    
    def findSymbol (self, suggestion:str) -> list:

        '''
        Returns a list of stock/ticker symbols from different markets.
        '''

        endpoint = f'https://www.alphavantage.co/query?function=SYMBOL_SEARCH&keywords={suggestion}&apikey={self.avKey}'
        response = requests.get( endpoint )
        return response.json()['bestMatches']

    def getFundamentals (self, symbol:str) -> dict:

        '''
        Returns the company information, financial ratios, and other key metrics for the equity specified. 
        Data is generally refreshed on the same day a company reports its latest earnings and financials. 
        
        Format:

        {
            'Symbol': 'AMD', 
            'AssetType': 'Common Stock', 
            'Name': 'Advanced Micro Devices Inc', 
            'Description': "Advanced Micro Devices, Inc. (AMD) is an American multinational semiconductor company based in Santa Clara, California, that develops computer processors and related technologies for business and consumer markets. AMD's main products include microprocessors, motherboard chipsets, embedded processors and graphics processors for servers, workstations, personal computers and embedded system applications.", 
            'CIK': '2488', 
            'Exchange': 'NASDAQ', 
            'Currency': 'USD', 
            'Country': 'USA', 
            'Sector': 'MANUFACTURING', 
            'Industry': 'SEMICONDUCTORS & RELATED DEVICES', 
            'Address': '2485 AUGUSTINE DRIVE, SANTA CLARA, CA, US', 
            'FiscalYearEnd': 'December', 
            'LatestQuarter': '2023-06-30', 
            'MarketCapitalization': '165202264000', 
            'EBITDA': '3100000000', 
            'PERatio': 'None', 
            'PEGRatio': '0.902', 
            'BookValue': '34.16', 
            'DividendPerShare': '0', 
            'DividendYield': '0', 
            'EPS': '-0.04', 
            'RevenuePerShareTTM': '13.55', 
            'ProfitMargin': '-0.0011', 
            'OperatingMarginTTM': '-0.0173', 
            'ReturnOnAssetsTTM': '-0.0035', 
            'ReturnOnEquityTTM': '-0.0005', 
            'RevenueTTM': '21876001000', 
            'GrossProfitTTM': '12051000000', 
            'DilutedEPSTTM': '-0.04', 
            'QuarterlyEarningsGrowthYOY': '-0.938', 
            'QuarterlyRevenueGrowthYOY': '-0.182', 
            'AnalystTargetPrice': '127.48', 
            'TrailingPE': '-', 
            'ForwardPE': '18.02', 
            'PriceToSalesRatioTTM': '4.699', 
            'PriceToBookRatio': '2.038', 
            'EVToRevenue': '4.75', 
            'EVToEBITDA': '18.86', 
            'Beta': '1.822', 
            '52WeekHigh': '132.83', 
            '52WeekLow': '54.57', 
            '50DayMovingAverage': '112.1', 
            '200DayMovingAverage': '92.19', 
            'SharesOutstanding': '1610360000', 
            'DividendDate': 'None', 
            'ExDividendDate': '1995-04-27'
        }
        '''

        uri = f'/query?function=OVERVIEW&symbol={symbol}&apikey={self.avKey}'
        endpoint = self.url + uri

        # make request and unpack data
        response = requests.get( endpoint )
        data = response.json()

        return data

    def getStockData (self, symbol:str, interval:int=5) -> list:

        '''
        Returns the last 30 days of stock data (time,o,h,l,c,v)
        from alphaVantage API. Data is saved in global .data object.
        '''

        uri = f'/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval={interval}min&outputsize=full&apikey={self.avKey}'
        endpoint = self.url + uri

        # make request and unpack data
        response = requests.get( endpoint )
        data = response.json()
        timeseries = data[f'Time Series ({interval}min)']
        
        dataset = []

        for k, v in timeseries.items():
            v = list(v.values())
            dataset.append([k]+[float(v[i]) for i in range(5)])
        dataset.reverse()

        # override global dataset
        self.data = dataset

        return dataset



if __name__ == '__main__':
    
    # print(load.loadOhlcFromFile('XBTUSD_1440.csv'))
    #api = krakenApi.price('xbt')
    pass
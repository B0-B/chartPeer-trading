#!/usr/bin/env python3
import numpy as np
from pathlib import Path
import requests
import csv

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
    
    def ohlcFromCryptoApi (symbol, refCurrency, interval=1440):

        '''
        Loads 720 recent ohlc candles from kraken API.
        '''




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

if __name__ == '__main__':
    print(load.loadOhlcFromFile('XBTUSD_1440.csv'))
#!/usr/bin/env python3

import tkinter
from os import path
import json
from requests import get
from time import sleep
from threading import Thread
import tkinter as tk
import ctypes as ct
from traceback import print_exc
import platform

class backend ():

    __PATH__ = path.dirname(path.abspath(__file__))
    __active__ = True

    # exported global variables
    last = 0
    closed = []
    threads = []
    change_last = 0

    def __init__ (self) -> None:
        
        self.parse_config()
    
    def ci (self):

        '''
        Repeating config parser for change incidents. 
        '''

        while (self.__active__):
            try:
                self.parse_config(ignore_data=True)
            except Exception as e:
                print('Error in CI:', e)
            finally:
                sleep(1)

    def parse_config (self, ignore_data=False):

        '''
        Parse config file once.
        '''

        with open(self.__PATH__ + '/config.json') as f:
            conf_pkg = json.load(f)
            if not ignore_data:
                self.config = conf_pkg["data"]
            self.config_interface = conf_pkg["interface"]

    def request_last_price (self):

        '''
        Requests last price from alpha vantage endpoint.

        Returns current price as float.
        If errors occur during the request, None object is returned.
        '''

        # construct the vantage endpoint address
        key = self.config["alpha_vantage_api_key"]
        symbol = self.config["symbol"]
        currency = self.config["currency"]

        if 'crypto' in self.config["asset_type"].lower():
            func = 'CURRENCY_EXCHANGE_RATE'
            param_insert = f'&from_currency={symbol.upper()}&to_currency={currency.upper()}'
        elif 'stock' in self.config["asset_type"].lower():
            func = 'TIME_SERIES_INTRADAY'
            param_insert = f'&symbol={symbol}&market={currency}&interval=1min'

        url = 'https://www.alphavantage.co'
        uri = f'/query?function={func}{param_insert}&apikey={key}'
        
        end_point = url + uri

        # make a request
        try:
            resp = get(end_point)
        except Exception as e:
            print('Request error:', e)
            return None

        # check how to unpack response
        if self.config["json_api"]:
            raw_data = resp.json()
        else:
            raw_data = resp

        # find the recent price observation
        last_price = 0

        # construct ohlc from lastPrice aggregation (no OHLCV response possible)
        if 'crypto' in self.config["asset_type"].lower():

            last_price = float(list(list(raw_data.values())[0].values())[4])

        # extract ohlcv from timeseries for stocks
        elif 'stock' in self.config["asset_type"].lower():
        
            for key in raw_data.keys():
                if 'series' in key.lower():
                    timeseries = raw_data[key]
                    break
        
            # from ts extract last price
            recent_candle = list(timeseries.values())[0]
            last_price = list(recent_candle.values())[3]
            
        return float(last_price)
    
    def main_thread (self):

        '''
        Main thread which requests time repeatedly and stores it.
        '''

        refresh_in_seconds = self.config["refresh_in_min"]*60

        while (self.__active__):
        
            try:

                self.last = self.request_last_price()
                self.closed.append(self.last)
                if len(self.closed) > self.config['aggregation_length']:
                    self.closed = self.closed[-self.config['aggregation_length']:]

                # determine last change
                if len(self.closed) > 2:
                    self.change_last = round((self.last / self.closed[-2] - 1) * 100, 2)

            except Exception as e:

                print('Backend Error:', e)

            finally:

                sleep(refresh_in_seconds)
    
    def run (self):

        '''
        Initializes asynchronous sub processes.
        '''

        self.threads.append(Thread(target=self.main_thread))
        self.threads.append(Thread(target=self.ci))

        for t in self.threads:
            t.start()
    
    def quit (self):

        self.__active__ = False


class gui (tk.Tk):

    '''
    Tracker GUI Wrapper
    '''

    _backend = backend()

    def __init__(self) -> None:

        tk.Tk.__init__(self)

        # build UI
        self.build_interface()

        # instantiate backend
        self._backend.run()

        # start UI thread
        ui_thread = Thread(target=self.ui_loop)
        ui_thread.start()

        # run main loop
        print('test 2')
        self.mainloop()

        

        self._backend.quit()

    def build_interface (self):

        bg_col = self._backend.config_interface["color_background"]
        fg_col = self._backend.config_interface["color_neutral"]
        font = "Arial"

        self.iconbitmap(self._backend.__PATH__ + '/fav.ico')

        self.symbol_var = tk.StringVar(self)
        self.price_last_var = tk.StringVar(self)
        self.price_change_last_var = tk.StringVar(self)

        # scale gui to resolution
        self.tk.call('tk', 'scaling', int(self._backend.config_interface["resolution_dpi"]/72))

        # check for dark theme
        if self._backend.config_interface["dark_frame"] and 'win' in platform.platform().lower():
            self.wm_attributes('-toolwindow', 'True')
            DWMWA_USE_IMMERSIVE_DARK_MODE = 20
            set_window_attribute = ct.windll.dwmapi.DwmSetWindowAttribute
            get_parent = ct.windll.user32.GetParent
            hwnd = get_parent(self.winfo_id())
            rendering_policy = DWMWA_USE_IMMERSIVE_DARK_MODE
            value = 2
            value = ct.c_int(value)
            set_window_attribute(hwnd, rendering_policy, ct.byref(value),ct.sizeof(value))
        
        # set title 
        self.title('ChartPeer Desktop Ticker')

        # set opacity
        self.attributes('-alpha', 1.-self._backend.config_interface["transparency"])

        # build layout
        height, width = 200, 500
        self.geometry(f'{width}x{height}')

        header = tk.Frame(self, bg=bg_col, pady=0, width=width, height=int(height*.3))
        main = tk.Frame(self, bg=bg_col, width=width, height=int(height*.4))
        second = tk.Frame(self, bg=bg_col, width=width, height=int(height*.2))
        footer = tk.Frame(self, bg=bg_col, width=width, height=int(height*.1))
        for el in [header, main, second, footer]:
            el.pack_propagate(0)
            el.pack(side='top')
        

        # format
        self.configure(background=bg_col)

        # add symbol unit name to header
        self.symbol_label = tk.Label(header, 
            bg=bg_col, 
            fg=fg_col,
            width=width, 
            font=[font, 18],
            height=int(height*.3),
            textvariable=self.symbol_var)
        self.symbol_label.pack(side='top')
        self.symbol_var.set(self._backend.config["unit_symbol"])


        # add price tag to main 
        self.price_label = tk.Label(main,
            bg=bg_col, 
            fg=fg_col,
            width=width, 
            font=[font, 30],
            height=int(height*.6),
            textvariable=self.price_last_var)
        self.price_label.pack(side='top')
        self.price_last_var.set(self._backend.config["unit_symbol"])


        # add price change to second frame
        self.price_change_last_label = tk.Label(main,
            bg=bg_col, 
            fg=fg_col,
            width=width, 
            font=[font, 20],
            height=int(height*.6),
            textvariable=self.price_change_last_var)
        self.price_change_last_label.pack(side='top')
    
    def ui_loop (self):
        
        '''
        Refresh all parameters.
        '''

        while self._backend.__active__:

            try:

                # update price label
                compare = None
                if len(self._backend.closed) > 1:
                    compare = self._backend.closed[-2:]
                    compare.reverse()
                    self.colorFormat(*compare, self.price_label)
                self.price_last_var.set(f'{self._backend.last} {self._backend.config["currency"]}')
            
                # update price change label
                if compare and compare[0]-compare[1] != 0:
                    change =  str(self._backend.change_last) + '%'
                    if self._backend.change_last < 0:
                        change = '-' + change
                    elif self._backend.change_last > 0:
                        change = '+' + change
                    self.colorFormat(self._backend.change_last, 0, self.price_change_last_label)
                    self.price_change_last_var.set('last change: ' + change)
            
            except:
                print_exc()
            finally:
                sleep(1)

    def colorFormat (self, value, reference, tkObject):

        '''
        Determines the correct bearish or bullish color by comparing two values and configuring the tk parent.
        '''
        print('prices', value, reference)
        if value > reference:
            col = self._backend.config_interface["color_bull"]
        elif value < reference:
            col = self._backend.config_interface["color_bear"]
        else:
            col = self._backend.config_interface["color_neutral"]         
        tkObject.config(fg=col) 


if __name__ == '__main__':
    gui()
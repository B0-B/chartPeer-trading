#!/usr/bin/env python3

'''
Conventions:

    Name            Object              Type
    -------------------------------------------------------
    N               scalar              int
    dataset         1D data vector      array

'''

import numpy as np
import matplotlib.pyplot as plt


class profit:

    def pnl (entryPrice, exitPrice, size, side):

        '''
        Profit and Loss computation for futures, derivatives or options trading.
        
        Arguments:
        entryPrice/
        exitPrice       prices in collateral currency
        side            "buy"/"sell"
        size            total contract size (leveraged) in collateral 
                        currency (usually dollars)
        
        Returns the absolute profit in collateral currency. 
        '''

        if side == 'buy':
            PnL = (exitPrice-entryPrice) * size
        elif side == 'sell':
            PnL = (1/exitPrice-1/entryPrice) * size
        else:
            raise ValueError(f'Provided side must be "buy" or "sell", not {side}')
        return PnL

    def roi (entryPrice, exitPrice):

        '''
        Return on Invest.
        Returns percentage ROI by the formula
        (gain from investment − cost of investment) / cost of investment
        https://en.wikipedia.org/wiki/Return_on_investment
        '''

        return (exitPrice-entryPrice)/entryPrice


class regression:

    def chi2(arg, dataSet, model):
        '''
        Sum over all square elements is distributed according to 
        the chi-squared distribution with k degrees of freedom.

        Returns a scalar (float).
        '''
        out = []
        for i in range(len(dataSet)):
            out.append( (dataSet[i]-model(arg, i))**2 )
        return sum(out)

    def linear (dataset, extrapolate=0):
        '''
        Improved linear regression fit.
        1. The linear fit should intersect the dataSet's mean which lies at the x range median -> anchor point
        2. Rotate the linear model around the anchor to minimize the chi squared. This saves iterating over a whole parameter.
        '''
        # define ranges
        if type(dataset) is np.ndarray:
            l = dataset.shape[0]
        else:
            l = len(dataset)
        m = np.mean(dataset) # from montecarlo
        Raise = ( dataset[-1] - dataset[0] ) / l # crude but fast estimate
        a_range = ( Raise*0.8, Raise*1.2) # +/-20% window
        a_step = ( a_range[1] - a_range[0] ) / 1000

        # anchor and rotation notation of a linear function
        def model(alpha, x):
            return alpha * (x-l/2) + m

        alpha = a_range[0]
        alpha_best = alpha
        c = statistics.meanSquaredDistance(alpha, dataset, model)
        while alpha < a_range[1]:
            alpha += a_step
            c_new = tools.chiSquared(alpha, dataset, model)
            if  c_new < c:
                c = c_new
                alpha_best = alpha
            
        
        return [model(alpha_best, x) for x in range(l+extrapolate)]


class statistics:

    def drift (dataset):

        '''
        Percentage Drift Implementation.
        Computes the expectation value of the drift from logarithmic
        returns array.

        '''

        return statistics.mean(statistics.logReturns(dataset))

    def mean (dataset, *args, **kwargs):
        '''
        Numpy mean alias.
        '''
        return np.mean(dataset, *args, **kwargs)

    def meanSquaredDistance (dataset1, dataset2):
        '''
        Sum over all square elements is distributed according to 
        the chi-squared distribution with k degrees of freedom.

        Returns a scalar (float).
        '''
        if dataset1.shape[0] != dataset2.shape[1]:
            raise ValueError(f'Provided datasets need to have different size, but {dataset1.shape[0]} and {dataset2.shape[1]} were provided!')
        out = []
        for i in range(len(dataset1)):
            out.append( (dataset1[i]-dataset2[i])**2 )
        return sum(out)

    def logReturns (dataset):

        '''
        Computes the logarithmic returns.
        Returns a 1D array
        numpyp.ndarray([ log(d[1]/d[0], log(d[2]/d[1]), ... ])
        '''

        out = []
        for i in range(len(dataset)-1):
            out.append(np.log(dataset[i+1]/dataset[i]))
        return np.array(out)

    def sampleVariance (dataset, correction=1):
        '''
        Sample variance.
        The correction factor refers to Bessel's correction 1/(n-1) (default).
        https://en.wikipedia.org/wiki/Variance#Sample_variance
        '''
        return np.std(dataset, ddof=correction)

    def standardDeviation (dataset):
        '''
        Return the standard deviation obtained from sample variance.
        '''

        return np.std(dataset)
    
    def volatility(dataset):

        '''
        Computes the expected/meaned percentage volatility per element.
        '''

        return statistics.standardDeviation(statistics.logReturns(dataset))


class indicators:

    def ema (dataset, window):

        '''
        Exponential Moving Average.
        Returns 1D array.
        '''

        a, out = 2/(window+1), [dataset[1]]
        for d in dataset[1:]: out.append(a*d+(1-a)*out[-1])
        return np.out
    
    def klinger(self, dataset, fastPeriod=34, slowPeriod=55, signalPeriod=13):
        
        '''
        Klinger Volume Oscillator Implementation.
        https://en.wikipedia.org/wiki/Volume_analysis

        Return
        [
            scalar value of recent difference (signal - slow),
            signal oscillator array,
            slow oscillator array
        ]
        '''
        
        osc, lastTrend, lastCm, lastDm, weightFast, weightSlow = [], 0, 0, 0, 2/(fastPeriod+1), 2/(slowPeriod+1)
        maFast, maSlow = dataset[1]['y'][-1], dataset[1]['y'][-1] # volumes
        for i in range(2,dataset.shape[0]):
            c = dataset[i]['y'] # ohlc candle
            v = dataset[i]['v']
            #print('c',c)
            dm = c[1] - c[2]
            currentTrend = c[1] + c[2] + c[3] - dataset[i-1]['y'][1] - dataset[i-1]['y'][2] - dataset[i-1]['y'][3]
            if currentTrend <= 0:
                TREND = -1
            else:
                TREND = 1
            if lastTrend == currentTrend:
                CM = lastCm + dm
            else:
                CM = lastDm + dm
                lastTrend = currentTrend
            lastCm = CM
            lastDm = dm
            if CM == 0:
                temp = -2
            else:
                temp = abs(2*(dm/CM-1))
            VF = v*temp*TREND*100
            maSlow = VF*weightSlow + maSlow*(1-weightSlow)
            maFast = VF*weightFast + maFast*(1-weightFast)
            osc.append(maFast-maSlow)
        oscSlow = indicators.ema(osc, signalPeriod)
        
        return [osc[-1]-oscSlow[-1], osc, oscSlow]

    def macd (dataset, signalPeriod=12, slowPeriod=26, macdPeriod=9):
        '''
        MACD implementation with typical period parameters.
        The most commonly used periods (default) are 12, 26, 9, respectively.
        Returns 1D array of difference between signal and slow oscillator.
        '''
        signal, slow = np.array(indicators.ema(dataset, signalPeriod)), np.array(indicator.ema(closed, slowPeriod))
        macd_sig = signal - slow
        macd_slow = np.array(indicators.ema(macd_sig, macdPeriod))
        return macd_sig - macd_slow

    def rsi (dataset, window=14):
        '''
        Relative Strength Index Implementation.
        Returns time-resolved 1D array with RSI values between 0 and 100.
        '''
        u,d,rsi=[],[],[]
        for i in range(1,len(dataset)):
            c = dataset[i] - dataset[i-1]
            if c > 0:
                u.append(c)
                d.append(0)
            elif c < 0:
                u.append(0)
                d.append(-c)
            else:
                u.append(0)
                d.append(0)
        u, d = indicators.ema(u,window), indicators.ema(d,window)
        for i in range(len(d)):
            if d[i]==0: # avoid zero division
                rsi.append(100)
            else:
                rsi.append(100*(1-1/(1+u[i]/d[i])))
        return np.array(rsi)

    def sma (dataset, window):

        '''
        Simple moving average implementation.
        '''
        
        out = []
        for i in range(window, dataset.shape[0]):
            out.append(np.mean(dataset[i-window:i]))
        return out


class plot:

    def chart (dataset, name='dataset', overlays={}, color='b', indicatorSets={}, predictionSets={}, 
               timeset=None, savePath=None, title='Chart', renderLegend=True):

        # position is splitted at the end of dataset and where the prediction begins
        L = dataset.shape[0]

        # determine the max prediction length
        maxPredictionLength = 0
        for pred in predictionSets.values():
            if type(pred) is np.ndarray and pred.shape[0] > maxPredictionLength:
                maxPredictionLength = pred.shape[0]
            elif len(pred) > maxPredictionLength:
                maxPredictionLength = len(pred)
        
        # define all arrays
        if timeset:
            if timeset.shape[0] != L:
                raise ValueError(f'dataset and timeset have to match in length, provided {L} and {timeset.shape[0]} do not match!')
            x_array = timeset
            x_extra = np.array([str(i+1) for i in range(maxPredictionLength)])
        else:
            x_array = np.arange(stop=L)
            x_extra = np.arange(start=L, stop=L+maxPredictionLength)
        
        x_array = np.concatenate((x_array, x_extra))
        y_array = dataset
        

        # create pyplot figure and format it
        fig = plt.figure(dpi=150, facecolor='black', edgecolor='white')
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title(title, c='white')
        ax.set_facecolor('black')
        # ax.set_xticks([])
        # ax.set_xlabel('')
        ax.yaxis.tick_right()
        ax.spines['right'].set_color('white')
        ax.xaxis.label.set_color('white')
        ax.tick_params(axis='y', colors='white')


        # add main dataset to chart
        ax.plot(x_array[:L], y_array, label=name, c=color)

        # add overlay datasets
        for name, overlay in overlays.items():
            if type(overlay) is list:
                L = len(overlay)
            else:
                L = overlay.shape[0]
            ax.plot(x_array[:L], overlay, label=name)

        # add all predictions accordingly
        for name, pred in predictionSets.items():
            ax.plot(x_array[L:L+pred.shape[0]], pred, label=name)
        

        # finally add the ledgend
        if renderLegend:
            plt.legend(facecolor='#383838', edgecolor='#ddd', labelcolor='linecolor')

        # save if enabled
        if (savePath):
            fig.savefig(savePath)
        else:
            plt.show()
        
        plt.close(fig)
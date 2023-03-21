#!/usr/bin/env python3

import os, sys
import numpy as np
from chartpeer.analysis import statistics

def gbm (dataset, extra, n=1, sampling='normal'):

    '''
    Geometric Brownian Motion.
    https://en.wikipedia.org/wiki/Geometric_Brownian_motion

    Arguments
    extra           extrapolation length per generation
    n               number of generations
    sampling        sampling algorithm,
                    "gauss" will result in Black Scholes
    
    Return
    1D generation array if n = 1
    list of 1D generation arrays if n > 1
    '''
    
    generations = []

    if n < 1 or type(n) != int:
        raise ValueError('n must be an integer >= 1!')
    
    if sampling == 'normal':

        for _ in range(n):

            generation = []
            drift = statistics.drift(dataset)
            volatility = statistics.volatility(dataset)
            wiener = np.random.norm(0, 1, extra)

            for i in range(extra):

                if len(generation) == 0:
                    price = dataset[-1]
                else:
                    price = generation[-1]

                vol = float(wiener[i] * volatility)
                price_new = price * (1 + drift + vol)
                generation.append(price_new)

            generations.append(np.array(generation))
    
    elif sampling == 'intrinsic':

        # construct intrinsic probability distribution
        # from the cummulative inverse
        logReturns = []
        for i in range(len(dataset)-1):
            logReturns.append(np.log(dataset[i+1]/dataset[i]))
        logReturns = np.array(logReturns)
        returnIncrement = 0.0001
        start, stop = np.min(logReturns), np.max(logReturns)
        returnRange = stop - start
        bins = int(returnRange/returnIncrement) + 1
        returnSpace = np.linspace(start, stop, bins)
        likelihood = np.zeros((bins,))
        for r in logReturns:
            for pointer in range(bins-1):
                if r < returnSpace[pointer+1] and r > returnSpace[pointer]:
                    likelihood[pointer] = likelihood[pointer] + 1
                    break
        N = len(logReturns)
        likelihood = [val/N for val in likelihood]  # normalize

        # build cummulative distribution function
        cummulativeDistribution = [likelihood[0]]
        for i in range(1, len(likelihood)):
            cummulativeDistribution.append(cummulativeDistribution[-1]+likelihood[i])

        # inverse sampling
        for _ in range(n):

            generation = []

            for j in range(extra):

                if len(generation) == 0:
                    price = dataset[-1]
                else:
                    price = generation[-1]

                u = np.random.uniform(0,1)
                logReturn = 0
                for i in range(len(cummulativeDistribution)):
                    if u < cummulativeDistribution[i]:
                        logReturn = returnSpace[i]
                        break
                    
                _return = np.exp(logReturn)
                generation.append( price * _return )

            generations.append(np.array(generation))

    else:

        raise ValueError(f'sampling argument must be normal or intrinsic, not {sampling}')

    # n-dependent return
    if n == 1:
        return np.array(generations[0])
    return np.array(generations)

def gbmBracket (gbm, limitPrice, stopPrice):

    '''
    Compute likelihood statistics for a price bracket, based on gbm generations.
    Returns a dict object with likelihoods to reach limit, stoploss and corresponding standard error within [0,1].
    '''
    
    w, l = 0, 0
    for g in gbm:
        for i in range(1,len(g)):
            if g[i] >= limitPrice and g[i-1] < limitPrice or g[i] <= limitPrice and g[i-1] > limitPrice:
                w += 1
                break
            elif g[i] >= stopPrice and g[i-1] < stopPrice or g[i] <= stopPrice and g[i-1] > stopPrice:
                l += 1
                break
    return {
        'limit': w/len(gbm),
        'stop': l/len(gbm),
        'std_err': 1/np.sqrt(len(gbm))}

def hw (dataset, extra, alpha, beta, gamma, periodInIntervals=None): 

    '''
    Holt Winters Algorithm.
    Triple season smoothing according to the smoothing parameters [alpha, beta, gamma].
    The parameters can be tuned by hand or pre computed automatically using hw_fit.

    Arguments:

    periodInIntervals       Number of intervals to screen for.
                            If the dataset contains daily prices,
                            then 7, 14, 30 would be appropriate,
                            for weeks, bi-weekly, or monthly period window.
                            If None the intervals will be half the total intervals in the dataset.
    '''

    d = dataset
    if type(d) == np.ndarray:
        T = d.shape[0]
    else:
        T = len(d)

    # get the main period (account for nyquist bound)
    if periodInIntervals :
        period = int(periodInIntervals)
    else:
        period = int(T/2)-1

    if T <= 2*period: raise ValueError('Data length must be at least 2 periods!')
    L, trend, pred = period, 0, []
    for i in range(L+1):
        trend += (d[L+i]-d[i])/(L**2)
    c, A, N = [], [], int(T/L)
    for j in range(N):
        sum = 0
        for i in range(L):
            sum += d[L*j+i]/L
        A.append(sum)
    for i in range(T):
        sum = 0
        for j in range(N):
            sum += d[L*j+i%(L-1)]/A[j]
        sum /= N # mean
        c.append(sum)
    S,B,C = [d[0]],[trend],[]
    for t in range(T):
        if t <= L:
            t_L = (t+T-1-L)%(T-1)
        else:
            t_L = t-L
        S.append(alpha*(d[t]/c[t_L]) + (1-alpha)*(S[-1]+B[-1]))
        B.append(beta*(S[-1]-S[-2]) + (1-beta)*B[-1])
        C.append(gamma*d[t]/S[-1] + (1-gamma)*c[t_L])
    for k in range(extra):
        pred.append((S[-1] + k*B[-1]) * C[(T-L+1+(k-1))%L])
    return np.array(pred)

def hw_fit (dataset, extra, periodInIntervals=None, fitRange=[0,1]):

    '''
    A spectral winters fit algorithm for [alpha, beta, gamma].
    The most optimal parameters are calculated using the least mean square error method.
    Returns python list object [alpha, beta, gamma].
    '''

    # determine total length
    d = dataset
    if type(d) == np.ndarray:
        T = d.shape[0]
    else:
        T = len(d)

    # determine period
    if periodInIntervals :
        period = int(periodInIntervals)
    else:
        period = int(T/2)-1

    # split
    testInput = dataset[:-extra]
    testTarget = dataset[-extra:]

    # convert to arrays if not yet
    for t in [testInput, testTarget]:
        if type(t) != np.ndarray:
            t = np.array(t)

    # fit algorithm
    fitStep = .001
    bins = int((fitRange[1]-fitRange[0])/fitStep)
    space = np.linspace(fitRange[0], fitRange[1], bins)
    alpha, beta, gamma = fitRange[0], fitRange[0], fitRange[0]
    for s in range(3):
        error, count = 10**100, 0
        for i in range(bins):
            out = hw(dataset, extra, alpha, beta, gamma, period)
            e = ((out - testTarget)**2).mean(axis=0)
            if e < error:
                error = e
                count = i
            if s == 0:
                alpha = space[i]
            elif s == 1:
                beta = space[i]
            elif s == 2:
                gamma = space[i]
        if s == 0:
            alpha = space[count]
        elif s == 1:
            beta = space[count]
        elif s == 2:
            gamma = space[count]

    # avoid edge results
    if alpha == 0:
        alpha = fitStep
    elif alpha == 1:
        alpha = 1-fitStep
    if beta == 0:
        beta = fitStep
    elif beta == 1:
        beta = 1-fitStep
    if gamma == 0:
        gamma = fitStep
    elif gamma == 1:
        gamma = 1-fitStep

    return [alpha, beta, gamma]


try:
    
    
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # no warning printing
    from keras.models import Sequential
    from keras.layers import Dense, LSTM
    from sklearn.preprocessing import MinMaxScaler
    import sys
    from traceback import print_exc

    class lstm:

        '''
        An artificial intelligence approach using Tensorflow/Keras and scientific python. 
        This RNN learns sequential patterns right on the spot and gives decent results with only limited amount of data points. 
        Using this implementation only a passing of an 1D array is demanded (rest is optional). 
        The automated algorithm will split the data into training sets and train itself epoch-wise and return a prediction of custom length.
        '''

        def __init__(self, sequence_length=60, feature_length=24, epochs=1, batch_size=1, neurons=100):

            # initialize
            self.initialize(sequence_length, feature_length, epochs, batch_size, neurons)

        def initialize(self, sequence_length=60, feature_length=24, epochs=1, batch_size=1, neurons=100):

            # make parameters global
            self.sequence_length = sequence_length
            self.feature_length = feature_length
            self.epochs = epochs
            self.batch_size = batch_size
            self.neurons = neurons

            # Build the LSTM model (ARCHITECTURE)
            self.model = Sequential()
            self.model.add(LSTM(neurons, return_sequences=True, input_shape=(sequence_length, 1)))
            self.model.add(LSTM(neurons, return_sequences=False))
            self.model.add(Dense(int((neurons+feature_length)/2)))
            self.model.add(Dense(feature_length))

            # compile the model
            self.model.compile(optimizer='adam', loss='mean_squared_error')
        
        def predict(self, data):

            # ---- Parameters ---- #
            if type(data) is list:
                self.training_data_length = len(data)
            elif type(data) is np.ndarray:
                self.training_data_length = data.shape[0]
            else:
                raise TypeError('data must be a 1d array or list!')
            # -------------------- #


            # ==== TRAINING ==== #

            # check the length of the data
            self.data_length = self.training_data_length

            # convert data to readable numpy array (mimick dataframe.values)
            data = np.array([[i] for i in data])
            self.data_backup = np.copy(data)

            # rescale the data (pre-processing)
            self.scaler = MinMaxScaler(feature_range=(0,1))
            self.scaled_data = self.scaler.fit_transform(data)

            # create the training data set
            self.train_data = self.scaled_data[0:self.data_length,:]

            # split the data 
            self.x_train, self.y_train = [], []
            for i in range(self.sequence_length, len(self.train_data)-self.feature_length):
                self.x_train.append( self.train_data[i-self.sequence_length:i, 0] )
                self.y_train.append( self.train_data[i:i+self.feature_length, 0])

            # convert x and y lists to numpy arrays
            self.x_train, self.y_train = np.array(self.x_train), np.array(self.y_train)
            
            # reshape the data 
            self.x_train = np.reshape(self.x_train, (self.x_train.shape[0], self.x_train.shape[1], 1))

            # fit model
            self.history = self.model.fit(self.x_train, self.y_train, batch_size=self.batch_size, epochs=self.epochs).history



            # ==== TESTING ==== #

            # keep prediction list for appending
            data = self.data_backup
            
            # refresh the data iteratively
            self.scaled_data = self.scaler.fit_transform(data)
            
            # create testing data
            self.test_data = self.scaled_data[self.training_data_length-self.sequence_length:,:]

            # create data sets x_test without y_test as this is not known
            self.x_test = [self.test_data]

            # convert the data to a numpy array
            self.x_test = np.array(self.x_test)

            # reshape the data
            self.x_test = np.reshape(self.x_test, (self.x_test.shape[0], self.x_test.shape[1], 1))

            self.predictions = self.model.predict(self.x_test)

            # unscale back from the 0,1 scale
            self.predictions = self.scaler.inverse_transform(self.predictions)

            return {"prediction": self.predictions[0], "loss": self.history['loss'][-1]}

except ImportError as e:

    print('LSTM could not be loaded due to missing modules:', e)
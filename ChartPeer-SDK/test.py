from chartpeer.loader import load
from chartpeer.extrapolate import lstm
from chartpeer.analysis import plot

# load bitcoin daily data (1440 minutes = 24h interval)
# extract the last 365 days
data = load.closedFromFile('XBTUSD_1440.csv')[-365:]

# initialize an LSTM network which predicts prices of next 14 days based on last 60 days, 
# for training all 365 days are sliced into training sets
nn = lstm(sequence_length=60, feature_length=7, epochs=50,  batch_size=10)
output = nn.predict(data) 
prediction = output['prediction']

# plot prediction
plot.chart(data, name='bitcoin closed', predictionSets = {
    'LSTM prediction': prediction
})
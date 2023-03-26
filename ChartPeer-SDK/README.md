<h1 align=center>ChartPeer SDK</h1> 

A developer kit with a user-friendly API to various technical analysis tools for traders.

- Indicators
- Regression
- Stochastic Prediction
- Machine Learning
- Plotting

Works great in Jupyter environments (also see [VSCode Extension](https://github.com/Microsoft/vscode-jupyter)).

<br>

# Getting Started
## Install 
```bash
git clone https://github.com/B0-B/chartPeer-trading/chartpeer-sdk 
cd chartpeer-sdk
python setup.py install
```

## Usage

```python
from chartpeer.loader import load
from chartpeer.extrapolate import lstm
from chartpeer.analysis import plot

# load bitcoin daily data (1440 minutes = 24h interval)
# extract the last 365 days
data = load.closedFromFile('XBTUSD_1440.csv')[-365:]

# initialize an LSTM network which predicts prices of next 14 days based on the previous 60 days, 
# for training all 365 days are sliced into training sets
nn = lstm(sequence_length=60, feature_length=14, epochs=50,  batch_size=10)
output = nn.predict(data) 
prediction = output['prediction']

# plot prediction
plot.chart(data, name='bitcoin closed', predictionSets = {
    'LSTM prediction': prediction
})
```

<img src='img/lstm.png'>

## Examples

More examples can be found within the [tutorial](https://github.com/B0-B/chartPeer-trading/blob/main/ChartPeer-SDK/tutorial.ipynb) and [live prediction](https://github.com/B0-B/chartPeer-trading/blob/main/ChartPeer-SDK/lstm_gbm_live_prediction.ipynb) notebooks.
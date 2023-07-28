<h1>LSTM-driven Geometric Brownian Motion with Intrinsic Volatility Monte Carlo Sampling for Timeseries Prediction</h1>





<h1>LSTM-driven Monte Carlo Sampling for Geometric Timeseries Forecasting</h1>
<h1>LSTM-driven Monte Carlo for Geometric Timeseries Forecast</h1>
<h1>LSTM-driven Monte Carlo Forecast for Geometric Timeseries Data</h1>

1. Background
   1. Black–Scholes–Merton Assumption
      1. Random Walk Hypothesis 
      2. Volatility and Drift
      3. Itô's Solution
      4. Data Generation
      5. Limits and Genericity Argument
   2. Extensions of Geometric Brownian Motion
      1. Stochastic Volatility Models
2. Proposal
   1. Geometric Argument
   2. Multivariate Analysis of SDE
   3. Efficacy of Long Short-Term-Memory RNNs
   4. Intrinsic Volatility Extraction
3. Sampling
   1. Generic and Selected Monte Carlo
   2. LSTM-steered Integration of SDE
   3. Unbiased Estimator Model
4. Results
5. Conclusion


# 1. Background

The following derivation of geometric brownian motion and black scholes theory is for the sake of understanding and motivated by the outlined proposal in this paper.

## 1.1 Black–Scholes–Merton Assumption


Black and Scholes showed already in the late 1960s that the dynamic revision of a portfolio can be formed to compensate the expected return of an underlying security. Their work evolved to a finally finished version in 1973 "The Pricing of Options and Corporate Liabilities" which successfully implemented risk management using a dynamic volatility model to predict the price development of options using probabilitic analysis in hedging positions. Their model can be then used to model a financial market or instruments which contain a devative option.

The pricing model underlies (by one part) the price modelling of the security. The security modelling can then be applied to other models and thus is not constrained to options modelling, why this paper will shift the focus away from derivatives theory and instead extract the core argument for more essential timeseries prediction.

### 1.1.1 Random Walk Hypothesis

In  general the "market" consists of the risk market or asset (stock, security, commodity etc.) and the risk-free market or instrument allowing for direct exchange between both markets. Thus the price of the risky asset can be measured in stock units of the risk-free market (something stable e.g. USD bond) and is assumed to be non-zero and discretely and equidistantly seperated by constant intervals of time $\Delta t$. This helps in the following definitions as $t \in \mathbb{N}$ acts as a natural discrete index such that the total time is mapped in linear increments $T \coloneqq t \cdot \Delta t$ with $\Delta t \in \mathbb{R}$. Prices will be denoted as $S_{t} \in \mathbb{Q}^+$ and can be related (dynamically) through the discrete evolution of time as $S_{t+1} \leftarrow S_{t}$.

Before introducing geometric brownian motion and black scholes theory 
In general, note that timeseries values can be interpreted in two ways: absolutely or relatively (successor price relators) - latter method was successfully used in in pure and applied mathematics, such as finance and other timeseries forecasts. Within uncertainty-related fields such as hedging positions it can be shown that the proper quantitative risk management allows for more profitable decision making.

Every  timeseries can be interpreted geometrically and this argument holds since

A geometric and discrete time step is said to be stationary if it can be expressed recursively 

$R_t : S_{t} = R_t \cdot S_{t-1}  \Leftrightarrow R_t := S_{t}/S_{t-1} \in \mathbb{Q^+}$

thus the price reading $S$ at some arbitrary time $t$ can be formulated in pure statistical  dependence to it' 
therefore a price at some arbitrary time $t$ can be reconstructed by knowing the starting price at $t_0 := 0$ and all returns

$S_{t} \leftarrow R_{t} \cdot S_{t-1} \leftarrow R_{t-1} \cdot S_{t-2} ... \leftarrow R_{1} \cdot S_0$

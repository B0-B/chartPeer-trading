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

The price of one instrument is assumed to be non-zero and discretely and equidistantly seperated by constant intervals of time $\Delta t$. This helps in the following definitions as $t \in \mathbb{N}$ acts as a natural discrete index such that the total time is mapped in linear increments $T \coloneqq t \cdot \Delta t$ with $\Delta t \in \mathbb{R}$. Prices will be denoted as $S_{t} \in \mathbb{Q}^+$ and can be related (dynamically) through the discrete evolution of time as $S_{t+1} \leftarrow S_{t}$.

In  general the "market" consists of the risk market or asset (stock, security, commodity etc.) and the risk-free market or instrument allowing for direct exchange between both markets. Thus the price of the risky asset can be measured in stock units of the risk-free market (something stable e.g. USD bond) and is assumed to be non-zero and discretely and equidistantly seperated by constant intervals of time $\Delta t$. This helps in the following definitions as $t \in \mathbb{N}$ acts as a natural discrete index such that the total time is mapped in linear increments $T \coloneqq t \cdot \Delta t$ with $\Delta t \in \mathbb{R}$. Prices will be denoted as $S_{t} \in \mathbb{Q}^+$ and can be related (dynamically) through the discrete evolution of time as $S_{t+1} \leftarrow S_{t}$.


A geometric and discrete time step is said to be stationary if it can be expressed recursively, between all consecutive steps the relation is preserved as 

$$R_t : S_{t} = R_t \cdot S_{t-1}  \Leftrightarrow R_t := S_{t}/S_{t-1} \in \mathbb{Q^+}$$

therefore a price at some arbitrary time $t$ can be reconstructed by knowing the starting price at $t_0 := 0$ and all returns

$$S_{t} \leftarrow R_{t} \cdot S_{t-1} \leftarrow R_{t-1} \cdot S_{t-2} ... \leftarrow R_{1} \cdot S_0$$

$S_t$ is assumed to obey a random walk (determined by the random vector $(R_1,..., R_t) :log(R_i)\sim\mathcal{N}(\mu,\sigma)$), this however does not automatically involve ergodicity. It can be shown that the mean of the arithmetically determined expected return $\mu=\left \langle log(R_i) \right \rangle$ does not resemble the mode i. e.  $argmax(P(log(R_i)))$ the most likely return.

### 1.1.2 Drift and Volatility

In the following we will define a geometric mean of the return in which we assume that on average $R_i = \left \langle R \right \rangle_g = const.$, the g indicates the geometric mean, across all discrete intervals. Although it needs to be disclaimed that this assumption is an expectation, with the $R_i$ fluctuating around it but when meaned will converge to this expectation value. This allows the product to collapse to a single exponential notation 


$$S_{t} = \prod_{i=0}^{t} R_{i} \cdot S_{0} \equiv \left \langle R \right \rangle_g^{t} \cdot S_{0} \\

\left \langle R \right \rangle_g := \sqrt[t]{\prod_{i=0}^{t} R_{i}}
$$

Substitute this notation in the initial price equation

$$
S_{t} = S_{0} \cdot \left \langle R \right \rangle_g^{t} = S_{0} \cdot e^{ \left\langle log \left (  R \right ) \right\rangle_a \cdot t } \equiv  S_{0} \cdot e^{ \,\mu \cdot t } \\

\mu := \left\langle log \left (  R \right ) \right\rangle_a $$

in the last step the global parameter $\mu$ was introduced which in literature is often declared as the "drift" i.e. the expectation (arithmetic mean) of logarithmic returns. Hence a constant geometric rate $\left\langle R \right\rangle_g$ (on increment scale) evidently results in an exponential time evolution of the price (on macroscopic scale). One obtains this form where the price depends only on time and is characterized by a single constant scalar - the drift. This parameter fixes the exponential falloff, and acts as the expected return (meaned over an interval unit) and thus needs to be unit-less since we initially defined $t$ without a unit.

We arrive at the expected time-dependent price equation which can be modelled geometrically as

$$S_{t} =  S_{0} \cdot e^{ \,\mu \cdot t }$$

where equidistant increments in time yield a constant growth rate for the price.

To focus on the more realistic case we introduce mutually distinct returns $R_t \neq const.$ to arrive at the general price equation again, recall

$$
S_{t} = S_{0} \cdot \prod_{i=1}^{t} R_{i}  $$

This time the $R_i$ will be estimated in a similar fashion as for the drift, using a mean-field approach. However, we are not focusing on the expected mean solution, which will evidently result in the drift again, but in the fluctuations.

To get a proper statistical interpretation in terms of an expectation and standard deviation an arithmetic mean is needed, for this purpose the drift definition becomes very handy, and will justify the apriori derivation effort. The mean field ansatz is analogous to the mean derivation except for one extension: every return can be expressed logarithmically to exploit the artithmetic form with constant mean and varying fluctuation

$$log(R_i) = \left\langle log \left (  R \right ) \right\rangle_a + \delta log(R_i) \\
\Leftrightarrow log(R_i) = \mu + \delta \mu_i
$$

since $R_i \neq R_j \, \forall \, i \neq j$ and $\mu$ is constant by definition, the only left option for a unique return formation are unique fluctuations

$$
\delta log(R_i) = log(R_i) - \left\langle log \left (  R \right ) \right\rangle_a \\
$$

<strong>Geometric brownian motion</strong> assumes randomly generated increments (sampled from a modelled distribution) on every interval. The differences $log(R_i) - \left\langle log \left (  R \right ) \right\rangle_a$ are assumed to be statistically independent and origin from a standard normal distribution which fulfill the definition of a <strong>Wiener process</strong>

$$
W := \left \{ W_i: \, W_i-W_{i-1} = \frac{log(R_i) - \left\langle log \left (  R \right ) \right\rangle_a}{\sigma} \sim \mathcal{N}(0,1) \land P(W_i \mid W_j) = P(W_i) \cdot P(W_j) \, \forall \, i,j \right \} 
$$

the increments $dW_i := W_i-W_{i-1}$ are scaled by a relative problem-specific factor $\sigma$, measured in same dimension as $\mu$. When cummulated they make up the $i$-th value in the sequence i. e. $W_t=dW_1+...+dW_t$.

Thus every incremental $R_i$ can be interpreted to move according to the expected incremental drift which is accompanied by a unique fluctuation around the drift expectation. $\sigma$ is called the <strong>volatility</strong> which accounts for the scaling of the random wiener contribution and is associated with a scalar uncertainty merit across the process. An arbitrary logarithmized return can be described as

$$
log(R_i) = \underset{\left\langle log \left (  R \right ) \right\rangle_a}{\underbrace{\mu}}  + \underset{\delta log(R_i) }{\underbrace{ \sigma \cdot dW_i }} 
$$

Note that this extension still respects the apriori derived mean solution, since in the $\sigma = 0$ case we return to the mean solution, this will be consistent in the following.

All $R_{i}$ are stat. indep. which legitimates the product notation 

$$
S_{t} = S_{0} \cdot \prod_{i=1}^{t} R_{i} = S_{0} \cdot \prod_{i=1}^{t} e^{log(R_i)} = S_{0} \cdot \prod_{i=1}^{t} e^{ \, \mu + \sigma \cdot dW_i} = S_{0} \cdot e^{\mu t} \prod_{i=1}^{t} e^{ \sigma \cdot dW_i}
$$

The uncertain wiener volatility contributes as a decoupled (indep.) global factor, a product at first glance, however the $W_i$ were sampled independent of each other and of time

$$
\prod_{i=1}^{t} e^{ \sigma \cdot dW_i} = e^{\sigma \cdot dW_1} \cdot ... \cdot e^{\sigma \cdot dW_t} = e^{\sigma \cdot \left\langle dW_i \right\rangle_a \cdot t}  \equiv e^{\sigma \cdot \sqrt{t} \cdot dW \cdot t}
$$

in essence 

$$
\sigma \cdot (dW_1 + ... + dW_t) = \sigma \cdot  \sqrt{t} \cdot dW \cdot t \,\, where \,\, dW_i \sim \mathcal{N}(0,1) \land \left\langle dW_i \right\rangle_a \sim \mathcal{N}(0,\frac{1}{\sqrt{t}}) \land \sqrt{t} \cdot dW \sim \mathcal{N}(0,1) \,\, for \,\, t\rightarrow \infty
$$

where $W$ acts as an arithmetic ("standardized") mean, simply by comparing with the left sum, and for large $t$ is governed by the `central limit theorem` which states that the sum of identically distributed and independently sampled $W_i$ (called iid), $W$ will (almost surely) converge to a normal distribution identically to the $W_i$, just scaled by a factor $\sqrt{t}$. Further concluded, the cumulated value (actual Wiener process value) $W_i$ which by definition can be formulated as an arithmetic expectation as well

$$
W_t := dW_1 + ... + dW_t = \frac{dW_1 + ... + dW_t}{t} \cdot t = \left\langle dW_i \right\rangle_a \cdot t = \sigma \cdot \sqrt{t} \cdot dW \cdot t
$$

can be expressed now by a single increment sample $dW$ from $\mathcal{N}(0,\frac{1}{\sqrt{t}})$. The equality arises from the same origin distribution for both random variables $\left\langle dW_i \right\rangle_a  \sim \sqrt{t} \cdot dW  \sim \mathcal{N}(0,1)$.

After inserting this into the product above we finally arrive at the <strong>mean-field-approximated</strong> geometric brownian price equation

$$
S_t = S_0 \cdot e^{\mu t + \sigma \cdot \sqrt{t} \cdot dW \cdot t} = S_0 \cdot e^{\mu t + \sigma \cdot W_t}
$$

which depends on time, the drift $\mu$ and volatility $\sigma$, as well as the simple integrated Wiener process value $W_t$. To highlight the improvement both versions are denoted for comparison: rather than sampling and integrating $t$ independend samples to obtain $W_t$, $S_t$ can be computed by a single sample obtained from $\mathcal{N}(0,1/\sqrt{t})$. This finding increases the computing efficiency by a factor of $t$ and the estimated equality is being approached almost surely for $t\rightarrow \infty$.


### 1.1.3 Itô's Solution

The interpretation in the last section is biased from the actual definition of the geometric brownian motion. This is because the found solution

$$
S_t = S_0 \cdot e^{\mu t + \sigma \cdot W_t}
$$

does not solve the stochastic differential equation

$$
dS_t = \mu S_t \cdot dt + \sigma \cdot S_t \cdot dW_i
$$

where $dt$ was neglected in the derivation before since $t$ was defined to be discrete and $\mu$ was meaned (to compensate the discrete time steps) over the actual interval $\Delta t$.

Instead, the analytical solution is

$$
S_t = S_0 \cdot e^{(\mu-\frac{\sigma^2}{2}) \cdot t + \sigma \cdot W_t}
$$


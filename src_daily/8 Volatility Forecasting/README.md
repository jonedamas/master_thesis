# Volatility Forecasting


## Input Data Format

- datetime: The date and time the article was published
    - **Type:** Datetime64[ns]: `YYYY-MM-DD hh:mm:ss`

- SVI: The Stock Volatility Index
    - **Type:** Float

## Models

### GARCH

Generalized Autoregressive Conditional Heteroskedasticity (GARCH) is a type of time series model that is used to model the volatility of a time series. In this model, we use a GARCH model to predict the volatility of a stock. The model is trained on a time series of historical stock prices and returns, and is able to predict the volatility of the stock at each time step. The model is trained using the maximum likelihood estimation method. The model is evaluated using the root mean squared error (RMSE) and the mean absolute error (MAE) on a test set of historical stock prices and returns.

$$
\begin{gather*}
r_t = \mu + \epsilon_t \\[10pt]
\epsilon_t = \sigma_t z_t \\[10pt]
\sigma_t^2 = \omega + \sum_{i=1}^p\alpha_i \epsilon_{t-1}^2 + \sum_{j=1}^q\beta_j \sigma_{t-1}^2
\end{gather*}
$$

### LSTM

Long Short-Term Memory (LSTM) is a type of recurrent neural network that is capable of learning order dependence in sequence prediction problems. In this model, we use an LSTM to predict the volatility of a stock. The model is trained on a time series of historical stock prices and returns, and is able to predict the volatility of the stock at each time step. The model is trained using the Adam optimizer and the mean squared error loss function. The model is evaluated using the root mean squared error (RMSE) and the mean absolute error (MAE) on a test set of historical stock prices and returns.

### BI-LSTM

Bi-directional Long Short-Term Memory (BI-LSTM) is a type of recurrent neural network that is capable of learning order dependence in sequence prediction problems. In this model, we use a BI-LSTM to predict the volatility of a stock. The model is trained on a time series of historical stock prices and returns, and is able to predict the volatility of the stock at each time step. The model is trained using the Adam optimizer and the mean squared error loss function. The model is evaluated using the root mean squared error (RMSE) and the mean absolute error (MAE) on a test set of historical stock prices and returns.

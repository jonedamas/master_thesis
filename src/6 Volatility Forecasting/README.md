# Volatility Forecasting

## Parameter set explanation

The `DATA_PARAMS` dictionary contains parameters used to configure the preprocessing and data handling for training the RNN model. These parameters collectively define how the data is prepared and split for training, validation, and testing. Here’s a summary:

- **Feature Columns**: Specifies the columns used as input features for the model.
- **Target Column**: Defines the column used as the target variable for prediction, in this case, 'REALIZED_VOL'.
- **Window Size**: Sets the size of the rolling window used to create sequences of data for the RNN input.
- **Test Size**: Indicates the proportion of the dataset to be used for testing, set to 20% in this case.
- **Validation Size**: Indicates the proportion of the dataset to be used for validation, also set to 20% in this case.
- **Scaler Type**: Specifies the type of scaler used to normalize the data, with 'RobustScaler' being chosen to handle outliers effectively.

These parameters are essential for ensuring that the data is properly prepared and split for training, validation, and testing, leading to better model performance and generalization.

***

The `MODEL_PARAMS` dictionary contains hyperparameters used to configure a Recurrent Neural Network (RNN) model. These parameters collectively define the structure and training behavior of the model. Here’s a summary:

- **Layer Units**: Specifies the number of neurons in the first and second RNN layers, controlling the model’s capacity to learn patterns.
- **L2 Regularization Strength**: Applies a penalty on large weights to prevent overfitting by encouraging smaller, more generalized weights.
- **Learning Rate**: Sets the step size for the optimizer during training, affecting how quickly the model learns.
- **Batch Size**: Defines the number of samples processed before updating the model’s parameters, impacting training stability and memory usage.
- **Gaussian Noise Standard Deviation**: Adds noise to the input data to make the model more robust to variations and prevent overfitting.

These hyperparameters are crucial for tuning the model’s performance and ensuring it generalizes well to new, unseen data.

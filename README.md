# Master Thesis Repository

## Repository structure

### src

This folder contains the source code of the project.

#### 1 - Data Gathering

***
This folder contains all news and futures price data gathering scripts. The scripts are used to gather data from Refinitiv Eikon API.

**Files:**

- eikon_headline_retrieval.ipynb
- eikon_HF_price_retrieval.ipynb
- eikon_stories_retrieval.ipynb

#### 2 - Data Cleaning and EDA

***
This folder contains all data cleaning and exploratory data analysis scripts. The scripts are used to clean and explore the gathered data.

**Files:**

- crude_oil_grade.ipynb
- news_data_analysis.ipynb
- price_data_analysis.ipynb
- text_treatment.ipynb
- word_analysis.ipynb

#### 3 - Topic Modelling

***
This folder contains all topic modelling scripts. The scripts are used to identify latent topics in the news data using LDA.

**Files:**

- topic_modelling.ipynb

#### 4 - Sentiment Analysis

***
This folder contains all sentiment analysis scripts. The scripts are used to perform sentiment analysis on the news data.

**Files:**

- sentiment_analysis.ipynb
- sentiment_applier.ipynb
- sentiment_index_analysis.ipynb

#### 5 - Event Analysis

***
This folder contains all event analysis scripts. The scripts are used to identify events in the news data and its effect on volatility.

**Files:**

- VAR.ipynb

#### 6 - Feature Engineering

***
This folder contains all feature engineering scripts. The scripts are used to create and filter features for the volatility forecasting model, and prepare the data.

**Files:**

- data_preparation.ipynb
- feature_selection.ipynb
- LASSO.ipynb

#### 7 - Volatility Forecasting

***
This folder contains all volatility forecasting scripts. The scripts are used to train and evaluate the volatility forecasting model.

**Files:**

- fit_GARCH.ipynb
- fit_RNN.ipynb
- forecast_comparison.ipynb
- hyperpm_tuning.ipynb

#### utils

***
This folder contains all utility functions used in the project.

### data

This folder contains the data used in the project.

## Usefull articles

### Sentiment Analysis

[Oil price volatility and new evidence from news and Twitter](https://www.sciencedirect.com/science/article/pii/S0140988323002098?ref=pdf_download&fr=RR-2&rr=81fbb4f16e03569a)

[Investor sentiment and machine learning: Predicting the price of China's crude oil futures market](https://www.sciencedirect.com/science/article/pii/S0360544222003747?pes=vor)

### RNN forecasting

[A Deep Learning based Model for Predicting the future prices of Bitcoin](https://ieeexplore.ieee.org/document/10157841)

<!-----



Conversion time: 0.376 seconds.


Using this Markdown file:

1. Paste this output into your source file.
2. See the notes and action items below regarding this conversion run.
3. Check the rendered output (headings, lists, code blocks, tables) for proper
   formatting and use a linkchecker before you publish this page.

Conversion notes:

* Docs to Markdown version 1.0β44
* Fri Feb 21 2025 15:49:31 GMT-0800 (PST)
* Source doc: project_proposal
----->


Predicting Stock Market Movements Using Machine Learning


## 1. Introduction/Background

Financial markets exhibit complex, dynamic behavior influenced by numerous factors such as economic indicators, investor sentiment, and technical trends. Traditional methods like fundamental and technical analysis have been widely used, but machine learning (ML) has emerged as a powerful alternative to improve predictive accuracy. Several studies have explored the use of ML techniques such as Support Vector Machines (SVM), Long Short-Term Memory (LSTM) networks, and ensemble methods to forecast stock prices or price movements.

For this project, we aim to develop a stock market prediction model using historical market data and technical indicators. The dataset will consist of stock prices, trading volume, and financial indices to train various ML models for classification and regression tasks.


### Literature Review



1. Patel et al. (2015) applied ML techniques, including Random Forests and SVM, to predict stock trends, demonstrating improved accuracy over traditional methods.
2. Fischer and Krauss (2018) employed LSTM networks for financial time series forecasting and found deep learning superior to shallow models.
3. Chen et al. (2021) explored ensemble learning and reinforcement learning for stock trading strategies, achieving competitive performance.


### Dataset Description

We will use publicly available financial datasets, such as:



* Yahoo Finance or Alpha Vantage: Provides historical stock data including Open, High, Low, Close prices (OHLC) and volume.
* Kaggle Stock Market Data: Offers curated datasets with additional financial indicators.
* FRED Economic Data: Provides macroeconomic indicators such as interest rates and inflation.
* CRSP (Center for Research in Security Prices) Data: Offers comprehensive, high-quality data on stock prices, returns, and volumes for securities traded on major U.S. exchanges.
* Fama-French 5 Factor Model Data: Provides factor returns (MKT-RF, SMB, HML, RMW, CMA) to help explain stock returns and assess investment performance based on size, value, profitability, and investment patterns.



## 2. Problem Definition


### Problem:

Stock price prediction is inherently challenging due to market volatility, noise, and complex dependencies. The objective is to classify whether a stock’s price will increase, decrease, or remain neutral based on historical data and technical indicators.


### Motivation:

Accurate stock market prediction is valuable for retail and institutional investors, helping optimize investment strategies. Machine learning offers the ability to analyze large datasets, detect patterns, and improve decision-making beyond human capabilities.


## 3. Methods


### Data Preprocessing Methods:



1. Feature Engineering: Compute technical indicators like Moving Averages (SMA, EMA), Relative Strength Index (RSI), and Bollinger Bands.
2. Data Normalization: Scale numerical features to improve model convergence.
3. Time-Series Windowing: Convert sequential stock data into supervised learning format.


### Machine Learning Models:



1. Random Forests: For feature importance analysis and robust classification.
2. LSTM Networks: To capture sequential dependencies in time-series data.
3. XGBoost: Gradient boosting for improved predictive performance.
4. Hidden Markov Models (HMM): To capture latent market conditions.

[CS 7641/CS 4641 Requirements:



* Supervised Learning: LSTMs and XGBoost for regression/classification.
* Unsupervised Learning: Principal Component Analysis (PCA) to reduce dimensionality.


## 4. (Potential) Results and Discussion


### Quantitative Metrics:



1. Accuracy (for classification models predicting stock movement up/down).
2. Mean Squared Error (MSE) (for regression models predicting stock prices).
3. Sharpe Ratio (for assessing trading strategy effectiveness based on ML predictions).


### Project Goals:



* Achieve a classification accuracy exceeding 55% (better than random guessing in a three-class setup).
* Maintain an MSE lower than 0.01 in regression models.
* Ensure ethical considerations, avoiding misleading financial predictions.


### Expected Results:

We expect deep learning models like LSTMs to outperform traditional ML models due to their ability to capture temporal dependencies. However, simpler models like Random Forests and XGBoost may offer more interpretability and robustness.


## 5. References

1. Sun, J. (2016, August). Daily News for Stock Market Prediction, Version 1. Retrieved 3/20/2025 from https://www.kaggle.com/aaron7sun/stocknews.
2. Valour, Maximus. (2025, January). Stock Market Sentiment Analysis, Version 1. Retrieved 3/20/2025 from https://www.kaggle.com/code/maximusvalor/stock-market-sentiment-analysis.
3. Patel, J., Shah, S., Thakkar, P., & Kotecha, K. (2015). Predicting stock market index using fusion of machine learning techniques. Expert Systems with Applications, 42(4), 2162-2172.
4. Fischer, T., & Krauss, C. (2018). Deep learning with long short-term memory networks for financial market predictions. European Journal of Operational Research, 270(2), 654-669.
5. Chen, W., Zhang, H., Zhang, W., & He, X. (2021). A survey on stock market prediction using machine learning and statistical techniques. IEEE Access, 9, 15039-15062.


## 6. Gantt Chart

https://gtvault-my.sharepoint.com/:x:/g/personal/lmanor3_gatech_edu/EQEjpjbv5GhMqv4rImskUEQBMC-x1_dlydS-XmuTr956hQ?e=s5Yhoi


## 7. Contributions Table

<table>
  <tr>
   <td><strong>Member</strong>
   </td>
   <td><strong>Proposal Contributions</strong>
   </td>
  </tr>
  <tr>
   <td>Munim
   </td>
   <td>In the written proposal: 
<p>
Literature Review
<p>
Dataset Description bullets 1-2
<p>
Machine learning models – supervised
<p>
Data processing methods 
   </td>
  </tr>
  <tr>
   <td>Amritesh
   </td>
   <td>In the written proposal: 
<p>
Dataset description bullets 3-5
<p>
Data processing methods
<p>
Quantitative metrics
   </td>
  </tr>
  <tr>
   <td>Luca
   </td>
   <td>Slides 2-9 of the PowerPoint for the video presentation
<p>
In the written proposal: 
<p>
Machine learning models – unsupervised
<p>
Problem Definition
   </td>
  </tr>
  <tr>
   <td>Lior
   </td>
   <td>Introduction/Background section in the written proposal
<p>
All GitHub operations
<p>
Gantt chart 
<p>
Voice-over for the video presentation
   </td>
  </tr>
  <tr>
   <td>Vamsi
   </td>
   <td>In the written proposal: 
<p>
Problem Definition
<p>
Project Goals / Expected Results
<p>
Facilitated meetings
   </td>
  </tr>
</table>


## 8. Project Award Eligibility 

Opt-in

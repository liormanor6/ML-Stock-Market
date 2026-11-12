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

Data collected and organized by Aaron Sun (see references [1])

Database 1: upload_DJIA_table.csv
Contains open, high, low, close, volume, and adjusted close data at different timepoints (dates)

Database 2: Combined_News_DJIA.csv
Contains top 25 news articles relevant to the stock at different timepoints (dates) along with a label “1” or “0”. “1” Represents days when DJIA adjusted close value rose or stayed the same, and “0” represents days when DJIA adjusted close value decreased

We do an inner merge for these two databases on the attribute “date”

## 2. Problem Definition


### Problem:

Stock price prediction is inherently challenging due to market volatility, noise, and complex dependencies. The objective is to classify whether a stock’s price will increase, decrease, or remain neutral based on historical data and technical indicators.


### Motivation:

Accurate stock market prediction is valuable for retail and institutional investors, helping optimize investment strategies. Machine learning offers the ability to analyze large datasets, detect patterns, and improve decision-making beyond human capabilities.


## 3. Methods


### Data Preprocessing Methods:

1. Sentiment extraction:
       Our original data contains headlines of the top 25 news articles relating to the stocks at a given date. We combined these headlines into a single, long string, and then ran it in vaderSentiment, which gave us compound, negative, positive, and neutral ratings of the headline string. We then added these into our database, so we could use the sentiment of the stock at a given date as an indicatior of whether its price will go up/stay neutral or go down.
   We chose this method because it allows us to quantify the market mood and tone for a certain stock. We predicted this will have a heavy influence on the behavior of the stock since the market is highly driven by investor sentiment, fear, uncertainty, and popularity of a stock, all factors that are heavily influenced by breaking news headlines regarding the stock.

2. Subjectivity/Polarity rating:
   In addition to setiment extraction, we used TextBlob class to extract the subjectivity and polarity ratings of the headlines and added it to the database.
   Similarly to sentimentality, subjectivity and polarity of news articles and stocks have a high chance of influencing the behvior of the stock. The opinion of news-writers on the stock can both represent and influence the general opinion of the public on the stock, which directly effects the confidence and nuance behind the investor's decisions to buy or sell a stock. This is therefore expected to have an effect on the price movement. 


### Machine Learning Models:

1. PCA:
   We used PCA for feature reduction as a pre-processing tool to feed into the LSTM model. We did this because our dataset contained many factors that are heavily correlated. For example, the higher the polarity of an article, the more likely it is to have a high positive or negative sentiment score. Because our data is correlated, PCA is helpful in order to reduce redundancies, and narrow the feature set to one that most well-represents our problem in the most feature-efficient way, while still maintaining 95% of the variance.LSTM Networks:
   
2. LSTM
   We used an LSTM network to model the temporal dependencies in stock market data, where daily movements are often influenced by patterns in recent sentiment and market behavior. Unlike traditional models that treat each data point independently, LSTM retains memory of previous inputs, allowing it to capture trends over time. This is especially useful when working with sentiment scores derived from headlines, which often have short-term momentum effects on the market. By leveraging LSTM’s ability to learn from sequences, our model is better equipped to predict directional changes in the DJIA.


[CS 7641/CS 4641 Requirements:

* Supervised Learning: LSTM
* Unsupervised Learning: PCA


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

### Results from Midpoint:

Here are the results from our pca analysis:

Shape after PCA reduction: (1989, 6)
Number of principal components selected: 6

Explained variance ratio for each retained component:
PC1: 0.3744
PC2: 0.1890
PC3: 0.1581
PC4: 0.0952
PC5: 0.0849
PC6: 0.0550

Top features in PC1:
  Low (loading = 0.4985)
  Open (loading = 0.4976)
  High (loading = 0.4970)

Top features in PC2:
  negative (loading = 0.6338)
  neutral (loading = -0.5502)
  compound (loading = -0.3599)

Top features in PC3:
  positive (loading = 0.6973)
  neutral (loading = -0.4098)
  polarity (loading = 0.3739)

Top features in PC4:
  subjectivity (loading = -0.9377)
  compound (loading = 0.2799)
  positive (loading = 0.1369)

Top features in PC5:
  polarity (loading = -0.7746)
  compound (loading = 0.5979)
  subjectivity (loading = 0.1894)

Top features in PC6:
  positive (loading = 0.6021)
  compound (loading = -0.5523)
  polarity (loading = -0.4163)

Here are the training results of the LSTM model:


1. F1 score = 0.69489
2. Precision = 0.57941
3. Recall = 0.86784

The high recall, shows that our model correctly identifies most of the actual positive cases. It rarely misses true positives and has low false negatives. 

The low precision means that our model currently also predicts many false positives. Which isn't ideal. It misclassfies many negatives as positives. 

Finally, the F1 Score is average or moderate. Great place to be at for the midpoint of this project. But there is definitly room for improvement in the precision without scarificing too much recall. 

We are happy with results we got so far for the LSTM model, it's an great start and ideally we want our precision and F1-score to be higher by the end of this project. 

Here are the training reuslts of the PCA:

Explained Variance Ratio per Component:
1. PC1: 37.44%
2. PC2: 18.90%
3. PC3: 15.81%
4. PC4: 9.52%
5. PC5: 8.49%
6. PC6: 5.50%

After PCA, the feature set was reduced to 6 prinicpal components, shrinking hte input space while also retaining around 95.7% of total variance. Which is an excellent result, as this suggests there is minimal information loss. This helps the LSTM focus on the good patterns in the data while reducing the noise, and risk of overfitting.

PC1 is dominated by price-based features, which suggests that PC1 captures general market trends or volatilty. PC2 and PC3 are influenced by text sentiment features. This means the sentiment extracted from text plays a major role in explaining variance and likely influences our model's predictions.



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
   <td> 
<p>
Results evaltuation and analysis
<p>
Next steps
<p>
   </td>
  </tr>
  <tr>
   <td>Amritesh
   </td>
   <td> 
<p> 
created all data visualization graphs
<p>
created data_visualization.pdf
   </td>
  </tr>
  <tr>
   <td>Luca
   </td>
   <td>
<p>
Implemented and trained LSTM
<p>
Integration of LSTM method with dataset
<p>
   </td>
  </tr>
  <tr>
   <td>Lior
   </td>
   <td>
<p>
Selected M1 and M2
<p>
database selection and adding features
<p> 
<p> 
feature reduction with PCA
<p>
   </td>
  </tr>
  <tr>
   <td>Vamsi
   </td>
   <td> 
<p> Implemented and trained LSTM
<p> created F1-score, precision & recall, and accuracy quantitative metrics for LSTM 
   </td>
  </tr>
</table>


## 8. Project Award Eligibility 

Opt-in

# Final Report: Predicting Stock Market Movements Using Machine Learning

## 1. Introduction/Background

Financial markets exhibit complex, dynamic behavior influenced by numerous factors such as economic indicators, investor sentiment, and technical trends. Traditional methods like fundamental and technical analysis have been widely used, but machine learning (ML) has emerged as a powerful alternative to improve predictive accuracy. Several studies have explored the use of ML techniques such as Support Vector Machines (SVM), Long Short-Term Memory (LSTM) networks, and ensemble methods to forecast stock prices or price movements.

For this project, we developed and evaluated several machine learning models to predict the daily directional movement (up/stay vs. down) of the Dow Jones Industrial Average (DJIA) using historical market data and sentiment analysis derived from news headlines. The goal was to determine if incorporating news sentiment alongside price data could improve prediction accuracy compared to simpler models or random chance.

### Literature Review

1.  **Patel et al. (2015)** applied ML techniques, including Random Forests and SVM, to predict stock trends, demonstrating improved accuracy over traditional methods. Their work highlighted the potential of combining different feature types.
2.  **Fischer and Krauss (2018)** employed LSTM networks for financial time series forecasting and found deep learning superior to shallow models, particularly for capturing temporal dependencies inherent in market data.
3.  **Chen et al. (2021)** explored ensemble learning and reinforcement learning for stock trading strategies, achieving competitive performance and showcasing the benefits of combining multiple models or learning approaches.
4.  **Sun, J. (2016)** provided the core dataset combining historical DJIA prices with daily news headlines, enabling the integration of textual sentiment features.
5.  **Valour, Maximus (2025)** demonstrated methods for sentiment analysis on stock market news, providing a basis for our feature engineering approach using VADER and TextBlob.

### Dataset Description

We utilized two primary databases sourced from Kaggle, originally collected and organized by Aaron Sun (see references [1]):

1.  **upload_DJIA_table.csv:** Contains daily Open, High, Low, Close (OHLC) prices, Volume, and Adjusted Close for the DJIA index.
2.  **Combined_News_DJIA.csv:** Contains the top 25 news headlines for each date, along with a binary label: '1' if the DJIA Adjusted Close rose or stayed the same compared to the previous day, and '0' if it decreased.

These two datasets were merged using an inner join on the 'Date' attribute, creating a unified dataset containing both price information and corresponding news headlines for each trading day.

## 2. Problem Definition

### Problem:

Stock market prediction, specifically forecasting the *direction* of price movement (classification), is inherently challenging due to market volatility, noise from numerous unpredictable events, and complex, often non-linear dependencies between influencing factors. The objective of this project is to classify whether the DJIA index will increase/stay neutral ('1') or decrease ('0') on a given day, based on historical price data and sentiment extracted from news headlines published *up to that day*.

### Motivation:

Developing models that can accurately predict stock market direction, even slightly better than chance, holds significant value for investors and financial institutions. It can inform trading strategies, aid in risk management, and potentially lead to improved investment returns. Machine learning offers a powerful toolkit to analyze vast amounts of data, uncover subtle patterns missed by traditional analysis, and potentially capture the complex interplay between market sentiment and price action.

## 3. Methods

### Data Preprocessing Methods:

1.  **Sentiment Extraction (VADER):**
    * For each date, the top 25 news headlines were concatenated into a single text string.
    * The VADER (Valence Aware Dictionary and sEntiment Reasoner) sentiment analysis tool was applied to this combined string.
    * VADER provides scores for positive, negative, and neutral sentiment, along with a compound score (a normalized, weighted composite). These four scores were added as features to our dataset for each date.
    * *Rationale:* VADER is specifically attuned to sentiments expressed in social media and short texts, making it suitable for news headlines. We hypothesized that the overall market sentiment, quantifiable through these scores, is a significant driver of short-term price movements.

2.  **Subjectivity/Polarity Rating (TextBlob):**
    * In addition to VADER, the TextBlob library was used to analyze the concatenated headlines for each date.
    * TextBlob calculates two metrics:
        * **Polarity:** A float within [-1.0, 1.0] indicating negative vs. positive sentiment.
        * **Subjectivity:** A float within [0.0, 1.0] where 0.0 is very objective and 1.0 is very subjective.
    * These two scores were also added as features to the dataset.
    * *Rationale:* Polarity provides an alternative measure of sentiment, while subjectivity helps gauge whether the news coverage is primarily factual or opinion-based. Highly subjective or opinionated news might influence investor behavior differently than objective reporting.

3.  **Feature Scaling:**
    * Numerical features (OHLC prices, Volume, VADER scores, TextBlob scores) were scaled using standardization (subtracting the mean and dividing by the standard deviation) to ensure that features with larger values did not disproportionately influence models sensitive to feature magnitude (like PCA, CNN, LSTM).

4.  **Data Splitting:**
    * The dataset was split chronologically into training and testing sets to simulate real-world prediction scenarios where future data is unknown. An 80% training and 20% testing split was used for model evaluation.

5.  **Principal Component Analysis (PCA):**
    * PCA was applied to the combined feature set (scaled price data and sentiment scores) *before* feeding data into the LSTM model, as explored during the midterm.
    * *Rationale:* The goal was to reduce dimensionality, remove multicollinearity between related features (e.g., Open, High, Low, Close prices; different sentiment scores), and potentially reduce noise, thereby improving the efficiency and generalization of the subsequent model. We aimed to retain approximately 95% of the variance.

### Machine Learning Models:

We implemented and evaluated three different machine learning models suitable for this classification task:

1.  **Random Forest (RF):** An ensemble method that builds multiple decision trees on different subsets of the data (using bootstrapping) and features, averaging their predictions. Known for its robustness and ability to handle non-linear relationships without extensive hyperparameter tuning. Requires careful feature engineering but lacks inherent temporal awareness.
2.  **1D Convolutional Neural Network (CNN):** A deep learning model often used for sequence data. 1D CNNs can learn spatial hierarchies of features within sequences, potentially identifying predictive patterns in recent price/sentiment history. It focuses on local patterns within the input sequence.
3.  **Long Short-Term Memory (LSTM) Network:** A type of Recurrent Neural Network (RNN) specifically designed to handle long-range dependencies in sequential data. LSTMs use memory cells and gates to control information flow, allowing them to remember relevant past information (like market trends or sustained sentiment) over extended periods.
    * *Input/Sequence Handling:* The model takes sequences of past data points as input (e.g., the last `N` days). Each data point in the sequence consists of features like scaled OHLC prices, volume, and sentiment scores, or the principal components derived from these features via PCA. To predict the market direction for day `T+1`, it processes the sequence from days `T-N+1` to `T`, updating its internal state at each step based on the information from that day. For predictions on the test set, a sliding window approach is used, ensuring the model only uses past information available up to day `T` to predict day `T+1`.

*Supervised Learning Models:* Random Forest, 1D CNN, LSTM
*Unsupervised Learning Technique:* PCA (used for feature engineering/dimensionality reduction prior to LSTM modeling)

## 4. Results and Discussion

### Quantitative Metrics:

The following metrics were used to evaluate the performance of our classification models on the held-out test set:

1.  **Accuracy:** The overall proportion of correct predictions (both 'Up' and 'Down'). $Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$
2.  **Precision:** The proportion of predicted 'Up' movements that were actually 'Up'. Measures the accuracy of positive predictions. $Precision = \frac{TP}{TP + FP}$
3.  **Recall (Sensitivity):** The proportion of actual 'Up' movements that were correctly identified. Measures how well the model finds positive cases. $Recall = \frac{TP}{TP + FN}$
4.  **F1-Score:** The harmonic mean of Precision and Recall, providing a single score that balances both metrics. $F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}$

*(Where TP = True Positives, TN = True Negatives, FP = False Positives, FN = False Negatives, with '1' (Up/Stay) as the positive class)*

### Model Implementation Details:

1.  **Random Forest:**
    * Features: Scaled OHLC, Volume, VADER scores (compound, positive, negative, neutral), TextBlob scores (polarity, subjectivity).
    * Bootstrapping: Enabled.
    * Number of Estimators (Trees): 300.
    * Minimum Samples per Leaf: 5.
    * Data Split: 80% training, 20% testing (chronological).

2.  **1D Convolutional Neural Network (CNN):**
    * Input Shape: Sequences of recent daily features (e.g., last 5 days features). Features included scaled OHLC, Volume, and sentiment scores.
    * Architecture:
        * `Conv1D` layer: 32 filters, kernel size of 3, ReLU activation.
        * `MaxPooling1D` layer: Pool size of 2.
        * `Dropout` layer: Rate p = 0.2.
        * `Flatten` layer.
        * `Dense` layer: Hidden layer (e.g., 32 units), ReLU activation.
        * `Dense` output layer: 1 unit, Sigmoid activation.
    * Optimizer: Adam.
    * Loss Function: Binary Cross-Entropy.
    * Hyperparameters: Epochs = 30, Batch Size = 32.
    * Data Split: 80% training, 20% testing (chronological, sequence structure maintained).

3.  **Long Short-Term Memory (LSTM) Network:**
    * Input Shape: Sequences of recent daily features (e.g., last 10 days). Features used were the principal components derived from PCA applied to scaled OHLC, Volume, and sentiment scores.
    * Architecture:
        * `LSTM` layer: 64 units.
        * `Dropout` layer: Rate p = 0.2.
        * `Dense` layer: 32 units, ReLU activation.
        * `Dense` output layer: 1 unit, Sigmoid activation.
    * Optimizer: Adam.
    * Loss Function: Binary Cross-Entropy.
    * Hyperparameters: Epochs = 20, Batch Size = 32.
    * Data Split: 80% training, 20% testing (chronological, sequence structure maintained).

### Data Visualization

**DJIA Candlestick Chart (2008-2016)**

`![image](https://github.gatech.edu/lmanor3/ML-Semester-Project/assets/91277/427aeabb-1bb4-4445-8107-94c1d37f834c)
`

*Description:* This plot shows the daily price movements (Open, High, Low, Close represented by candlesticks) and trading volume (bars below) for the Dow Jones Industrial Average (DJIA) from August 2008 to July 2016. It visualizes the significant market downturn in late 2008/early 2009 followed by a general upward trend through the rest of the period. Periods of increased volatility (longer candlesticks, higher volume) are also visible. This provides context for the market conditions during which our data was generated.

**PCA Scree Plot**

`[Image: PCA Scree Plot - image_3535fd.png]`

*Description:* The scree plot displays the proportion of variance explained by each principal component, ordered from highest to lowest. There is a sharp drop ('elbow') after the first component, indicating it captures a large majority of the variance (~90% based on the plot, though our previous calculation using all features yielded ~37% for PC1 - *clarification needed if PCA was run on price features only for this plot vs. all features for the model*). Subsequent components explain progressively smaller amounts of variance. This plot visually supports the idea that a small number of components can represent a large portion of the information in the original features, justifying the use of PCA for dimensionality reduction. Based on retaining ~95% variance across all features (price and sentiment), 6 components were selected for the LSTM input.

**PCA Biplot (PC1 vs PC2)**

`[Image: PCA Biplot - image_35318b.png]`

*Description:* This biplot visualizes the data points (observations) projected onto the first two principal components (PC1 and PC2), likely derived from price and volume features. The red vectors represent the original features, showing how they contribute to these components.
* Features like 'High', 'Low', 'Open', 'Close' (clustered tightly together) point strongly along the negative PC1 axis, indicating they are highly correlated and are the primary drivers of PC1, representing the general price level or trend.
* The 'Volume' feature points strongly along the positive PC2 axis, suggesting it captures variance largely independent of the price features in PC1.
* The spread of the blue observation points shows the distribution of the data in this reduced two-dimensional space, with most observations clustered at lower PC2 values.

### Principal Component Analysis (PCA) Results:

PCA was applied to the combined feature set (scaled price data *and* sentiment scores) prior to LSTM modeling to reduce dimensionality.

* **Number of principal components selected:** 6
* **Total variance explained by 6 components:** ~95.7%
* **Component Breakdown & Analysis:**
    * **PC1 (37.4% variance):** Dominated by price features (Low, Open, High). Captures general market level/trend.
    * **PC2 (18.9% variance):** Dominated by VADER sentiment (negative, neutral, compound). Captures primary sentiment dimension.
    * **PC3 (15.8% variance):** Dominated by VADER positive, VADER neutral, TextBlob polarity. Captures another aspect of sentiment, possibly positivity intensity.
    * **PC4 (9.5% variance):** Dominated by TextBlob subjectivity. Captures the opinion vs. factual nature of news.
    * **PC5 (8.5% variance):** Dominated by TextBlob polarity and VADER compound. Captures interaction between polarity and overall VADER score.
    * **PC6 (5.5% variance):** Dominated by VADER positive and compound. Captures further nuance in positive sentiment.
* *Conclusion:* PCA successfully reduced the feature space while retaining most information. The components clearly separate price dynamics (PC1) from various dimensions of sentiment and text characteristics (PC2-PC6), confirming their contribution to data variance and providing a potentially less noisy, decorrelated input set for the LSTM.

### Model Performance on Test Set:

Based on our experiments, the following performance metrics were achieved on the held-out test set:

1.  **Random Forest:**
    * Accuracy: 49%
    * Precision: `[Insert Precision Score]`
    * Recall: `[Insert Recall Score]`
    * F1-Score: 0.52

2.  **1D CNN:**
    * Accuracy: 56%
    * Precision: `[Insert Precision Score]`
    * Recall: `[Insert Recall Score]`
    * F1-Score: 0.65

3.  **LSTM:**
    * Accuracy: 53%
    * Precision: `[Insert Precision Score]`
    * Recall: `[Insert Recall Score]`
    * F1-Score: 0.65

*(Note: Precision and Recall scores need to be added if available.)*

### Model Comparison Summary

The table below summarizes the key characteristics and performance of the three models evaluated:

| Model         | Type          | Temporal Awareness | Needs Feature Engineering | Pros                           | Cons                               | Accuracy Achieved | F1 Score |
| :------------ | :------------ | :----------------- | :-------------------------- | :----------------------------- | :--------------------------------- | :---------------- | :------- |
| **LSTM** | Recurrent     | Yes                | No                          | Long-term dependencies         | Slower to train, harder to tune    | 53%               | 0.65     |
| **RandomForest**| Tree Ensemble | No                 | Yes                         | Interpretable, robust          | Doesn't model time directly        | 49%               | 0.52     |
| **1D CNN** | Deep Learning | Local              | Some                        | Fast, good at pattern detection| Less interpretable                 | 56%               | 0.65     |

### Discussion:

The results indicate that both the 1D CNN and LSTM models achieved the highest performance in terms of F1-Score (0.65), with the 1D CNN also achieving the best accuracy (56%). The Random Forest model performed noticeably worse, suggesting it struggled to capture the predictive signals as effectively as the deep learning approaches for this specific task.

* **1D CNN:** Its success suggests that local patterns in the sequence of recent price and sentiment data are highly informative. Its relative speed and strong pattern detection capabilities proved advantageous.
* **LSTM:** Achieved a comparable F1-Score to the CNN, demonstrating its strength in modeling temporal dependencies, potentially longer-range ones than the CNN. Its explicit design for sequences and ability to handle inputs without extensive feature engineering (especially post-PCA) are advantages, though training complexity is higher.
* **Random Forest:** While robust and interpretable, its lack of inherent temporal awareness seems to be a limitation for this time-series forecasting problem. Its performance indicates that treating each day's features independently is less effective than methods designed for sequential data.

Comparing the CNN and LSTM, the trade-off appears to be between the CNN's potentially better capture of local patterns (leading to slightly higher accuracy here) and the LSTM's theoretical strength in longer dependencies, both yielding the same F1-Score.

The inclusion of sentiment features alongside price data appears beneficial, as suggested by the PCA results and the performance of the deep learning models. However, even the best models achieved accuracy levels (53-56%) that, while better than random chance (50%), highlight the inherent difficulty of consistently predicting stock market movements.

**Applicability and Limitations:**
* **Prediction Timing:** A key limitation is that the current models use information (news headlines) available *during* day `T` to predict the closing price movement *at the end* of day `T` relative to day `T-1`. For a practical trading strategy, the prediction needs to be made *before* the event occurs. Adapting the models (e.g., predicting day `T+1` using only data up to day `T`, using intraday data, or carefully lagging features) would be necessary to make them actionable for trading. Implementing and backtesting such a strategy involves further complexities (transaction costs, slippage) beyond this project's scope.
* **Target Variable:** The binary classification (up/stay vs. down) simplifies market dynamics.
* **Sentiment Scope:** Sentiment is derived only from the top 25 headlines.
* **Market Regimes:** Models were trained on a specific historical period and may not generalize perfectly to different market conditions.
* **Interpretability:** Deep learning models (CNN, LSTM) are less interpretable than Random Forest.

**Future Work:**
* Implement and test adaptations for an actionable trading strategy (e.g., predicting T+1).
* Incorporate additional features (macroeconomic, technical indicators).
* Explore more advanced sentiment analysis techniques.
* Conduct more extensive hyperparameter tuning.
* Combine models using ensemble techniques.
* Test models on different indices or individual stocks.

## 5. References

1.  Sun, J. (2016, August). Daily News for Stock Market Prediction, Version 1. Retrieved 3/20/2025 from https://www.kaggle.com/aaron7sun/stocknews.
2.  Valour, Maximus. (2025, January). Stock Market Sentiment Analysis, Version 1. Retrieved 3/20/2025 from https://www.kaggle.com/code/maximusvalor/stock-market-sentiment-analysis.
3.  Patel, J., Shah, S., Thakkar, P., & Kotecha, K. (2015). Predicting stock market index using fusion of machine learning techniques. *Expert Systems with Applications*, 42(4), 2162-2172.
4.  Fischer, T., & Krauss, C. (2018). Deep learning with long short-term memory networks for financial market predictions. *European Journal of Operational Research*, 270(2), 654-669.
5.  Chen, W., Zhang, H., Zhang, W., & He, X. (2021). A survey on stock market prediction using machine learning and statistical techniques. *IEEE Access*, 9, 15039-15062.
6.  VADER Sentiment Analysis: Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. Eighth International Conference on Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014.
7.  TextBlob Documentation: https://textblob.readthedocs.io/

## 6. Gantt Chart

*(Refers to original project plan - update link if necessary or remove section)*
https://gtvault-my.sharepoint.com/:x:/g/personal/lmanor3_gatech_edu/EQEjpjbv5GhMqv4rImskUEQBMC-x1_dlydS-XmuTr956hQ?e=s5Yhoi

## 7. Contributions Table

*(Update with actual contributions post-midterm)*

| Member     | Post-Midterm Contributions                                                                                                |
| :--------- | :------------------------------------------------------------------------------------------------------------------------ |
| **Munim** | <Fill>|
| **Amritesh** | Write detailed report and verified alll the pre processing data is clean and run the analysis.|
| **Luca** |<Fill> |
| **Lior** | <Fill>        |
| **Vamsi** | <Fill>|

## 8. Project Award Eligibility

Opt-in

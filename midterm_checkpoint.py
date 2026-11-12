import pandas as pd
import numpy as np
from textblob import TextBlob as tb
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as sent
import matplotlib.pyplot as plt

## Imported PCA class from pca.py
from pca import PCA as pca

## LSTM implementation
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler


## Create pandas databases
database1 = pd.read_csv("Combined_News_DJIA.csv")
print(database1.head())
database2 = pd.read_csv("Upload_DJIA_table.csv")
print(database2.head())

## Merge the databases
merged_database = database1.merge(database2, how='inner', on='Date')
print(merged_database.head())

## Append subjectivity and polarity to the merged database
headlines = [" ".join(str(x) for x in merged_database.iloc[row, 2:27]) for row in range(len(merged_database))]

subjectivities = []
polarities = []
for text in headlines:
    subjectivities.append(tb(text).sentiment.subjectivity)
    polarities.append(tb(text).sentiment.polarity)
merged_database['subjectivity'] = subjectivities
merged_database['polarity'] = polarities

## Append sentiment scores (compound, negative, positive, neutral)
analyzer = sent()  # Create an instance of SentimentIntensityAnalyzer
sentiments = [analyzer.polarity_scores(text) for text in headlines]
merged_database['compound'] = [s['compound'] for s in sentiments]
merged_database['negative'] = [s['neg'] for s in sentiments]
merged_database['positive'] = [s['pos'] for s in sentiments]
merged_database['neutral']  = [s['neu'] for s in sentiments]

print(merged_database.head())

## Separate labels from database and store into y
y = merged_database['Label']
merged_database = merged_database.drop('Label', axis=1)

## Select features for the model
X = merged_database[['Open', 'High', 'Low', 'Volume',
                     'subjectivity', 'polarity',
                     'compound', 'negative', 'positive', 'neutral']]


scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)  # Standardize each feature
pca_model = pca()
pca_model.fit(X_standardized)
# Then either transform using:
#X_reduced = pca_model.transform(X_standardized, K=1)
# or use your variance threshold function:
X_reduced = pca_model.transform_rv(X_standardized, retained_variance=0.95)

print("Shape after PCA reduction:", X_reduced.shape)

# ----------------------------------------------------------------
# Print PCA info (explained variance & top features)
# ----------------------------------------------------------------

## Reduce features while maintaining 95% of variance
X_reduced_rv = pca_model.transform(X, K=8)
print("Shape after PCA reduction:", X_reduced_rv.shape)

## Scale the reduced features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_reduced)

## Reshape for LSTM: (samples, timesteps, features)
X_lstm = np.reshape(X_scaled, (X_scaled.shape[0], 1, X_scaled.shape[1]))

## Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_lstm, y, test_size=0.2, random_state=42
)

## Define the LSTM model
model = Sequential()
model.add(LSTM(64, input_shape=(1, X_scaled.shape[1]), return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # binary classification

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

## Train the model
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

## Evaluate the model
y_pred = model.predict(X_test)
y_pred_class = (y_pred > 0.5).astype(int)

acc_value = accuracy_score(y_test, y_pred_class)
print("Accuracy:", acc_value)
print("Classification Report:\n", classification_report(y_test, y_pred_class))

## Calculate and print additional metrics
f1 = f1_score(y_test, y_pred_class)
precision = precision_score(y_test, y_pred_class)
recall = recall_score(y_test, y_pred_class)

print("F1 Score:", f1)
print("Precision:", precision)
print("Recall:", recall)

## Plot training history
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("Training History")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()


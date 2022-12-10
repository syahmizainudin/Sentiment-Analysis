# %%
from tensorflow import keras
from keras.utils import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras import Sequential
from keras.layers import LSTM, Dense, Bidirectional
from keras.layers import Embedding
from keras.callbacks import TensorBoard
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import re, os, datetime, json, pickle

DATASET_URL = 'https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv'

# %% Step 1. Data Loading
df = pd.read_csv(DATASET_URL)

# %% 2. Data Inspection
df.head(10)
df.tail(10)

df.info()
df.describe()

df['sentiment'].value_counts() # Check if the target is balanced
df.isna().sum() # No NaN is found
df.duplicated().sum() # 418 duplicated data need to be remove

# %% 3. Data Cleaning
features = df['review']
targets = df['sentiment']

print(features[10]) # HTML Tags <br /><br /> need to be remove

for idx,rev in enumerate(features):
    features[idx] = re.sub('<.*?>', ' ', rev)
    features[idx] = re.sub('[^a-zA-Z]', ' ',  features[idx]).lower()

# Remove duplicated data
df_drop = pd.DataFrame([features, targets])
df_drop = df.drop_duplicates()
df_drop.duplicated().sum()

# %% 4. Features Selection
# Define features and labels
features = df_drop['review']
targets = df_drop['sentiment']

# %% 5. Data Pre-processing
# Target preprocessing
ohe = OneHotEncoder(sparse=False)
train_sentiment = np.expand_dims(features, -1)
train_sentiment = ohe.fit_transform(train_sentiment)

# Features preprocessing
# Tokenization to convert text into numbers
num_words = 5000
oov_token = '<OOV>'

tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)
tokenizer.fit_on_texts(features)

word_index = tokenizer.word_index
train_sequences = tokenizer.texts_to_sequences(features)

# Padding + truncating
train_sequences = pad_sequences(train_sequences, maxlen=200, padding='post', truncating='post')

# Train-test split
train_sequences = np.expand_dims(train_sequences, -1)
X_train, X_test, y_train, y_test = train_test_split(train_sequences, train_sentiment, random_state=123)

# %% Model Development
embedding_size = 64

model = Sequential()
model.add(Embedding(num_words, embedding_size))
model.add(LSTM(embedding_size, return_sequences=True))
model.add(LSTM(64))
model.add(Dense(y_train.shape[1], activation='softmax'))
model.summary()

# %% Model compilation
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='acc')

# Define callbacks
LOG_DIR = os.path.join(os.getcwd(), 'logs', datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
tb = TensorBoard(log_dir=LOG_DIR)

# Model training
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, callbacks=tb)

# %% Model Evaluation
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

print('Confusion Matrix:\n', confusion_matrix(y_true, y_pred))
print('Classification Report:\n', classification_report(y_true, y_pred))
print('Accuracy:\n', accuracy_score(y_true, y_pred))

# %% Saving models
# Save tokenizer
with open('tokenizer.json', 'w') as f:
    json.dump(tokenizer.to_json(), f)

# Save encoder
with open('ohe.pkl', 'wb') as f:
    pickle.dump(ohe, f)

# Save model
model.save('sentiment-analysis.h5')

# %%

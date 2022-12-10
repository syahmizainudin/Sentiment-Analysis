# %%
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import json, pickle

# %%
# load trained DL model
loaded_model = load_model('sentiment-analysis.h5')

# %%
loaded_model.summary()

# %% load tokenizer
with open('tokenizer.json', 'r') as f:
    loaded_tokenizer = json.load(f)

loaded_tokenizer = tokenizer_from_json(loaded_tokenizer)

# %%
# %% load ohe
with open('ohe.pkl', 'rb') as f:
    loaded_ohe = pickle.load(f)

# %% Time to deploy
test_review = ["I crashed my car after watching the movie from remembering the plot. 10/10"]
test_review = loaded_tokenizer.texts_to_sequences(test_review)

test_review = pad_sequences(test_review, 200, padding='post', truncating='post')

y_pred = np.argmax(loaded_model.predict(test_review))

if y_pred == 1:
    out = [0,1]
else:
    out = [1,0]

print(loaded_ohe.inverse_transform([out]))


# %%

import pandas as pd
import os
CSV_Path = os.path.join('Copy The Storm Is Here.csv')
df = pd.read_csv(CSV_Path, nrows = 15600) # starts 11/19 at 8kun
df.columns = ["Date", "Text1", "Text2"]

df['Date'].fillna(method='ffill', inplace=True)
df = df.dropna()

arr = df.iloc[:,1].to_numpy()
dates = df.iloc[:,0].to_numpy()
dates = dates.tolist()[0:15600]
sentences = arr.tolist()[0:15600]

# print(len(sentences))
# print(len(dates))
# j = 0
# for i in sentences:
#     if(len(i) > 100 or len(i) < 10):
#         sentences.remove(i)
#         dates.pop(j)
#     j = j + 1
# print(len(sentences))
# print(len(dates))

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words=1600, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, padding = "post", truncating="post", maxlen=10)

reverse_word_index = dict([(value,key) for (key,value) in word_index.items()])
def decode(text):
    return " ".join([reverse_word_index.get(i, "_") for i in text])
# print(decode(sequences[2]))

out = pd.DataFrame(columns=('Dates', 'Drop'))

print(word_index)
counter = 0
for i in range(len(padded)):
    for num in padded[i]:
        if (num == 298):
            counter = counter + 1
            print("___")
            print(padded[i])
            #print(decode(padded[i]))
            out.loc[i] = [dates[i],sentences[i]]

print(out)
print(counter)


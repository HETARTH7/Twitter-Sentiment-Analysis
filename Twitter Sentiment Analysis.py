# %%
import numpy as np 
import pandas as pd
import re

# %%
train_ds=pd.read_csv("./twitter_training.csv", header=None)
train_ds

# %%
val_ds=pd.read_csv("./twitter_validation.csv", header=None)
val_ds

# %%
def preprocess_text(text):
    text = re.sub(r'[^A-Za-z0-9 ]+', '', text)
    text = str(text.lower())

    return text

# %%
train_ds['processed']=train_ds[3].apply(lambda x:preprocess_text(str(x)))
train_ds.head()

# %%
val_ds['processed']=val_ds[3].apply(lambda x:preprocess_text(str(x)))
val_ds.head()

# %%
from wordcloud import WordCloud
import matplotlib.pyplot as plt

corpus=''.join(train_ds['processed'])
wordcloud = WordCloud(width=800, height=400, random_state=21,
                      max_font_size=110, background_color='black').generate(corpus)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.title('Word Cloud of processed training data')
plt.axis('off')
plt.show()

# %%
from wordcloud import WordCloud
import matplotlib.pyplot as plt

corpus=''.join(val_ds['processed'])
wordcloud = WordCloud(width=800, height=400, random_state=21,
                      max_font_size=110, background_color='black').generate(corpus)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.title('Word Cloud of processed validation data')
plt.axis('off')
plt.show()

# %%
def tokenize_and_remove_stopwords(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    return ' '.join(tokens)

# %%
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

train_ds['tokenized_text'] = train_ds['processed'].apply(
    tokenize_and_remove_stopwords)

val_ds['tokenized_text'] = val_ds['processed'].apply(
    tokenize_and_remove_stopwords)

# %%
train_ds.head()

# %%
val_ds.head()

# %%
from sklearn.feature_extraction.text import CountVectorizer

stop_words = stopwords.words('english')
cv=CountVectorizer(tokenizer=word_tokenize,stop_words=stop_words,ngram_range=(1,1))

# %%
X,y=cv.fit_transform(train_ds['tokenized_text']),train_ds[2]

# %%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

lr_classifier = LogisticRegression(max_iter=500)
lr_classifier.fit(X_train, y_train)

lr_pred = lr_classifier.predict(X_test)
print("Logistic Regression Accuracy: ", accuracy_score(y_test, lr_pred) * 100)

# %%
from sklearn.svm import SVC

svm_classifier = SVC(kernel='linear', random_state=0)
svm_classifier.fit(X_train, y_train)

svm_pred = svm_classifier.predict(X_test)
print("SVM Accuracy: ", accuracy_score(y_test, svm_pred) * 100)

# %%
from sklearn.tree import DecisionTreeClassifier

dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(X_train, y_train)

dt_pred = dt_classifier.predict(X_test)
print("Decision Tree Accuracy: ", accuracy_score(y_test, dt_pred) * 100)

# %%
from sklearn.ensemble import RandomForestClassifier

rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train, y_train)

rf_pred = rf_classifier.predict(X_test)
print("Random Forest Accuracy: ", accuracy_score(y_test, rf_pred) * 100)

# %%




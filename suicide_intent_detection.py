from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import pandas as pd
import numpy as np

happy = pd.read_csv('data/happy.csv')
emotions = pd.read_csv('data/tweet_emotions.csv')
suicide = pd.read_csv('data/twitter-suicidal_data.csv')

data = []

happy_data = happy[['cleaned_hm']].rename(
    columns={'cleaned_hm': 'text'}).head(1000)
happy_data['class'] = 'Happy Intent'
data.append(happy_data)

sad_emotions_data = emotions[emotions['sentiment'] == 'sadness'][[
    'content']].rename(columns={'content': 'text'}).head(1000)
sad_emotions_data['class'] = 'Sad Intent'
data.append(sad_emotions_data)

neutral_emotions_data = emotions[emotions['sentiment'].isin(
    ['empty', 'neutral'])][['content']].rename(columns={'content': 'text'}).head(1000)
neutral_emotions_data['class'] = 'Normal Intent'
data.append(neutral_emotions_data)

suicidal_intent_data = suicide[suicide['intention'] == 1][[
    'tweet']].rename(columns={'tweet': 'text'}).head(1000)
suicidal_intent_data['class'] = 'Suicidal Intent'
data.append(suicidal_intent_data)

dataset = pd.concat(data, ignore_index=True)
# dataset.to_csv('suicide_intent_dataset.csv')

# dataset=pd.read_csv('suicide_intent_dataset.csv')
dataset

class_counts = dataset['class'].value_counts()
print(class_counts)


# nltk.download('stopwords')
# nltk.download('punkt')


def preprocess_text(text):
    text = re.sub(r'http\S+', '', text)
    text = ''.join([char for char in text if ord(char) < 128])
    text = text.lower()

    return text


def tokenize_and_remove_stopwords(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    return ' '.join(tokens)


dataset['cleaned_text'] = dataset['text'].apply(preprocess_text)
dataset['tokenized_text'] = dataset['cleaned_text'].apply(
    tokenize_and_remove_stopwords)
dataset

corpus = dataset['tokenized_text']


all_words = ' '.join(corpus)
wordcloud = WordCloud(width=800, height=400, random_state=21,
                      max_font_size=110, background_color='white').generate(all_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.title('Word Cloud of Tokenized Text')
plt.axis('off')
plt.show()

cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()
y = dataset['class'].values

words = cv.get_feature_names_out()
word_frequencies = X.sum(axis=0)
df_word_freq = pd.DataFrame({'Word': words, 'Frequency': word_frequencies})
df_word_freq = df_word_freq.sort_values(by='Frequency', ascending=False)

top_n = 50
plt.figure(figsize=(12, 6))
plt.bar(df_word_freq['Word'][:top_n],
        df_word_freq['Frequency'][:top_n], color='skyblue')
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.title(f'Top {top_n} Word Frequencies')
plt.xticks(rotation=45, ha='right')
plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    dataset['tokenized_text'], y, test_size=0.2, random_state=42)

tfidf_vectorizer = TfidfVectorizer(max_features=1000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

lr_classifier = LogisticRegression()
lr_classifier.fit(X_train_tfidf, y_train)
lr_pred = lr_classifier.predict(X_test_tfidf)

dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(X_train_tfidf, y_train)
dt_pred = dt_classifier.predict(X_test_tfidf)

rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train_tfidf, y_train)
rf_pred = rf_classifier.predict(X_test_tfidf)

svm_classifier = SVC(kernel='linear', random_state=0)
svm_classifier.fit(X_train_tfidf, y_train)
svm_pred = svm_classifier.predict(X_test_tfidf)

cm = confusion_matrix(y_test, svm_pred)
print('Confusion Matrix:')
print(cm)
print(f'Accuracy:', accuracy_score(y_test, svm_pred)*100, '%')

plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()

classes = dataset['class'].unique()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

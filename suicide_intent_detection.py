from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import nltk
import re
import pandas as pd
import numpy as np

happy = pd.read_csv('happy.csv')
emotions = pd.read_csv('tweet_emotions.csv')
suicide = pd.read_csv('Suicide_Detection.csv')

data = []

happy_data = happy[['cleaned_hm']].rename(
    columns={'cleaned_hm': 'text'}).head(1000)
happy_data['class'] = 'happy'
data.append(happy_data)

sad_emotions_data = emotions[emotions['sentiment'] == 'sadness'][[
    'content']].rename(columns={'content': 'text'}).head(1000)
sad_emotions_data['class'] = 'sad'
data.append(sad_emotions_data)

neutral_emotions_data = emotions[emotions['sentiment'].isin(
    ['empty', 'neutral'])][['content']].rename(columns={'content': 'text'}).head(1000)
neutral_emotions_data['class'] = 'normal'
data.append(neutral_emotions_data)

suicide_data = suicide[suicide['class'] == 'suicide'][[
    'text']].rename(columns={'text': 'text'}).head(1000)
suicide_data['class'] = 'suicide'
data.append(suicide_data)

dataset = pd.concat(data, ignore_index=True)
dataset.to_csv('suicide_intent_dataset.csv', index=False)
dataset = pd.read_csv('suicide_intent_dataset.csv')
class_counts = dataset['class'].value_counts()

# nltk.download('stopwords')
corpus = []
for i in range(0, 4000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['text'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    review = [ps.stem(word)
              for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    corpus.append(review)

cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()
y = dataset['class'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

classifier = GaussianNB()
classifier.fit(X_train, y_train)

classifier = SVC(kernel='linear', random_state=0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)

import os
from google.cloud import dialogflow_v2
import uuid
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

happy = pd.read_csv('happy.csv')
emotions = pd.read_csv('tweet_emotions.csv')
suicide = pd.read_csv('twitter-suicidal_data.csv')

data = []

happy_data = happy[['cleaned_hm']].rename(
    columns={'cleaned_hm': 'text'}).head(100)
happy_data['class'] = 'Happy Intent'
data.append(happy_data)

sad_emotions_data = emotions[emotions['sentiment'] == 'sadness'][[
    'content']].rename(columns={'content': 'text'}).head(100)
sad_emotions_data['class'] = 'Sad Intent'
data.append(sad_emotions_data)

neutral_emotions_data = emotions[emotions['sentiment'].isin(
    ['empty', 'neutral'])][['content']].rename(columns={'content': 'text'}).head(100)
neutral_emotions_data['class'] = 'Normal Intent'
data.append(neutral_emotions_data)

suicidal_intent_data = suicide[suicide['intention'] == 1][[
    'tweet']].rename(columns={'tweet': 'text'}).head(400)
suicidal_intent_data['class'] = 'Suicidal Intent'
data.append(suicidal_intent_data)

dataset = pd.concat(data, ignore_index=True)

# dataset=pd.read_csv('suicide_intent_dataset.csv')
dataset

rows_to_remove = dataset[dataset['text'].apply(len) > 256].index
dataset = dataset.drop(rows_to_remove, axis=0).reset_index(drop=True)
print("Updated dataset size:", dataset.shape[0])

# c=0
# for i in range(dataset['text'].size):
#     if(len(dataset['text'][i])>256):
#         c=c+1

# print(c)
# print(dataset['text'].size)

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
plt.axis('off')
plt.show()

cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()
y = dataset['class'].values

X_train, X_test, y_train, y_test = train_test_split(
    dataset['tokenized_text'], y, test_size=0.2, random_state=42)

tfidf_vectorizer = TfidfVectorizer(max_features=1500)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

lr_classifier = LogisticRegression()
lr_classifier.fit(X_train_tfidf, y_train)

dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(X_train_tfidf, y_train)

rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train_tfidf, y_train)

svm_classifier = SVC(kernel='linear', random_state=0)
svm_classifier.fit(X_train_tfidf, y_train)

y_pred = svm_classifier.predict(X_test_tfidf)
# print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

cm = confusion_matrix(y_test, y_pred)
print(cm)
print(f'Accuracy:', accuracy_score(y_test, y_pred)*100, '%')

df_x = np.array(dataset['text'])
dfx_train, dfx_test, dfy_train, dfy_test = train_test_split(
    df_x, y, test_size=0.2, random_state=0)
train_data = pd.DataFrame({'text': dfx_train, 'class': dfy_train})
# train_data.to_csv('dialogflow_training_dataset.csv', index=False)

print(dfx_test.size)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:/Users/HETARTH RAVAL/Desktop/AI/Machine Learning/projects/suicide-intent-detection-d40b46ea43fa.json"


def detect_intent(text, project_id, session_id, language_code):
    session_client = dialogflow_v2.SessionsClient()
    session = session_client.session_path(project_id, session_id)

    text_input = dialogflow_v2.TextInput(
        text=text, language_code=language_code)
    query_input = dialogflow_v2.QueryInput(text=text_input)

    response = session_client.detect_intent(
        session=session, query_input=query_input
    )

    intent = response.query_result.intent.display_name
    return intent


project_id = "suicide-intent-detection"
session_id = str(uuid.uuid4())
language_code = "en"

predicted_intents = []
for i in range(len(dfx_test)):
    text = dfx_test[i]
    predicted_intent = detect_intent(
        text, project_id, session_id, language_code)
    predicted_intents.append(predicted_intent)
predicted_intents = np.array(predicted_intents)

print(f'Accuracy:', accuracy_score(dfy_test, predicted_intents)*100, '%')

# print(np.concatenate((dfy_test.reshape(len(dfy_test),1), predicted_intents.reshape(len(predicted_intents),1)),1))
# print(np.concatenate((dfx_test.reshape(len(dfx_test),1), dfy_test.reshape(len(dfy_test),1)),1))

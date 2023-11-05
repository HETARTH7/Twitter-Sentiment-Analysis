import os
from google.cloud import dialogflow_v2
import uuid
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

suicide_data = suicide[suicide['class'] == 'suicide'][[
    'text']].rename(columns={'text': 'text'}).head(100)
suicide_data['class'] = 'Suicide Intent'
data.append(suicide_data)

dataset = pd.concat(data, ignore_index=True)
# dataset.to_csv('cleaned_suicide_intent_dataset.csv', index=False)

# dataset=pd.read_csv('suicide_intent_dataset.csv')
# dataset

class_counts = dataset['class'].value_counts()
print(class_counts)

# nltk.download('stopwords')
corpus = []
for i in range(0, 400):
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

print(corpus)

cv = CountVectorizer()
x = np.array(corpus)
X = cv.fit_transform(corpus).toarray()
y = dataset['class'].values

X_train, X_test, Y_train, Y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0)

classifier = GaussianNB()
classifier.fit(X_train, Y_train)

classifier = SVC(kernel='linear', random_state=0)
classifier.fit(X_train, Y_train)

y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), Y_test.reshape(len(Y_test), 1)), 1))

cm = confusion_matrix(Y_test, y_pred)
print(cm)
print(f'Accuracy:', accuracy_score(Y_test, y_pred)*100, '%')

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
for i in range(0, 80):
    text = x_test[i]
    text_chunks = [text[i:i+256] for i in range(0, len(text), 256)]
    predicted_intents_for_text = []

    for chunk in text_chunks:
        predicted_intent = detect_intent(
            chunk, project_id, session_id, language_code)
        predicted_intents_for_text.append(predicted_intent)
    combined_intent = " ".join(predicted_intents_for_text)
    predicted_intents.append(combined_intent)

predicted_intents = np.array(predicted_intents)
predicted_intents.size

print(f'Accuracy:', accuracy_score(y_test, predicted_intents)*100, '%')

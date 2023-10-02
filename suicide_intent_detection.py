import pandas as pd
import numpy as np

happy = pd.read_csv('happy.csv')
emotions = pd.read_csv('tweet_emotions.csv')
suicide = pd.read_csv('Suicide_Detection.csv')

data = []

happy_data = happy[['cleaned_hm']].rename(
    columns={'cleaned_hm': 'text'}).head(10000)
happy_data['class'] = 'happy'
data.append(happy_data)

sad_emotions_data = emotions[emotions['sentiment'] == 'sadness'][[
    'content']].rename(columns={'content': 'text'})
sad_emotions_data['class'] = 'sad'
data.append(sad_emotions_data)

neutral_emotions_data = emotions[emotions['sentiment'].isin(
    ['empty', 'neutral'])][['content']].rename(columns={'content': 'text'})
neutral_emotions_data['class'] = 'normal'
data.append(neutral_emotions_data)

suicide_data = suicide[suicide['class'] == 'suicide'][[
    'text']].rename(columns={'text': 'text'}).head(10000)
suicide_data['class'] = 'suicide'
data.append(suicide_data)

dataset = pd.concat(data, ignore_index=True)
dataset.to_csv('suicide_intent_dataset.csv', index=False)

dataset = pd.read_csv('suicide_intent_dataset.csv')

class_counts = dataset['class'].value_counts()

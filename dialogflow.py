import uuid
from google.cloud import dialogflow_v2
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ""


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
testing_data = [
    {"text": "Hello", "intent": "Default Welcome Intent"},
    {"text": "I'm feeling sad", "intent": "Suicide Intent"},
]

predicted_intents = []

for item in testing_data:
    text = item["text"]
    actual_intent = item["intent"]
    predicted_intent = detect_intent(
        text, project_id, session_id, language_code)
    predicted_intents.append(predicted_intent)
    print(predicted_intent)
correct_predictions = sum(1 for actual, predicted in zip(
    testing_data, predicted_intents) if actual == predicted)
accuracy = correct_predictions / len(testing_data) * 100

print(f"Accuracy: {accuracy}%")

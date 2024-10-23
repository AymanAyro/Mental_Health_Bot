import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
import torch
import os
import gdown
import json
import mlflow
import mlflow.pytorch
import secrets

# Azure Text Analytics credentials
key = "3d2b9f93f2514509acf274e8dc77f04e"
endpoint = "https://sent-anlysis.cognitiveservices.azure.com/"
text_analytics_client = TextAnalyticsClient(endpoint=endpoint, credential=AzureKeyCredential(key))

# Paths for saving the model and tokenizer
model_path = './saved_model'
tokenizer_path = './saved_tokenizer'

# Google Drive URLs for individual model and tokenizer files
MODEL_FILES = {
    'model.safetensors': 'https://drive.google.com/uc?id=1CRKCU_UjfjnLm_jANuLjRc3eZCUDIBOZ',
    'config.json': 'https://drive.google.com/uc?id=1MhmKb1_MHiDpAZ-cdVoIN2ymXEN2x99O'
}
TOKENIZER_FILES = {
    'tokenizer_config.json': 'https://drive.google.com/uc?id=1XXohv9gm1GuqWMI2WYkwAZD1_Xb8CGFm',
    'vocab.txt': 'https://drive.google.com/uc?id=1LuWjmJHwkiY8hASpaz7BCbabE066Xr9j',
    'special_tokens_map.json': 'https://drive.google.com/uc?id=1Zy5j5zyKP2W-xRS6rzaJS4keUNn_SMNC'
}


# Function to download files from Google Drive
def download_file_from_drive(drive_url, output_path):
    gdown.download(drive_url, output_path, quiet=False)

# Download model files if not present locally
if not os.path.exists(model_path):
    os.makedirs(model_path, exist_ok=True)
    for filename, url in MODEL_FILES.items():
        download_file_from_drive(url, os.path.join(model_path, filename))

# Download tokenizer files if not present locally
if not os.path.exists(tokenizer_path):
    os.makedirs(tokenizer_path, exist_ok=True)
    for filename, url in TOKENIZER_FILES.items():
        download_file_from_drive(url, os.path.join(tokenizer_path, filename))

# Load pre-trained model and tokenizer
tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load intents.json
with open('intents.json', 'r') as f:
    intents = json.load(f)

# Prepare label mapping
label_to_id = {intent['tag']: idx for idx, intent in enumerate(intents['intents'])}
id_to_label = {idx: intent['tag'] for idx, intent in enumerate(intents['intents'])}

# Function for sentiment analysis using Azure Text Analytics
def analyze_sentiment(text):
    response = text_analytics_client.analyze_sentiment(documents=[text])[0]
    return response.sentiment, response.confidence_scores

# Function for intent prediction using the pre-trained model
def predict_intent(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = outputs.logits.softmax(dim=1)
    predicted_label = probs.argmax().item()
    predicted_intent = id_to_label[predicted_label]
    return predicted_intent

# Function to get bot response based on intent and sentiment
def get_response(intent, sentiment):
    for intent_data in intents['intents']:
        if intent_data['tag'] == intent:
            return f"{intent_data['responses'][0]} (Sentiment: {sentiment})"
    return "Sorry, I don't understand."

# Initialize Streamlit app
st.title("Mental Health Chatbot")
st.write("A chatbot powered by Azure Text Analytics and a BERT model.")

# Session state to store conversation history
if 'conversation_history' not in st.session_state:
    st.session_state['conversation_history'] = []

# User input
user_message = st.text_input("You:", "")

if user_message:
    # Sentiment analysis
    sentiment, confidence_scores = analyze_sentiment(user_message)

    # Intent prediction
    intent = predict_intent(user_message)

    # Get bot response
    response = get_response(intent, sentiment)

    # Append conversation to history
    st.session_state['conversation_history'].append({'user': user_message, 'bot': response})

    # Log with MLflow
    experiment_name = 'Mental Health Chatbot'
    if not mlflow.get_experiment_by_name(experiment_name):
        mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)

    mlflow.autolog()
    with mlflow.start_run():
        mlflow.log_param("input_text", user_message)
        mlflow.log_param("predicted_intent", intent)
        mlflow.log_param("sentiment", sentiment)
        mlflow.pytorch.log_model(model, "bert_model")

# Display conversation history
for chat in st.session_state['conversation_history']:
    st.write(f"You: {chat['user']}")
    st.write(f"Bot: {chat['bot']}")

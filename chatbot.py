import requests
from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# Hugging Face Models for Emotion and Sentiment Detection
emotion_detector = pipeline("text-classification", model="ahmettasdemir/distilbert-base-uncased-finetuned-emotion")
sentiment_analyzer = pipeline("text-classification", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")

# Replicate API configuration for LLaMA 3.1
REPLICATE_API_TOKEN = "your_replicate_api_token_here"  # Replace with your Replicate API token
REPLICATE_MODEL_URL = "https://replicate.com/meta/meta-llama-3-70b-instruct/api"  # The API URL for Replicate LLaMA model

# Function to detect emotion
def detect_emotion(user_input):
    result = emotion_detector(user_input)[0]
    return result["label"], result["score"]

# Function to detect sentiment
def detect_sentiment(user_input):
    result = sentiment_analyzer(user_input)[0]
    return result["label"], result["score"]

# Function to generate a response using LLaMA 3.1
def generate_response_with_llama(prompt):
    headers = {
        "Authorization": f"Token {REPLICATE_API_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "version": "fbfb20b472b2f3bdd101412a9f70a0ed4fc0ced78a77ff00970ee7a2383c575d",  # Replace with the actual LLaMA model version ID from Replicate
        "input": {"prompt": prompt}
    }
    response = requests.post(REPLICATE_MODEL_URL, headers=headers, json=payload)

    if response.status_code == 200:
        prediction = response.json()
        return prediction["output"]
    else:
        return "I'm having trouble generating a response right now. Please try again later."

# Combine emotion, sentiment, and LLaMA response
def chatbot_response(user_input):
    emotion, emotion_confidence = detect_emotion(user_input)
    sentiment, sentiment_confidence = detect_sentiment(user_input)
    llama_response = generate_response_with_llama(user_input)

    # Customize responses based on emotion and sentiment
    if sentiment == "POSITIVE":
        sentiment_text = "It's great to hear something positive from you! 🌟"
    else:
        sentiment_text = "It seems like you're expressing something challenging. I'm here for you. 💙"

    emotion_text = {
        "joy": "I'm so happy to hear that! 😊",
        "sadness": "I'm sorry you're feeling this way. 💙",
        "anger": "It seems you're upset. 🧡",
        "fear": "I understand that you're feeling worried. 🌟",
        "love": "That's wonderful to hear! ❤️",
        "surprise": "Wow! That sounds exciting! 🤩",
    }.get(emotion, "Tell me more about how you're feeling.")

    return f"{sentiment_text} {emotion_text} {llama_response} (Emotion: {emotion} Confidence: {emotion_confidence:.2f}, Sentiment: {sentiment} Confidence: {sentiment_confidence:.2f})"

# Define Flask route for chatbot
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data.get("message", "")
    response = chatbot_response(user_input)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)

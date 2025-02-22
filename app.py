import re
import random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from flask import Flask, render_template, request, jsonify

# Chatbot class
class Chatbot:
    def __init__(self):
        self.intents = {
            'greeting': {
                'patterns': ['hello', 'hi', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening'],
                'responses': ['Hello! How can I assist you today?', 'Hi there! What can I do for you?', 'Greetings! How may I help you?']
            },
            'farewell': {
                'patterns': ['bye', 'goodbye', 'see you later', 'farewell', 'take care'],
                'responses': ['Goodbye! Have a great day!', 'Farewell! Don\'t hesitate to return if you need assistance.', 'Take care! Come back anytime.']
            },
            'thanks': {
                'patterns': ['thank you', 'thanks', 'appreciate it', 'thank you so much'],
                'responses': ['You\'re welcome!', 'Glad I could help!', 'It\'s my pleasure to assist you!']
            },
            'about': {
                'patterns': ['who are you', 'what are you', 'tell me about yourself'],
                'responses': ['I\'m a chatbot created to assist users with various queries.', 'I\'m an AI assistant designed to help you with information and tasks.']
            },
            'help': {
                'patterns': ['help', 'I need assistance', 'can you help me', 'what can you do'],
                'responses': ['Sure, I\'d be happy to help! What do you need assistance with?', 'I can help with various topics. What specific area do you need help with?']
            },
            'weather': {
                'patterns': ['what\'s the weather like', 'is it going to rain', 'temperature today'],
                'responses': ['I\'m sorry, I don\'t have real-time weather information. You might want to check a weather app or website for accurate forecasts.']
            },
            'time': {
                'patterns': ['what time is it', 'current time', 'tell me the time'],
                'responses': ['I\'m sorry, I don\'t have access to real-time clock information. You can check the time on your device.']
            },
            'joke': {
                'patterns': ['tell me a joke', 'say something funny', 'make me laugh'],
                'responses': [
                    'Why don\'t scientists trust atoms? Because they make up everything!',
                    'Why did the scarecrow win an award? He was outstanding in his field!',
                    'Why don\'t eggs tell jokes? They\'d crack each other up!'
                ]
            },
        }
        
        self.fallback_responses = [
            "I'm not sure I understand. Could you please rephrase that?",
            "I don't have information about that. Is there something else I can help with?",
            "I'm still learning and don't have an answer for that yet. Can I assist you with something else?",
        ]
        
        # Prepare training data
        self.X = []
        self.y = []
        for intent, data in self.intents.items():
            for pattern in data['patterns']:
                self.X.append(pattern)
                self.y.append(intent)
        
        # Train the model
        self.model = make_pipeline(TfidfVectorizer(), MultinomialNB())
        self.model.fit(self.X, self.y)
    
    def preprocess(self, text):
        # Convert to lowercase and remove punctuation
        text = re.sub(r'[^\w\s]', '', text.lower())
        return text
    
    def classify_intent(self, text):
        preprocessed_text = self.preprocess(text)
        intent = self.model.predict([preprocessed_text])[0]
        return intent
    
    def extract_entities(self, text):
        # This is a very basic entity extraction. In a real-world scenario,
        # you'd use more sophisticated NLP techniques or named entity recognition.
        entities = {}
        
        # Extract dates (very basic implementation)
        date_match = re.search(r'\d{1,2}/\d{1,2}/\d{4}', text)
        if date_match:
            entities['date'] = date_match.group()
        
        # Extract numbers
        number_matches = re.findall(r'\d+', text)
        if number_matches:
            entities['numbers'] = number_matches
        
        return entities
    
    def generate_response(self, text):
        intent = self.classify_intent(text)
        entities = self.extract_entities(text)
        
        if intent in self.intents:
            response = random.choice(self.intents[intent]['responses'])
        else:
            response = random.choice(self.fallback_responses)
        
        # If entities were extracted, append them to the response
        if entities:
            response += f"\nI noticed the following entities in your message: {entities}"
        
        return response

# Initialize Flask app and Chatbot
app = Flask(__name__)
chatbot = Chatbot()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    user_message = request.json['message']
    response = chatbot.generate_response(user_message)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
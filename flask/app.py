import os
from flask import Flask, request, jsonify, render_template
import google.generativeai as genai
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import random
from flask_cors import CORS
import json
import re
from datetime import datetime

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = "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"

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

app = Flask(__name__)
CORS(app)

# In-memory storage for chat histories
chat_histories = {}

class ChatHistoryManager:
    def __init__(self, max_history=10):
        self.max_history = max_history
    
    def add_message(self, session_id, message, response):
        if session_id not in chat_histories:
            chat_histories[session_id] = []
        chat_histories[session_id].append({
            'timestamp': datetime.now().isoformat(),
            'user_message': message,
            'bot_response': response
        })
        if len(chat_histories[session_id]) > self.max_history:
            chat_histories[session_id] = chat_histories[session_id][-self.max_history:]
    
    def get_recent_context(self, session_id, target_language, num_messages=3):
        if session_id not in chat_histories:
            return ""
        recent = chat_histories[session_id][-num_messages:]
        context = ""
        target_language_lower = target_language.lower()
        for chat in recent:
            context += f"User: {chat['user_message']}\n"
            bot_resp = chat['bot_response'].get('translations', {}).get(target_language_lower, '')
            context += f"Bot: {bot_resp}\n"
        return context
    
    def clear_history(self, session_id):
        if session_id in chat_histories:
            chat_histories[session_id] = []

history_manager = ChatHistoryManager()

def get_prompt_template():
    prompt_template_str = """
You are 'Attari Language Bot', a friendly, conversational chatbot teaching {target_language} with explanations in Roman Urdu.
Your role:
- Speak only in {target_language}.
- Use Roman Urdu to explain difficult words, give meanings and grammar tips.
- Detect and correct user's mistakes gently, and explain corrections in Roman Urdu.
- If the user speaks in any language other than {target_language}, politely tell them (in {target_language} and in Roman Urdu) to please speak only in {target_language}.
- Adapt language complexity to {learning_level}:
   * Beginner: Simple sentences, easy vocab, daily life questions.
   * Intermediate: More complex sentences, cultural context.
   * Expert: Advanced grammar, idioms, sophisticated topics.
- Continue conversation naturally, ask follow-up questions in {target_language}.

Previous conversation context:
{chat_context}

User said:
{user_input}

Respond strictly in this JSON format:

{{
  "original_language": "{target_language}",
  "translations": {{
    "{target_language_lower}": "Chatty, friendly {target_language} reply or question or a polite reminder to use only {target_language}.",
    "urdu": "Roman Urdu main wazaahat aur tarjuma, mushkil alfaaz ki tafseel ya user ko sirf {target_language} istemal karne ki darkhwast"
  }},
  "difficult_words": [
    {{
      "language_of_word": "{target_language}",
      "word": "challenging_word_from_response",
      "meaning_in_urdu": "Is lafz ka matlab aur istemal ki misaal"
    }}
  ],
  "learning_tip": "Ek madadgar mashwara {learning_level} satah ke liye.",
  "follow_up_question": "Ek dilchasp sawal guftagu ko aage barhane ke liye."
}}
"""
    return PromptTemplate(
        template=prompt_template_str,
        input_variables=["user_input", "learning_level", "chat_context", "target_language", "target_language_lower"]
    )


def process_user_input(user_input, learning_level, target_language, session_id):
    try:
        model = ChatGroq(
            api_key=GROQ_API_KEY,
            model_name="llama-3.3-70b-versatile",
            temperature=0.8,
            max_tokens=1500
        )
        prompt_template = get_prompt_template()
        
        chat_context = history_manager.get_recent_context(session_id, target_language)
        
        prompt_text = prompt_template.format(
            user_input=user_input,
            learning_level=learning_level,
            chat_context=chat_context,
            target_language=target_language,
            target_language_lower=target_language.lower()
        )
        
        response = model.invoke(prompt_text)
        response_text = getattr(response, 'content', str(response))
        
        # Extract JSON object from the response
        match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if match:
            json_str = match.group(0)
            parsed = json.loads(json_str)
            history_manager.add_message(session_id, user_input, parsed)
            return parsed
        else:
            # fallback if JSON not found
            fallback = fallback_response(user_input, learning_level, target_language, session_id)
            return fallback
    except Exception as e:
        print(f"Error: {e}")
        return error_response()

def fallback_response(user_input, learning_level, target_language, session_id):
    fallback = {
        "original_language": target_language,
        "translations": {
            target_language.lower(): "Sorry, I had a little trouble understanding. Can you please try again?",
            "urdu": "معذرت، مجھے سمجھنے میں تھوڑی مشکل ہوئی۔ کیا آپ دوبارہ کوشش کریں گے؟"
        },
        "difficult_words": [],
        "learning_tip": "صبر اور مشق سے زبان پر عبور حاصل ہوتا ہے۔ کوشش جاری رکھیں۔",
        "follow_up_question": f"Let's try a simple question: What's your favorite word in {target_language}?"
    }
    history_manager.add_message(session_id, user_input, fallback)
    return fallback

def error_response():
    return {
        "original_language": "System",
        "translations": {
            "english": "Oops! Something went wrong. Let's keep practicing!",
            "urdu": "اوہ! کچھ غلط ہو گیا۔ چلو مشق جاری رکھتے ہیں!",
            "arabic": "عفواً! حدث خطأ ما. لنواصل الممارسة!"
        },
        "difficult_words": [],
        "learning_tip": "Every mistake is a step forward in learning!",
        "follow_up_question": "Can you tell me a simple sentence you know in English or Arabic?"
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_input = data.get('message', '').strip()
    learning_level = data.get('level', 'beginner').strip().lower()
    target_language = data.get('language', 'English').strip()
    session_id = data.get('session_id', 'default')

    if not user_input:
        return jsonify({"error": "Message is empty"}), 400

    if learning_level not in ['beginner', 'intermediate', 'expert']:
        return jsonify({"error": "Invalid learning level"}), 400

    if target_language not in ['English', 'Arabic']:
        return jsonify({"error": "Invalid language selected"}), 400

    response = process_user_input(user_input, learning_level, target_language, session_id)
    return jsonify({"response": response})

@app.route('/clear-history', methods=['POST'])
def clear_history():
    data = request.get_json()
    session_id = data.get('session_id', 'default')
    history_manager.clear_history(session_id)
    return jsonify({"message": "History cleared"}), 200

if __name__ == "__main__":
    app.run(debug=True, port=5000)

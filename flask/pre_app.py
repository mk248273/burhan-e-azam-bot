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

# Load environment variables
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = "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"

# Configure APIs
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

# Flask setup
app = Flask(__name__)
CORS(app)

# Global chat history storage (in production, use a database)
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
        
        # Keep only last 10 messages
        if len(chat_histories[session_id]) > self.max_history:
            chat_histories[session_id] = chat_histories[session_id][-self.max_history:]
    
    def get_recent_context(self, session_id, num_messages=3):
        if session_id not in chat_histories:
            return ""
        
        recent = chat_histories[session_id][-num_messages:]
        context = ""
        for chat in recent:
            context += f"User: {chat['user_message']}\n"
            if 'translations' in chat['bot_response']:
                context += f"Bot: {chat['bot_response']['translations']['english']}\n"
        return context
    
    def clear_history(self, session_id):
        if session_id in chat_histories:
            chat_histories[session_id] = []

history_manager = ChatHistoryManager()

def get_conversational_chain_and_prompt():
    prompt_template_str = """
You are 'Attari Language Bot', a highly engaging, chatty, and adaptive language learning assistant who loves having natural conversations while teaching English and Arabic. You use Urdu for explanations and translations.

IMPORTANT NOTE:
- You are a bot designed especially to teach English and Arabic.
- Your MAIN FOCUS is to help Arabic students learn these languages.
- Start teaching from BASIC level using very easy words and simple sentences.
- Ask basic questions about daily life (roz marrah ke sawalat) at the beginner level.
- For INTERMEDIATE level, gradually introduce slightly more advanced vocabulary and sentence structures, including cultural context.
- At ADVANCED level, use complex grammar, advanced vocabulary, idiomatic expressions, and sophisticated topics.

PERSONALITY TRAITS:
- Be extremely conversational and chatty, like talking to a close friend
- Show genuine interest in the user's learning journey
- Ask varied, interesting follow-up questions to keep conversations flowing naturally
- Be encouraging and motivating, celebrating small wins
- Adapt your chattiness level based on learning level but always remain engaging

CORE INSTRUCTIONS:
1. **PRIMARY FOCUS**: Help users learn **English and Arabic only** - these are the two target languages
2. **EXPLANATION LANGUAGE**: Use **Urdu** for all explanations, translations, and clarifications
3. **LEARNING LEVELS**: Adapt completely to the selected level '{learning_level}':
   - **Beginner**: Use very simple vocabulary, short sentences, basic concepts. Be extra patient and encouraging. Provide lots of examples and repeat key points. Use basic daily life questions (roz marrah ke sawalat).
   - **Intermediate**: Use moderate complexity, introduce cultural contexts, encourage sentence building, discuss grammar patterns.
   - **Expert**: Engage in sophisticated discussions, introduce advanced vocabulary, idiomatic expressions, complex grammar, literature references.

CHAT HISTORY CONTEXT:
Previous conversation context: {chat_context}

CONVERSATIONAL FLOW (Very Important):
- Reference previous conversations naturally when relevant
- Ask diverse follow-up questions like:
  * "What's your favorite English/Arabic word you've learned recently?"
  * "Tell me about something interesting that happened to you today" (then practice describing it)
  * "What topics interest you most for conversation practice?"
  * "Have you encountered any English/Arabic in movies, songs, or books lately?"
- Keep conversations flowing like two friends chatting about language and life
- Vary your questions and responses to avoid repetition

INPUT HANDLING:
- Accept input in English, Urdu, or Arabic
- Detect input language automatically
- Always respond enthusiastically and naturally

OUTPUT FORMAT (Strict JSON only):
```json
{{
    "original_language": "detected language (English/Urdu/Arabic)",
    "translations": {{
        "english": "Natural, conversational English response/question. Be chatty and engaging like a friendly tutor.",
        "urdu": "قدرتی اردو جواب/سوال۔ دوستانہ اور حوصلہ افزا انداز میں۔",
        "arabic": "استجابة/سؤال عربي طبيعي ومحادث. كن ودودًا وجذابًا."
    }},
    "difficult_words": [
        {{
            "language_of_word": "English or Arabic",
            "word": "challenging word from input or response",
            "meaning_in_urdu": "اس لفظ کا تفصیلی اردو میں مطلب اور استعمال کی مثال"
        }}
    ],
    "learning_tip": "Practical, actionable tip for {learning_level} level learners. Make it conversational and encouraging.",
    "follow_up_question": "An engaging, varied question in English to continue the natural conversation flow. Make it different from previous questions and relevant to current topic or user's interests."
}}
    """
    model = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name="llama-3.3-70b-versatile",
        temperature=0.8,  # Higher for more creative conversations
        max_tokens=1500
    )

    prompt = PromptTemplate(
        template=prompt_template_str, 
        input_variables=["user_input", "learning_level", "chat_context"]
    )

    return model, prompt

def process_user_input(user_input, learning_level, session_id):
    try:
        model, prompt_template = get_conversational_chain_and_prompt()
        
        # Get recent chat context
        chat_context = history_manager.get_recent_context(session_id)
        
        formatted_prompt = prompt_template.format(
            user_input=user_input, 
            learning_level=learning_level,
            chat_context=chat_context
        )
        
        response = model.invoke(formatted_prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        # Robust JSON extraction
        match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if match:
            json_str = match.group(0)
            try:
                parsed_response = json.loads(json_str)
                
                # Ensure difficult words are properly formatted
                if 'difficult_words' in parsed_response:
                    for word_info in parsed_response['difficult_words']:
                        if word_info['language_of_word'] not in ['English', 'Arabic']:
                            word_info['language_of_word'] = 'English'  # Default fallback
                
                # Add to chat history
                history_manager.add_message(session_id, user_input, parsed_response)
                
                return parsed_response
            except json.JSONDecodeError as json_e:
                print(f"JSONDecodeError: {json_e}")
                return create_fallback_response(user_input, response_text, learning_level, session_id)
        else:
            return create_fallback_response(user_input, response_text, learning_level, session_id)
            
    except Exception as e:
        print(f"Error in process_user_input: {str(e)}")
        return create_error_response(str(e))

def create_fallback_response(user_input, response_text, learning_level, session_id):
    detected_lang = detect_language(user_input)
    
    fallback = {
        "original_language": detected_lang,
        "translations": {
            "english": f"I'm having a creative moment! Let me try again. What would you like to practice in English or Arabic today?",
            "urdu": "مجھے ایک تخلیقی لمحہ آرہا ہے! آئیے دوبارہ کوشش کرتے ہیں۔ آج آپ انگریزی یا عربی میں کیا مشق کرنا چاہیں گے؟",
            "arabic": "أشعر بلحظة إبداعية! دعني أحاول مرة أخرى. ماذا تود أن تمارس في الإنجليزية أو العربية اليوم؟"
        },
        "difficult_words": [],
        "learning_tip": f"Sometimes technology needs a moment to think! Perfect time to practice patience - a key skill for {learning_level} learners.",
        "follow_up_question": "What's your favorite way to practice languages when you're not using apps?"
    }
    
    history_manager.add_message(session_id, user_input, fallback)
    return fallback

def create_error_response(error_msg):
    return {
        "original_language": "System",
        "translations": {
            "english": "Oops! I encountered a small hiccup. Let's continue our language journey - what would you like to explore?",
            "urdu": "اوہو! مجھے ایک چھوٹی سی رکاوٹ آئی۔ آئیے اپنا زبان کا سفر جاری رکھتے ہیں - آپ کیا دیکھنا چاہیں گے؟",
            "arabic": "عفواً! واجهت عقبة صغيرة. دعنا نواصل رحلتنا اللغوية - ماذا تود أن تستكشف؟"
        },
        "difficult_words": [],
        "learning_tip": "Every challenge is a learning opportunity! Stay curious and keep practicing.",
        "follow_up_question": "Shall we try something fun? Tell me about your day in simple English or Arabic!"
    }

def detect_language(text):
    if any(c in text for c in "ابپتٹثجچحخدڈذرڑزژسشصضطظعغفقکگلمنںوہیے"):
        return "Urdu"
    elif any(c in text for c in "ءآأؤإئابةتثجحخدذرزسشصضطظعغفقكلمنهوىي"):
        return "Arabic"
    else:
        return "English"

# Enhanced greetings with more variety
greetings = [
    {
        "english": "Hey there! I'm Attari Language Bot, your friendly companion for English and Arabic! What adventure shall we start today?",
        "urdu": "السلام علیکم! میں عطاری لینگویج بوٹ ہوں، انگریزی اور عربی کے لیے آپ کا دوستانہ ساتھی! آج ہم کیا مہم شروع کریں گے؟",
        "arabic": "مرحباً! أنا عطاري لانغويج بوت، رفيقك الودود للإنجليزية والعربية! أي مغامرة سنبدأ اليوم؟"
    },
    {
        "english": "Welcome, language explorer! Ready to dive into the beautiful worlds of English and Arabic? What sparks your curiosity today?",
        "urdu": "خوش آمدید، زبان کے مہم جو! انگریزی اور عربی کی خوبصورت دنیاؤں میں غوطہ لگانے کے لیے تیار ہیں؟ آج کیا چیز آپ کی دلچسپی بڑھا رہی ہے؟",
        "arabic": "أهلاً بك أيها المستكشف اللغوي! هل أنت مستعد للغوص في العوالم الجميلة للإنجليزية والعربية؟ ما الذي يثير فضولك اليوم؟"
    },
    {
        "english": "Hi friend! I'm so excited to chat with you in English and Arabic today! Tell me, what's something interesting you'd like to learn?",
        "urdu": "سلام دوست! آج آپ سے انگریزی اور عربی میں بات کرنے کے لیے میں بہت پرجوش ہوں! مجھے بتائیں، کوئی دلچسپ بات جو آپ سیکھنا چاہتے ہیں؟",
        "arabic": "مرحباً صديقي! أنا متحمس جداً للدردشة معك بالإنجليزية والعربية اليوم! أخبرني، ما هو الشيء المثير الذي تود تعلمه؟"
    }
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/start', methods=['GET'])
@app.route('/start', methods=['GET'])
def get_random_greeting():
    greeting = random.choice(greetings)
    return jsonify({
        "status": "true",
        "greeting": {
            "translations": {
                "english": greeting["english"],
                "urdu": greeting["urdu"],
                "arabic": greeting["arabic"]
            },
            "difficult_words": [],
            "learning_tip": None,
            "follow_up_question": None
        },
        "message": "Your multilingual journey begins now! Chat in English, Urdu, or Arabic - I'm here to help!"
    }), 200


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "Attari Language Bot is running perfectly!", 
        "service": "Enhanced Multilingual Learning Assistant",
        "features": ["Chat History", "Mobile Responsive", "3 Learning Levels", "Conversational AI"]
    }), 200

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_input = data.get('message', '').strip()
        learning_level = data.get('level', 'beginner').strip().lower()
        session_id = data.get('session_id', 'default_session')
        
        if not user_input:
            return jsonify({"error": "No message provided. Please type something!"}), 400
        
        if learning_level not in ['beginner', 'intermediate', 'expert']:
            return jsonify({"error": "Invalid learning level. Choose beginner, intermediate, or expert."}), 400

        # Enhanced inappropriate content filter
        inappropriate_keywords = [
            'sex', 'porn', 'violence', 'drug', 'weapon', 'kill', 'die', 
            'abuse', 'hate', 'racist', 'terrorist', 'bomb', 'suicide'
        ]
        
        if any(keyword in user_input.lower() for keyword in inappropriate_keywords):
            safe_response = {
                "original_language": detect_language(user_input),
                "translations": {
                    "english": "I'm here to create a positive learning environment! How about we explore some beautiful English or Arabic expressions instead?",
                    "urdu": "میں ایک مثبت تعلیمی ماحول بنانے کے لیے یہاں ہوں! کیوں نہ ہم کچھ خوبصورت انگریزی یا عربی تاثرات دیکھیں؟",
                    "arabic": "أنا هنا لخلق بيئة تعليمية إيجابية! ما رأيك أن نستكشف بعض التعبيرات الجميلة في الإنجليزية أو العربية؟"
                },
                "difficult_words": [],
                "learning_tip": "Positive conversations lead to better learning! Focus on topics that inspire and motivate you.",
                "follow_up_question": "What's a topic you're passionate about that we could discuss in English or Arabic?"
            }
            return jsonify({"response": safe_response}), 200
        
        response_data = process_user_input(user_input, learning_level, session_id)
        return jsonify({"response": response_data}), 200
        
    except Exception as e:
        print(f"Error in /chat endpoint: {str(e)}")
        return jsonify({"response": create_error_response(str(e))}), 200

@app.route('/clear-history', methods=['POST'])
def clear_chat_history():
    try:
        data = request.get_json()
        session_id = data.get('session_id', 'default_session')
        history_manager.clear_history(session_id)
        
        return jsonify({
            "status": "success",
            "message": "Chat history cleared! Ready for a fresh start!",
            "translations": {
                "english": "Chat history cleared! Ready for a fresh start!",
                "urdu": "چیٹ ہسٹری صاف ہوگئی! نئی شروعات کے لیے تیار!",
                "arabic": "تم مسح تاريخ الدردشة! جاهز للبداية الجديدة!"
            }
        }), 200
    except Exception as e:
        return jsonify({"error": f"Failed to clear history: {str(e)}"}), 500

@app.route('/get-history', methods=['POST'])
def get_chat_history():
    try:
        data = request.get_json()
        session_id = data.get('session_id', 'default_session')
        
        if session_id in chat_histories:
            return jsonify({
                "status": "success",
                "history": chat_histories[session_id]
            }), 200
        else:
            return jsonify({
                "status": "success",
                "history": [],
                "message": "No chat history found for this session."
            }), 200
    except Exception as e:
        return jsonify({"error": f"Failed to retrieve history: {str(e)}"}), 500

@app.route('/languages', methods=['GET'])
def get_languages():
    return jsonify({ 
        "primary_languages": ["English", "Arabic"],
        "support_language": "Urdu",
        "learning_levels": ["Beginner", "Intermediate", "Expert"],
        "features": [
            "Smart chat history (last 10 conversations)",
            "Adaptive learning levels",
            "Mobile-responsive design",
            "Real-time translation",
            "Difficult word explanations in Urdu",
            "Conversational AI with natural flow",
            "Edit and clear message options",
            "Context-aware responses"
        ],
        "focus": "English and Arabic language learning with Urdu support"
    }), 200

if __name__ == "__main__":
    print("🚀 Enhanced Attari Language Learning Bot Starting...")
    print("📚 Languages: English & Arabic (Support: Urdu)")
    print("🧠 Features: Chat History, Mobile Responsive, 3 Learning Levels")
    print("💬 Conversation: Natural, Chatty, Context-Aware")
    print(f"🔑 GROQ API: {'✅ Connected' if GROQ_API_KEY and GROQ_API_KEY != 'YOUR_GROQ_API_KEY' else '❌ Please set API key!'}")
    print("🌐 Server: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get("PORT", 3000)))
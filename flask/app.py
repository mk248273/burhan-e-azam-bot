import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from flask import Flask, request, jsonify, render_template
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import random
from flask_cors import CORS

# Load environment variables
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # Load from .env file for security
genai.configure(api_key=GOOGLE_API_KEY)

# Flask setup
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Helper functions
def get_pdf_text(pdf):
    text = ""
    pdf_reader = PdfReader(pdf)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500)
    chunks = splitter.split_text(text)
    return chunks  # list of strings

def get_vector_store(chunks):
    # Create directory if it doesn't exist
    os.makedirs("faiss_index", exist_ok=True)
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # type: ignore
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

def get_conversational_chain():
    prompt_template = """
System Instructions for Burhan e Azam Assistant:

1. Role: You are the official virtual assistant for Burhan e Azam.
2. Response Rule: Answer ONLY using information from the provided context.
3. Knowledge Limit: If information isn't in context, say "I apologize, this information is not available in the current context."
4. Tone: Maintain warm, professional, and empathetic communication.
5. Prohibited: No URLs, no assumptions, no personal opinions, no external information.
6. Quality: Ensure responses are grammatically correct and professionally written.
7. Privacy: Protect sensitive information and maintain confidentiality.
8. Support: For unavailable information, direct users to the Burhan e Azam office.
9. Values: Every response should reflect Burhan e Azam values of faith, community, and service.\n\n
    Context:\n {context}\n
    Question: \n{question}?\n

    Answer:
    """

    model = ChatGroq(
        api_key="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",
        model_name="llama-3.3-70b-versatile",
    )

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    try:
        # Use absolute path for more reliability
        base_dir = os.path.dirname(os.path.abspath(__file__))
        index_path = os.path.join(base_dir, "faiss_index")
        
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        docs = docs[:10]
        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        # Ensure response format matches what frontend expects
        return {"output_text": response["output_text"]}
    except Exception as e:
        # Return a user-friendly error message
        return {"output_text": f"I'm having trouble accessing my knowledge base. Please make sure documents have been uploaded first. Technical details: {str(e)}"}

# Fixed greetings with simple ASCII characters
greetings = {
    1: "As-salamu alaykum! I'm Gaby, your friendly assistant here at Burhan E Azam. How can I serve you today?",
    2: "Welcome to Burhan E Azam! I'm Gaby, here to help with anything you need. Let's get started!",
    3: "As-salamu alaykum! I'm Gaby, your guide to everything at Burhan E Azam. What's on your heart today?",
    4: "As-salamu alaykum, friend! I'm Gaby, here at Burhan E Azam to walk this journey with you. How can I support you today?",
    5: "As-salamu alaykum! I'm Gaby - think of me as Burhan E Azam's friendly helper. What's on your mind today?",
    6: "As-salamu alaykum! I'm Gaby from Burhan E Azam. If you've got questions or need help, I'm all ears!",
    7: "As-salamu alaykum! I'm Gaby, your Burhan E Azam assistant. Let's make this easy - what can I help with?",
    8: "As-salamu alaykum! Whether you need prayer, info about our services, or just a friendly chat, I'm here for you at Burhan E Azam.",
    9: "As-salamu alaykum, friend! Looking to connect at Burhan E Azam? I'd love to help you find your place!",
    10: "As-salamu alaykum! Welcome to Burhan E Azam! I'm Gaby, and I'd love to help you feel at home. How can I assist?",
    11: "As-salamu alaykum! I'm Gaby from Burhan E Azam. You're in the right place. What can I help you with today?"
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/start', methods=['GET'])
def get_random_greeting():
    start_message = random.choice(list(greetings.values()))
    return jsonify({"status": "true", "start chat": start_message}), 200

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "API is running"}), 200

# Add a route to upload PDF files and create the index
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and file.filename.endswith('.pdf'):
        try:
            text = get_pdf_text(file)
            chunks = get_text_chunks(text)
            get_vector_store(chunks)
            return jsonify({"message": "Document processed and index created successfully"}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "Only PDF files are supported"}), 400

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    user_question = data.get('query')

    if not user_question:
        return jsonify({"error": "No query provided"}), 400
    
    # Check if index exists before proceeding
    base_dir = os.path.dirname(os.path.abspath(__file__))
    index_path = os.path.join(base_dir, "faiss_index", "index.faiss")
    
    if not os.path.exists(index_path):
        return jsonify({
            "response": {
                "output_text": "I don't have any knowledge yet. Please upload documents first by using the /upload endpoint."
            }
        }), 200
    
    try:
        response = user_input(user_question)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({
            "response": {
                "output_text": f"I encountered an error while processing your question. Please try again later. Error: {str(e)}"
            }
        }), 200  # Return 200 with error message in response for better UX

# For debugging and development only
@app.route('/init-dummy-data', methods=['GET'])
def init_dummy_data():
    """Initialize with some dummy data for testing purposes"""
    try:
        dummy_text = """
        Burhan e Azam is a community center dedicated to Islamic education and worship.
        We offer daily prayer services, Quran study classes, and community events.
        Our mission is to foster a sense of faith, community, and service among all members.
        We are located at 123 Islamic Way, with services starting at 5 AM daily.
        For any questions, please contact our office at info@burhanazam.org or call 555-123-4567.
        """
        chunks = get_text_chunks(dummy_text)
        get_vector_store(chunks)
        return jsonify({"message": "Dummy data initialized successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=7888)
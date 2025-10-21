from openai import OpenAI
from dotenv import load_dotenv
import os
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Cho phép frontend gọi API

load_dotenv()  # Load biến môi trường từ file .env


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))



@app.route('/chatbot', methods=['POST'])  
def chat_with_ai():
    data = request.get_json()
    prompt = data.get('message', '')
    
    if not prompt:
        return jsonify({'error': 'No message provided'}), 400
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return jsonify({
            'response': response.choices[0].message.content.strip()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=8080)
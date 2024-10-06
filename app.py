from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
from dotenv import load_dotenv
from openai import OpenAI

app = Flask(__name__)
CORS(app)

endpoint = "https://models.inference.ai.azure.com"
model_name = "gpt-4o-mini"
token = "api key"

client = OpenAI(
    base_url=endpoint,
    api_key=token,
)

SYSTEM_PROMPTS = {
    "medical": "You are a medical information assistant. Provide accurate and helpful information about health and medical topics. Always advise users to consult with a healthcare professional for personalized medical advice. All response should be short and precise and don't need to write more details be specific and to the point",
    "sports": "You are a sports information assistant. Provide up-to-date information about various sports, athletes, teams, and events. All response should be short and precise and don't need to write more details be specific and to the point",
    "education": "You are an educational assistant. Provide helpful information on various academic subjects and learning topics. All response should be short and precise and don't need to write more details be specific and to the point",
    "entertainment": "You are an entertainment information assistant. Provide information about movies, TV shows, music, celebrities, and other entertainment topics. All response should be short and precise and don't need to write more details be specific and to the point",
    "mathematics": "You are a mathematics assistant. Help with mathematical concepts, problem-solving, and calculations. All response should be short and precise and don't need to write more details be specific and to the point"
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data['message']
    prompt_type = data['promptType']
    
    system_prompt = SYSTEM_PROMPTS.get(prompt_type, SYSTEM_PROMPTS['education'])
    
    try:
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_message,
                }
            ],
            temperature=1.0,
            top_p=1.0,
            max_tokens=1000,
            model=model_name
        )
        bot_response = response.choices[0].message.content
        return jsonify({"response": bot_response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
Mental Health Companion - NLP Chatbot 💬
Overview
Mental Health Companion is an AI-powered chatbot designed to assist users with mental health concerns. It analyzes user queries, detects emotional states, and provides supportive responses based on natural language processing (NLP). The chatbot is built using Streamlit, BERT-based classification, and Facebook’s BlenderBot for response generation.

Features
✅ Sentiment & Mental Health Analysis - Classifies mood as anxiety, depression, normal, panic, stress, or suicidal.
✅ AI Chatbot - Responds naturally using Facebook’s BlenderBot.
✅ Interactive UI - Built with Streamlit for a smooth user experience.
✅ Pre-trained NLP Model - Uses BERT for sentiment classification.
✅ Session-Based Chat History - Stores past interactions dynamically.

Installation
1. Clone the Repository
git clone https://github.com/your-username/Mental-Health-Companion_NLP-Chatbot.git
cd Mental-Health-Companion_NLP-Chatbot

2. Create a Virtual Environment
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

3. Install Dependencies
pip install -r requirements.txt

**Usage**
Run the Chatbot
streamlit run app.py

Interact with the Chatbot
Open the Streamlit UI in your browser.
Type your message in the input box.
The chatbot will analyze your mood and provide a response.
Technology Stack
Python
Streamlit (Frontend)
PyTorch & Transformers (BERT & BlenderBot)
Scikit-learn (Label Encoding)
NumPy
How It Works
User inputs a message 📩
BERT model classifies mood 🤖
BlenderBot generates a response 💬
Chat history is updated dynamically 📜
Future Improvements
🔹 Integrate real-time mental health resources 📚
🔹 Improve chatbot conversational ability 🗣️
🔹 Deploy as a web service 🌐

Contributing
Feel free to fork this repository and submit pull requests. Contributions are always welcome!

License
This project is licensed under the MIT License.

Let me know if you need any modifications! 🚀

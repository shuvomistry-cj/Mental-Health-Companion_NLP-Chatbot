import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Streamlit UI Configuration
st.set_page_config(page_title="Mental Health Chatbot", page_icon="💬", layout="centered")
st.markdown("<h1 style='color:#FF5733; text-align: center;'>Mental Health Companion</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='color:#2ECC71; text-align: center;'>Your AI mental health assistant</h4>", unsafe_allow_html=True)

# Initialize session states
if "model_loaded" not in st.session_state:
    st.session_state["model_loaded"] = False

if "mood" not in st.session_state:
    st.session_state["mood"] = "Unknown"
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Load models with progress indication
if not st.session_state["model_loaded"]:
    with st.spinner('Loading models... This might take a minute...'):
        try:
            # Load Classification Model (BERT)
            model_name = "bert-base-uncased"
            bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
            bert_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=6)

            # Initialize Label Encoder
            label_encoder = LabelEncoder()
            label_encoder.classes_ = np.array(['anxiety', 'depression', 'normal', 'panic', 'stress', 'suicidal'])

            # Store models in session state
            st.session_state["bert_tokenizer"] = bert_tokenizer
            st.session_state["bert_model"] = bert_model
            st.session_state["label_encoder"] = label_encoder
            
            # Initialize chatbot
            st.session_state["chatbot"] = pipeline("text2text-generation", model="facebook/blenderbot-400M-distill")
            
            st.session_state["model_loaded"] = True
            st.success("Models loaded successfully!")
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
            st.stop()

def classify_mood(user_input):
    try:
        inputs = st.session_state["bert_tokenizer"](user_input, return_tensors="pt", truncation=True, padding=True, max_length=128)
        outputs = st.session_state["bert_model"](**inputs)
        mood = torch.argmax(outputs.logits, dim=1).item()
        return st.session_state["label_encoder"].inverse_transform([mood])[0]
    except Exception as e:
        st.error(f"Error classifying mood: {str(e)}")
        return "Unknown"

def generate_response(user_input):
    try:
        response = st.session_state["chatbot"](user_input, max_length=150, truncation=True)
        return response[0]["generated_text"]
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return "I apologize, but I'm having trouble generating a response right now."

# Chat Display
st.markdown("---")
for message in st.session_state["chat_history"]:
    role, text = message
    if role == "bot":
        st.markdown(f"<div style='background-color:#2b2b2b; padding:10px; border-radius:5px; margin:5px 0;'><span style='color:#FF5733; font-weight:bold;'>Bot:</span><br><span style='color: #FFFFFF;'>{text}</span></div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='background-color:#1e1e1e; padding:10px; border-radius:5px; margin:5px 0;'><span style='color:#2ECC71; font-weight:bold;'>You:</span><br><span style='color:#FFFFFF;'>{text}</span></div>", unsafe_allow_html=True)

# User Input
user_input = st.text_input("Type your message here:", key="input")

if st.button("Send") and user_input:
    with st.spinner("Processing..."):
        # Get Mood Classification
        mood = classify_mood(user_input)
        
        # Get Response
        response = generate_response(user_input)
        
        # Update chat history
        st.session_state["chat_history"].append(("user", user_input))
        st.session_state["chat_history"].append(("bot", f"*Mood:* {mood}\n*Response:* {response}"))
        
        st.rerun()
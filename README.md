# RAG-model

A simple Retrieval-Augmented Generation (RAG) app built with Streamlit and OpenAI.
Upload any text-based PDF, and ask natural-language questions — the app retrieves relevant parts and answers intelligently.

How to run locally -
git clone https://github.com/<your-username>/pdf-rag-chatbot.git
cd pdf-rag-chatbot
pip install -r requirements.txt
export OPENAI_API_KEY="sk-your-api-key"
streamlit run app.py

Tech stack -
Streamlit · OpenAI · FAISS · NumPy · PyPDF

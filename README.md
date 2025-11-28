# RAG-Chatbot

A Retrieval-Augmented Generation based chatbot that allows users upload PDF and chat with its content. It uses langchain memory to maintain conversational context and provide context-aware responses.

### Technologies Used: 

LangChain, Pinecone, Streamlit 

### How to run: 

pip install -r requirements.txt

Create .env file with PINECONE_API_KEY,GROQ_API_KEY

streamlit run app.py

# RAG-Chatbot

A Corrective Retrieval-Augmented Generation-cRAG chatbot that allows users to chat with the content of uploaded pdf. It provides context-aware responses using LangChain memory. It also performs web search to answer user queries when context is insufficient.

For agent workflow, see [src/agent.png]


## Technologies Used: 

LangChain, LangGraph, Pinecone, Streamlit

## How to run: 

pip install -r requirements.txt

Create .env file with PINECONE_API_KEY, GROQ_API_KEY, TAVILY_API_KEY

streamlit run app.py

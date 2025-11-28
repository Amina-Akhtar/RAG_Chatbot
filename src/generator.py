from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from src.embeddings import retrieve_embeddings

prompt="You are an expert assistant to answer the questions based on the pdf content. " \
       "Use only the following extracted information  to answer the questions. " \
       "If you do not know the answer say: I could not find any relevant information in the document. " \
       "Keep the answers concise, use 3 sentences maximum." \
       "{context}"

template=ChatPromptTemplate.from_messages(
    [
        ("system", prompt),
        ("human", "{input}"),
    ]
)
memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key='answer' ,
    k=3
)
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.5,
    max_tokens=None
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm, 
    retriever=retrieve_embeddings(),
    memory=memory,
    return_source_documents=True
)

def generate_response(user_input,chat_history):
    response=qa_chain.invoke({"question":user_input, "chat_history":chat_history})
    response=response["answer"]
    return response

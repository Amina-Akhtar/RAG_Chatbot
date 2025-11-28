import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone,ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()
PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"]=PINECONE_API_KEY
index_name="custom-chatbot"
embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
pc = Pinecone(api_key=PINECONE_API_KEY)

def create_index(): 
    #run only one time to create index
    if index_name not in [index.name for index in pc.list_indexes()]:
        pc.create_index(       
            name=index_name,
            dimension=384, 
            metric="cosine", 
            spec=ServerlessSpec(
                cloud="aws", 
                region="us-east-1") 
            ) 
        print(f"Index '{index_name}' created successfully")
    else:
        print(f"Index '{index_name}' already exists")

def delete_index():
    index = pc.Index(index_name)
    index.delete(delete_all=True)

# store embeddings in Pinecone
def store_embeddings(chunks):
        index_data = PineconeVectorStore.from_documents(
        documents=chunks,
        index_name=index_name,
        embedding=embeddings )
    
def retrieve_embeddings():
    load_doc = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings)
    retriever = load_doc.as_retriever(search_type="similarity", search_kwargs={"k":3})
    return retriever
    
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def upload_document(file):
    with open("PDF/" + file.name, "wb") as f:
        f.write(file.getbuffer())

def load_document(doc_path):
    loader = PDFPlumberLoader(doc_path)
    document = loader.load()
    return document

def split_document(doc_text):
    split = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100)
    text = split.split_documents(doc_text)
    #texts = [t.page_content for t in text]
    return text
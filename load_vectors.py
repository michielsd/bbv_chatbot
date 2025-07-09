import os
from dotenv import load_dotenv
import fitz  # from PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
from langchain.docstore.document import Document

# Load variables from .env file
load_dotenv()

# GLOBALS
PDF_FOLDER = "C:/Dashboard/Werk/Streamlit/chatbot/bbv_chatbot/notities_bbv/"

def main():

    documents = load_and_split_pdfs(PDF_FOLDER)
    vectorstore = create_vectorstore(documents)
    
    print("Vectorstore created")
    

def load_and_split_pdfs(folder_path):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    all_chunks = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            filepath = os.path.join(folder_path, filename)
            pdf = fitz.open(filepath)
            documents = []

            for page in pdf:
                page_text = page.get_text()
                page_number = page.number
                
                # Wrap the page text in a Document object with metadata
                document = Document(
                    page_content=page_text,
                    metadata={
                        "source": filename,
                        "page": page_number
                    }
                )
                documents.append(document)

            source_doc = documents

            # Split into smaller chunks, keeping the metadata
            chunks = splitter.split_documents(documents)
            all_chunks.extend(chunks)

    return all_chunks

#def create_vectorstore(docs, persist_directory="vectorstore_bbv"):
#    embeddings = OpenAIEmbeddings()
    
#    vectorstore = Chroma.from_documents(
#        documents=docs,
#        embedding=embeddings,
#        persist_directory=persist_directory
#    )

#    vectorstore.persist()  # Ensure it's saved to disk
#    return vectorstore


def create_vectorstore(docs):
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("vectorstore_bbv")
    return vectorstore

if __name__ == "__main__":
    main()
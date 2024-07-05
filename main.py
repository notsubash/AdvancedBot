import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import WebBaseLoader
from langchain.memory import ConversationBufferMemory
import os
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import pickle
import shutil
from langchain.docstore.document import Document
import warnings
import json
warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()

# Set up OpenAI API key
os.environ["OPENAI_API_KEY"] = st.secrets["openai"]["api_key"]

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
os.makedirs(STATIC_DIR, exist_ok=True)

# Initialize session state
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "collections" not in st.session_state:
    st.session_state.collections = {}

if "current_collection" not in st.session_state:
    st.session_state.current_collection = None

# Function to add to chat history
def add_to_chat_history(question, answer):
    if isinstance(st.session_state.chat_history, list):
        st.session_state.chat_history.append({"question": question, "answer": answer})
    else:
        st.session_state.chat_history = [{"question": question, "answer": answer}]

# Function to display chat history
def display_chat_history():
    for entry in reversed(st.session_state.chat_history):
        st.write(f"Human: {entry['question']}")
        st.write(f"AI: {entry['answer']}")
        st.write("---")


# File paths
COLLECTIONS_DIR = "collections"
CHAT_HISTORY_FILE = "chat_history.pkl"

# Ensure collections directory exists
os.makedirs(COLLECTIONS_DIR, exist_ok=True)

# Load existing collections
st.session_state.collections = {
    name: None for name in os.listdir(COLLECTIONS_DIR) if os.path.isdir(os.path.join(COLLECTIONS_DIR, name))
}

# Load chat history
#if os.path.exists(CHAT_HISTORY_FILE):
    #with open(CHAT_HISTORY_FILE, "rb") as f:
        #st.session_state.chat_history = pickle.load(f)

def process_pdf(pdf_file):
    if pdf_file is not None:
        docs = []
        reader = PdfReader(pdf_file)
        i = 1
        for page in reader.pages:
            docs.append(Document(page_content=page.extract_text(), metadata={"source": pdf_file.name,'page':i}))
            i += 1
        
    return docs

def process_url(url):
    loader = WebBaseLoader(url)
    documents = loader.load()
    return documents

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    return texts

def save_collection(name, vectorstore):
    collection_dir = os.path.join(COLLECTIONS_DIR, name)
    os.makedirs(collection_dir, exist_ok=True)
    # Save the vectorstore with consistent naming
    vectorstore.save_local(collection_dir, index_name=f"{name}")

def load_collection(name):
    collection_dir = os.path.join(COLLECTIONS_DIR, name)
    if os.path.exists(os.path.join(collection_dir, f'{name}.faiss')):
        return FAISS.load_local(collection_dir, OpenAIEmbeddings(), allow_dangerous_deserialization=True, index_name=f'{name}')
    return None

def delete_collection(name):
    collection_dir = os.path.join(COLLECTIONS_DIR, name)
    if os.path.exists(collection_dir):
        shutil.rmtree(collection_dir)

def save_chat_history():
    with open(CHAT_HISTORY_FILE, "wb") as f:
        pickle.dump(st.session_state.chat_history, f)

def is_duplicate(vectorstore, text):
    # Search for similar documents
    results = vectorstore.similarity_search(text, k=1)
    if results:
        similarity = vectorstore.similarity_search_with_score(text, k=1)[0][1]
        # You can adjust this threshold as needed
        return similarity > 0.90
    return False

def main():
    st.title("PDF📑 & URL🔗 Chatbot")

    # Sidebar for collection management
    st.sidebar.header("Collection Management")
    
    # Create new collection
    new_collection_name = st.sidebar.text_input("Create new collection")
    if st.sidebar.button("Create"):
        if new_collection_name and new_collection_name not in st.session_state.collections:
            collection_dir = os.path.join(COLLECTIONS_DIR, new_collection_name)
            if not os.path.exists(collection_dir):
                vectorstore = FAISS.from_texts([""], OpenAIEmbeddings())
                save_collection(new_collection_name, vectorstore)
                st.session_state.collections[new_collection_name] = None
                st.sidebar.success(f"Collection '{new_collection_name}' created.")
            else:
                st.sidebar.error("Collection already exists on disk. Please choose a different name.")
        elif new_collection_name in st.session_state.collections:
            st.sidebar.error("Collection already exists.")
        else:
            st.sidebar.error("Please enter a collection name.")

    # Select existing collection
    st.session_state.current_collection = st.sidebar.selectbox(
        "Select collection", 
        options=list(st.session_state.collections.keys()),
        index=0 if st.session_state.collections else None
    )

    # Delete collection
    if st.sidebar.button("Delete Selected Collection"):
        if st.session_state.current_collection:
            delete_collection(st.session_state.current_collection)
            del st.session_state.collections[st.session_state.current_collection]
            st.sidebar.success(f"Collection '{st.session_state.current_collection}' deleted.")
            st.session_state.current_collection = None
        else:
            st.sidebar.error("No collection selected.")

    # File upload and URL input
    uploaded_file = st.sidebar.file_uploader("Upload a document (PDF or CSV)", type=["pdf"])
    url = st.sidebar.text_input("Or enter a URL")
    
    # Submit button for processing documents
    if st.sidebar.button("Submit Document"):
        if st.session_state.current_collection:
            try:
                vectorstore = load_collection(st.session_state.current_collection)
                if vectorstore is None:
                    vectorstore = FAISS.from_texts([""], OpenAIEmbeddings(), index_name=new_collection_name)

                new_docs = []
                if uploaded_file:
                    if uploaded_file.type == "application/pdf":
                        documents = process_pdf(uploaded_file)
                    
                    texts = split_documents(documents)
                    for doc in texts:
                        if not is_duplicate(vectorstore, doc.page_content):
                            new_docs.append(doc.page_content)
                    
                elif url:
                    documents = process_url(url)
                    texts = split_documents(documents)
                    for doc in texts:
                        if not is_duplicate(vectorstore, doc.page_content):
                            new_docs.append(doc.page_content)

                if new_docs:
                    vectorstore.add_texts(new_docs)
                    save_collection(st.session_state.current_collection, vectorstore)
                    st.sidebar.success(f"{len(new_docs)} new document chunks added to the collection.")
                else:
                    st.sidebar.info("No new content to add. All documents are already in the collection.")

            except Exception as e:
                st.sidebar.error(f"An error occurred: {str(e)}")
        else:
            st.sidebar.error("Please select a collection first.")

    # Clear chat history button
    if st.sidebar.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.session_state.memory.clear()
        save_chat_history()
        st.sidebar.success("Chat history cleared.")

    # Question answering
    st.header("Chat with your documents")
    question = st.text_input("Enter your question")
  

    if question and st.session_state.current_collection:
        try:
            vectorstore = load_collection(st.session_state.current_collection)
            if vectorstore is None:
                vectorstore = FAISS.from_texts([""], OpenAIEmbeddings(), index_name=st.session_state.current_collection)
            if vectorstore:
                qa = ConversationalRetrievalChain.from_llm(
                    llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
                    retriever=vectorstore.as_retriever(),
                    memory=st.session_state.memory
                )

                response = qa({"question": question})
                
                # Extract answer from response
                answer = response.get('answer', str(response))
                
                # Add to chat history
                add_to_chat_history(question, answer)
                
                save_chat_history()

                # Display chat history
                display_chat_history()
            else:
                st.error("Selected collection is empty. Please add documents to it.")
        except Exception as e:
            st.error(f"An error occurred while processing your question: {str(e)}")
            st.write("Debug: Exception details:", str(e))

if __name__ == "__main__":
    main()

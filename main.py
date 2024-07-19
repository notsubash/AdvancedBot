__import__('pysqlite3')
import sys

sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import chat, collection_management, document_management
import logging
from config import OPENAI_API_KEY, COLLECTIONS_FOLDER
from utils.vector_store import list_collections, select_collection
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

def main():
    try:
        st.set_page_config(page_title="RAG Application", layout="wide")
        st.sidebar.title("Navigation")
        page = st.sidebar.radio("Go to", ["Chat", "Collection Management", "Document Management"])

        if page == "Chat":
            chat.render()
        elif page == "Collection Management":
            collection_management.render()
        elif page == "Document Management":
            document_management.render()

        st.sidebar.title("Select Active Collection")
        collections = list_collections()
        default_collection = "NMB"  # Set your default collection name here

        if collections:
            active_collection = st.sidebar.selectbox(
                "Choose",
                options=collections,
                index=collections.index(default_collection) if default_collection in collections else 0
            )
            select_collection(active_collection)
        else:
            st.sidebar.warning("No collections available. Please create a collection first.")
        
        logger.info(f"User navigated to {page} page")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        st.error(f"An error occurred: {str(e)}")
        #st.experimental_rerun()

if __name__ == "__main__":
    if not OPENAI_API_KEY:
        st.error("OpenAI API key is not set. Please set it in your environment variables.")
    else:
        main()

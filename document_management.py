import streamlit as st
from utils.document_loader import load_document
from utils.document_splitter import split_document
from utils.vector_store import add_texts_to_collection, get_indexed_documents
import logging
import pandas as pd

logger = logging.getLogger(__name__)

def render():
    st.title("Document Management")

    upload_type = st.radio("Upload type", ["File", "URL"])
    if upload_type == "File":
        uploaded_files = st.file_uploader("Choose files", type=["txt", "pdf", "docx", "csv", "md", "json"], accept_multiple_files=True)
        if uploaded_files:
            for uploaded_file in uploaded_files:
                try:
                    doc = load_document(uploaded_file)
                    logger.info(f"File uploaded: {uploaded_file.name}")
                    chunks = split_document(doc)
                    metadatas = [{"filename": uploaded_file.name, "source": uploaded_file.name} for _ in chunks]
                    add_texts_to_collection([chunk.page_content for chunk in chunks], metadatas)

                    st.success(f"File processed and added to the collection: {uploaded_file.name}")
                except Exception as e:
                    logger.error(f"Error processing file {uploaded_file.name}: {str(e)}")
                    st.error(f"Error processing file {uploaded_file.name}: {str(e)}")

    else:
        url = st.text_input("Enter a URL")
        if url:
            try:
                doc = load_document(url)
                logger.info(f"URL loaded: {url}")
            except Exception as e:
                logger.error(f"Error loading URL: {str(e)}")
                st.error(f"Error loading URL: {str(e)}")

    if st.button("Process Documents"):
        if uploaded_files:
            for uploaded_file in uploaded_files:
                try:
                    docs = load_document(uploaded_file)
                    logger.info(f"File uploaded: {uploaded_file.name}")
                    chunks = split_document(docs)
                    metadatas = [{"source": uploaded_file.name} for _ in chunks]
                    add_texts_to_collection([chunk.page_content for chunk in chunks], metadatas)

                    st.success(f"File processed and added to the collection: {uploaded_file.name}")
                except Exception as e:
                    logger.error(f"Error processing file {uploaded_file.name}: {str(e)}")
                    st.error(f"Error processing file {uploaded_file.name}: {str(e)}")
        else:
            st.warning("Please upload files before processing.")

    st.subheader("Indexed Documents")
    try:
        indexed_docs = get_indexed_documents()
        if indexed_docs:
            df = pd.DataFrame({"Document": indexed_docs})
            st.table(df)
        else:
            st.write("No documents indexed yet.")
    except Exception as e:
        logger.error(f"Error retrieving indexed documents: {str(e)}")
        st.error("An error occurred while retrieving indexed documents.")




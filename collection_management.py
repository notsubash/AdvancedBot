import streamlit as st
from utils.vector_store import create_collection, delete_collection, list_collections, select_collection
import logging

logger = logging.getLogger(__name__)

def render():
    st.title("Collection Management")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Create Collection")
        new_collection = st.text_input("Enter new collection name")
        if st.button("Create"):
            try:
                create_collection(new_collection)
                logger.info(f"Collection '{new_collection}' created successfully")
                st.success(f"Collection '{new_collection}' created successfully!")
            except Exception as e:
                logger.error(f"Error creating collection: {str(e)}")
                st.error(f"Error creating collection: {str(e)}")

    with col2:
        st.subheader("Delete Collection")
        collections = list_collections()
        collection_to_delete = st.selectbox("Select collection to delete", collections)
        if st.button("Delete"):
            try:
                delete_collection(collection_to_delete)
                logger.info(f"Collection '{collection_to_delete}' deleted successfully")
                st.success(f"Collection '{collection_to_delete}' deleted successfully!")
            except Exception as e:
                logger.error(f"Error deleting collection: {str(e)}")
                st.error(f"Error deleting collection: {str(e)}")

    st.subheader("Select Active Collection")
    active_collection = st.selectbox("Select active collection", list_collections())
    if st.button("Set Active"):
        try:
            select_collection(active_collection)
            logger.info(f"Collection '{active_collection}' set as active")
            st.success(f"Collection '{active_collection}' set as active!")
        except Exception as e:
            logger.error(f"Error setting active collection: {str(e)}")
            st.error(f"Error setting active collection: {str(e)}")

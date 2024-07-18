from langchain.retrievers import ParentDocumentRetriever, SelfQueryRetriever, MultiQueryRetriever
from langchain_openai import ChatOpenAI
import logging
from langchain.storage import InMemoryStore
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.schema import Document

logger = logging.getLogger(__name__)

def get_retriever(vector_store):
    try:
        retriever = vector_store.as_retriever()
        logger.debug("Basic retriever created")
        return retriever
    except Exception as e:
        logger.error(f"Error creating basic retriever: {str(e)}")
        raise

def get_parent_child_retriever(vector_store, child_splitter):
    try:
        byte_store = InMemoryStore()
        retriever = ParentDocumentRetriever(
            vectorstore=vector_store,
            child_splitter=child_splitter,
            byte_store=byte_store,
        )
        logger.debug("Parent-child retriever created")
        return retriever
    except Exception as e:
        logger.error(f"Error creating parent-child retriever: {str(e)}")
        raise


def get_self_query_retriever(vector_store, llm):
    try:
        metadata_field_info = [
            {"name": "source", "description": "The source of the document", "type": "string"},
            {"name": "date", "description": "The date the document was created", "type": "string"},
            # Add more metadata fields as needed
        ]
        document_content_description = "Document containing information about various topics"
        retriever = SelfQueryRetriever.from_llm(
            llm,
            vector_store,
            document_content_description,
            metadata_field_info=metadata_field_info,
            verbose=True
        )
        logger.debug("Self-query retriever created")
        return retriever
    except Exception as e:
        logger.error(f"Error creating self-query retriever: {str(e)}")
        raise


def get_multi_query_retriever(vector_store):
    try:
        llm = ChatOpenAI(temperature=0)
        retriever = MultiQueryRetriever.from_llm(
            retriever=vector_store.as_retriever(),
            llm=llm
        )
        logger.debug("Multi-query retriever created")
        return retriever
    except Exception as e:
        logger.error(f"Error creating multi-query retriever: {str(e)}")
        raise

from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging

logger = logging.getLogger(__name__)

def split_document(document, chunk_size=1000, chunk_overlap=200):
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = text_splitter.split_documents(document)
        logger.debug(f"Document split into {len(chunks)} chunks")
        return chunks
    except Exception as e:
        logger.error(f"Error splitting document: {str(e)}")
        raise

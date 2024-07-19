__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb
from chromadb.config import Settings
from langchain_openai import OpenAIEmbeddings
import logging
from config import OPENAI_API_KEY
import uuid
from chromadb.utils import embedding_functions

logger = logging.getLogger(__name__)

class ChromaOpenAIEmbeddings:
    def __init__(self, openai_embeddings):
        self.openai_embeddings = openai_embeddings

    def __call__(self, input):
        return self.openai_embeddings.embed_documents(input)

class ChromaClientSingleton:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = chromadb.PersistentClient(
                path="./collections",
                settings=Settings(anonymized_telemetry=False)
            )
        return cls._instance

def get_chroma_client():
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name="text-embedding-ada-002"
    )
    client = ChromaClientSingleton.get_instance()
    return client, openai_ef


client = get_chroma_client()
active_collection = None

openai_embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=OPENAI_API_KEY)
embeddings = ChromaOpenAIEmbeddings(openai_embeddings)

def create_collection(name):
    global active_collection
    try:
        client, openai_ef = get_chroma_client()
        active_collection = client.create_collection(name, embedding_function=openai_ef)
        logger.info(f"Collection created: {name}")
    except Exception as e:
        logger.error(f"Error creating collection: {str(e)}")
        raise



def delete_collection(name):
    global active_collection
    try:
        client, _ = get_chroma_client()
        client.delete_collection(name)
        if active_collection and active_collection.name == name:
            active_collection = None
        logger.info(f"Collection deleted: {name}")
    except Exception as e:
        logger.error(f"Error deleting collection: {str(e)}")
        raise

def list_collections():
    try:
        client, _ = get_chroma_client()
        return [col.name for col in client.list_collections()]
    except Exception as e:
        logger.error(f"Error listing collections: {str(e)}")
        raise


def select_collection(name):
    global active_collection
    try:
        client, openai_ef = get_chroma_client()
        active_collection = client.get_collection(name, embedding_function=openai_ef)
        logger.info(f"Collection selected: {name}")
    except Exception as e:
        logger.error(f"Error selecting collection: {str(e)}")
        raise

def get_vector_store():
    global active_collection
    if active_collection is None:
        client, openai_ef = get_chroma_client()
        active_collection = client.get_or_create_collection("default_collection", embedding_function=openai_ef)
        logger.info("Created default collection")
    return active_collection


def add_texts_to_collection(texts, metadatas):
    try:
        active_collection = get_vector_store()
        ids = [f"{metadata['source']}_{str(uuid.uuid4())}" for metadata in metadatas]
        active_collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        logger.info(f"Added {len(texts)} texts to collection {active_collection.name}")
    except Exception as e:
        logger.error(f"Error adding texts to collection: {str(e)}")
        raise



def get_indexed_documents():
    global active_collection
    if active_collection is None:
        logger.error("No active collection selected")
        raise ValueError("No active collection selected")
    
    try:
        results = active_collection.get()
        unique_documents = set()
        for metadata in results['metadatas']:
            source = metadata.get('filename') or metadata.get('source_url') or metadata.get('source', 'Unknown')
            unique_documents.add(source)
        
        logger.info(f"Retrieved {len(unique_documents)} unique indexed documents")
        return list(unique_documents)
    except Exception as e:
        logger.error(f"Error retrieving indexed documents: {str(e)}")
        raise


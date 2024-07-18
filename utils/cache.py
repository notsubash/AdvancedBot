import hashlib
import json
from typing import Dict, Any
import openai
import logging

logger = logging.getLogger(__name__)

class QueryCache:
    def __init__(self):
        self.cache: Dict[str, Any] = {}

    def get_embedding(self, text: str) -> list:
        try:
            response = openai.Embedding.create(input=[text], model="text-embedding-ada-002")
            return response['data'][0]['embedding']
        except Exception as e:
            logger.error(f"Error getting embedding: {str(e)}")
            raise

    def get_cache_key(self, query: str) -> str:
        embedding = self.get_embedding(query)
        return hashlib.md5(json.dumps(embedding).encode()).hexdigest()

    def get(self, query: str) -> Any:
        key = self.get_cache_key(query)
        return self.cache.get(key)

    def set(self, query: str, response: Any):
        key = self.get_cache_key(query)
        self.cache[key] = response
        logger.info(f"Cached response for query: {query[:50]}...")

query_cache = QueryCache()

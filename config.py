import os
from dotenv import load_dotenv
from langsmith import Client
import logging

load_dotenv()

langsmith_client = Client()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COLLECTIONS_FOLDER = os.getenv("COLLECTIONS_FOLDER", "./collections")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "Bot"
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION = os.getenv("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
TOKENIZERS_PARALLELISM = "false"


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY is not set in the environment variables.")

logger.info(f"Collections folder path set to: {COLLECTIONS_FOLDER}")


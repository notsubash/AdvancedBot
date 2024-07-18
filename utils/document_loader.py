import re
import os
import tempfile
from langchain_community.document_loaders import TextLoader, PDFMinerLoader, UnstructuredWordDocumentLoader, WebBaseLoader, CSVLoader, UnstructuredMarkdownLoader
from streamlit.runtime.uploaded_file_manager import UploadedFile
import logging
import pandas as pd
from langchain.schema import Document
import json

logger = logging.getLogger(__name__)

def load_document(source):
    try:
        if isinstance(source, UploadedFile):
            metadata = {
                "filename": source.name,
                "file_type": source.type,
                "size": source.size
            }
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(source.name)[1]) as tmp_file:
                tmp_file.write(source.getvalue())
                tmp_file_path = tmp_file.name
            
            if source.name.endswith(".txt"):
                loader = TextLoader(tmp_file_path)
            elif source.name.endswith(".pdf"):
                loader = PDFMinerLoader(tmp_file_path)
            elif source.name.endswith(".docx"):
                loader = UnstructuredWordDocumentLoader(tmp_file_path)
            elif source.name.endswith(".csv"):
                df = pd.read_csv(tmp_file_path)
                documents = []
                for _, row in df.iterrows():
                    content = " ".join(f"{col}: {val}" for col, val in row.items() if pd.notna(val))
                    doc = Document(page_content=content, metadata={**metadata, "row_index": _})
                    documents.append(doc)
                return documents
            elif source.name.endswith(".md"):
                loader = UnstructuredMarkdownLoader(tmp_file_path)
            elif source.name.endswith(".json"):
                with open(tmp_file_path, 'r') as json_file:
                    data = json.load(json_file)
                documents = []
                for item in data:
                    content = json.dumps(item, indent=2)
                    doc = Document(page_content=content, metadata={**metadata, "source": "json"})
                    documents.append(doc)
                return documents
            else:
                raise ValueError("Unsupported file type")
            
            result = loader.load()
            for doc in result:
                doc.metadata.update(metadata)
                if source.name.endswith('.md'):
                    # Extract URL source from markdown content
                    url_match = re.search(r'URL Source: (https?://\S+)', doc.page_content)
                    if url_match:
                        doc.metadata['url_source'] = url_match.group(1)
            os.unlink(tmp_file_path)
            return result
        elif isinstance(source, str) and source.startswith("http"):
            loader = WebBaseLoader(source)
            result = loader.load()
            for doc in result:
                doc.metadata["source_url"] = source
            return result
        else:
            raise ValueError("Unsupported source type")
    except Exception as e:
        logger.error(f"Error loading document: {str(e)}")
        raise

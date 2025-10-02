# config.py

class RAGConfig:
    # Embedding model configuration
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_MODEL_KWARGS = {'device': 'cpu'}
    
    # Text splitting configuration
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # Vector store configuration
    SEARCH_TYPE = "similarity"
    SEARCH_K = 4
    
    # LLM configuration
    GROQ_MODEL = "llama-3.1-8b-instant"
    TEMPERATURE = 0.1
    
    # File processing
    SUPPORTED_EXTENSIONS = ['.pdf']
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

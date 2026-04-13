import os
 
# PATHS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_MODEL_DIR = os.path.join(BASE_DIR, "models", "all-MiniLM-L6-v2")
PERSIST_BASE_DIR = os.path.join(BASE_DIR, "chroma_db_langchain")
 
# CHUNKING

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 300
 
# RETRIEVAL
DEFAULT_K = 8
MAX_K = 20
K_PER_FILE = 6
 

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_KEEP_ALIVE = -1  # keep model in RAM indefinitely
DEFAULT_MODELS = ["qwen:7b", "gemma3n:e2b", "qwen3.5:2b", "nemotron-3-nano:4b", "qwen3.5:0.8b", "mistral"]
 
USE_STEMMING = True
 
CHAT_HISTORY_WINDOW = 6  # number of past messages to include in prompt
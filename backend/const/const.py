from adalflow import OllamaClient

class Const:
    CODE_EXTENSIONS = [".py", ".js", ".ts", ".java", ".cpp", ".c", ".h", ".hpp", ".go", ".rs",
                       ".jsx", ".tsx", ".html", ".css", ".php", ".swift", ".cs"]
    DOC_EXTENSIONS = [".md", ".txt", ".rst", ".json", ".yaml", ".yml"]
    DIR_SKIP_LIST = ["node_modules", "venv", "__pycache__", ".git", "dist", "build", ".venv"]

    EMBEDDING_CONFIG = {
        "client": OllamaClient(),
        "model_kwargs": {
        "model": "nomic-embed-text"
        }
    }

    TEXT_SPLIT_CONFIG = {
        "split_by": "word",
        "chunk_size": 800,  
        "chunk_overlap": 150  
    }

    GENERATION_MODEL = "qwen3-coder:30b"
    EMBEDDING_MODEL = "nomic-embed-text"
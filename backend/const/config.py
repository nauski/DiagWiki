import os
import subprocess
import json
import logging
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import Optional, List

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class ClaudeResponse:
    """Response from Claude Code CLI."""
    content: str
    success: bool
    error: Optional[str] = None


class ClaudeCodeClient:
    """
    Client that uses Claude Code CLI for LLM generation.

    Uses the `claude` CLI with `-p` flag (print mode) to generate text.
    This leverages your existing Claude Code subscription without additional API costs.
    """

    def __init__(self, timeout: float = 300.0):
        """
        Initialize the Claude Code CLI client.

        Args:
            timeout: Timeout in seconds for CLI calls (default 5 minutes)
        """
        self.timeout = timeout
        self._verify_claude_available()

    def _verify_claude_available(self):
        """Verify that claude CLI is available."""
        try:
            result = subprocess.run(
                ["claude", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                logger.info(f"Claude Code CLI available: {result.stdout.strip()}")
            else:
                logger.warning("Claude Code CLI returned non-zero exit code")
        except FileNotFoundError:
            raise RuntimeError(
                "Claude Code CLI not found. Please install it with: npm install -g @anthropic-ai/claude-code"
            )
        except Exception as e:
            logger.warning(f"Could not verify Claude CLI: {e}")

    def generate(self, prompt: str, json_output: bool = False) -> ClaudeResponse:
        """
        Generate text using Claude Code CLI.

        Args:
            prompt: The prompt to send to Claude
            json_output: If True, instruct Claude to output JSON

        Returns:
            ClaudeResponse with content or error
        """
        try:
            # Build the command
            cmd = ["claude", "-p", prompt]

            # If JSON output is requested, add instruction to the prompt
            if json_output:
                prompt = f"{prompt}\n\nIMPORTANT: Respond ONLY with valid JSON, no other text."
                cmd = ["claude", "-p", prompt]

            logger.debug(f"Calling Claude CLI with prompt length: {len(prompt)}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )

            if result.returncode == 0:
                content = result.stdout.strip()
                logger.debug(f"Claude CLI response length: {len(content)}")
                return ClaudeResponse(content=content, success=True)
            else:
                error_msg = result.stderr.strip() or "Unknown error"
                logger.error(f"Claude CLI error: {error_msg}")
                return ClaudeResponse(content="", success=False, error=error_msg)

        except subprocess.TimeoutExpired:
            logger.error(f"Claude CLI timeout after {self.timeout}s")
            return ClaudeResponse(
                content="",
                success=False,
                error=f"Timeout after {self.timeout} seconds"
            )
        except Exception as e:
            logger.error(f"Claude CLI exception: {e}")
            return ClaudeResponse(content="", success=False, error=str(e))

    def generate_json(self, prompt: str) -> dict:
        """
        Generate JSON output using Claude Code CLI.

        Args:
            prompt: The prompt (should request JSON output)

        Returns:
            Parsed JSON dict or empty dict on error
        """
        response = self.generate(prompt, json_output=True)

        if not response.success:
            logger.error(f"Failed to generate JSON: {response.error}")
            return {}

        try:
            # Try to extract JSON from the response
            content = response.content.strip()

            # Handle markdown code blocks
            if content.startswith("```"):
                lines = content.split("\n")
                # Skip first line (```json or ```)
                if lines[0].startswith("```"):
                    lines = lines[1:]
                # Skip last line (```)
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                content = "\n".join(lines)

            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.debug(f"Raw response: {response.content[:500]}")
            return {}


class SentenceTransformerEmbedder:
    """
    Local embeddings using sentence-transformers.

    Uses CPU-friendly models that don't require a GPU.
    The all-MiniLM-L6-v2 model is fast and produces 384-dimensional embeddings.
    """

    _instance = None
    _model = None

    def __new__(cls, model_name: str = None):
        """Singleton pattern to avoid loading model multiple times."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, model_name: str = None):
        """
        Initialize the sentence transformer embedder.

        Args:
            model_name: Model name from sentence-transformers (default: all-MiniLM-L6-v2)
        """
        if SentenceTransformerEmbedder._model is None:
            model_name = model_name or os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
            logger.info(f"Loading sentence-transformers model: {model_name}")

            try:
                from sentence_transformers import SentenceTransformer
                SentenceTransformerEmbedder._model = SentenceTransformer(model_name)

                # Get embedding dimension
                test_embedding = SentenceTransformerEmbedder._model.encode("test")
                self.dimension = len(test_embedding)
                logger.info(f"Embedding model loaded. Dimension: {self.dimension}")
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. Run: pip install sentence-transformers"
                )

    def embed(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            List of floats (embedding vector)
        """
        embedding = SentenceTransformerEmbedder._model.encode(text)
        return embedding.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        embeddings = SentenceTransformerEmbedder._model.encode(texts)
        return [emb.tolist() for emb in embeddings]

    def get_dimension(self) -> int:
        """Get the embedding dimension."""
        return self.dimension


class Config:
    """Application configuration from environment variables"""

    # ============================================================
    # Application Settings
    # ============================================================
    APP_NAME: str = "DiagWiki"
    APP_VERSION: str = "0.2.0"  # Updated for Claude Code integration
    ENVIRONMENT: str = os.environ.get("NODE_ENV", "development")
    PORT: int = int(os.environ.get("PORT", "8001"))
    LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO")

    # ============================================================
    # LLM Configuration - Using Claude Code CLI
    # ============================================================
    # Timeout (seconds) for Claude CLI calls
    LLM_TIMEOUT: float = float(os.environ.get("LLM_TIMEOUT", "300.0"))

    # Model identifier (for display purposes - actual model is determined by Claude Code)
    GENERATION_MODEL: str = "claude-code-cli"

    # Embedding model (sentence-transformers)
    # all-MiniLM-L6-v2: Fast, 384 dimensions, good quality
    # all-mpnet-base-v2: Better quality, 768 dimensions, slower
    EMBEDDING_MODEL: str = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # Embedding dimension (set by model, will be updated on first use)
    EMBEDDING_DIMENSION: int = int(os.environ.get("EMBEDDING_DIMENSION", "384"))

    # ============================================================
    # Text Splitting Configuration
    # ============================================================
    TEXT_SPLIT_BY: str = os.environ.get("TEXT_SPLIT_BY", "token")
    TEXT_CHUNK_SIZE: int = int(os.environ.get("TEXT_CHUNK_SIZE", "1000"))
    TEXT_CHUNK_OVERLAP: int = int(os.environ.get("TEXT_CHUNK_OVERLAP", "50"))

    # ============================================================
    # Localization
    # ============================================================
    DEFAULT_LANGUAGE: str = os.environ.get("DEFAULT_LANGUAGE", "en")

    # ============================================================
    # RAG Configuration
    # ============================================================
    # Maximum characters for RAG context to prevent LLM overflow
    MAX_RAG_CONTEXT_CHARS: int = int(os.environ.get("MAX_RAG_CONTEXT_CHARS", "100000"))

    # Maximum number of source files to include in RAG context
    MAX_SOURCES: int = int(os.environ.get("MAX_SOURCES", "40"))

    # Maximum characters per file when reading manual references
    MAX_FILE_CHARS: int = int(os.environ.get("MAX_FILE_CHARS", "50000"))

    # Default top_k for RAG queries
    RAG_TOP_K: int = int(os.environ.get("RAG_TOP_K", "40"))

    # Special top_k for section identification iterations
    RAG_SECTION_ITERATION_TOP_K: int = int(os.environ.get("RAG_SECTION_ITERATION_TOP_K", "80"))

    # Maximum tokens for document chunking
    MAX_TOKEN_LIMIT: int = int(os.environ.get("MAX_TOKEN_LIMIT", "8192"))

    # Maximum tokens for embedding (to prevent overflow)
    MAX_EMBEDDING_TOKENS: int = int(os.environ.get("MAX_EMBEDDING_TOKENS", "6000"))

    # Preview length for file sources
    SOURCE_PREVIEW_LENGTH: int = int(os.environ.get("SOURCE_PREVIEW_LENGTH", "600"))

    # ============================================================
    # LLM Generation Parameters
    # ============================================================
    # Default temperature for creative generation (diagrams, wiki)
    DEFAULT_TEMPERATURE: float = float(os.environ.get("DEFAULT_TEMPERATURE", "0.7"))

    # Lower temperature for focused tasks (title generation)
    FOCUSED_TEMPERATURE: float = float(os.environ.get("FOCUSED_TEMPERATURE", "0.3"))

    # Context window size - Claude has large context
    LARGE_CONTEXT_WINDOW: int = int(os.environ.get("LARGE_CONTEXT_WINDOW", "100000"))

    # ============================================================
    # API Configuration
    # ============================================================
    # Thread pool size for async operations
    MAX_WORKERS: int = int(os.environ.get("MAX_WORKERS", "4"))

    # ============================================================
    # Computed Configuration
    # ============================================================
    @classmethod
    def get_llm_client(cls) -> ClaudeCodeClient:
        """
        Get a Claude Code CLI client configured with proper timeout.

        Returns:
            ClaudeCodeClient instance
        """
        return ClaudeCodeClient(timeout=cls.LLM_TIMEOUT)

    @classmethod
    def get_embedder(cls) -> SentenceTransformerEmbedder:
        """
        Get the sentence-transformers embedder.

        Returns:
            SentenceTransformerEmbedder instance
        """
        return SentenceTransformerEmbedder(cls.EMBEDDING_MODEL)

    @classmethod
    def get_embedding_dimension(cls) -> int:
        """Get the embedding dimension from the loaded model."""
        try:
            embedder = cls.get_embedder()
            return embedder.get_dimension()
        except Exception:
            return cls.EMBEDDING_DIMENSION

    @classmethod
    def get_text_split_config(cls):
        """Get text splitting configuration dict"""
        return {
            "split_by": cls.TEXT_SPLIT_BY,
            "chunk_size": cls.TEXT_CHUNK_SIZE,
            "chunk_overlap": cls.TEXT_CHUNK_OVERLAP,
        }

    @classmethod
    def is_development(cls) -> bool:
        return cls.ENVIRONMENT != "production"


# Export commonly used values for backward compatibility
APP_NAME = Config.APP_NAME
APP_VERSION = Config.APP_VERSION
GENERATION_MODEL = Config.GENERATION_MODEL
EMBEDDING_MODEL = Config.EMBEDDING_MODEL

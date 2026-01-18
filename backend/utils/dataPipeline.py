import os
import hashlib
import re
import logging
from const.config import Config
import adalflow as adal
from adalflow.core.component import DataComponent
from adalflow.core.types import Document
from copy import deepcopy
from tqdm import tqdm
from typing import Sequence, List
from adalflow.components.data_process import TextSplitter
from adalflow.core.db import LocalDB

logger = logging.getLogger(__name__)


def generate_db_name(folder_path: str) -> str:
    """Generate a deterministic database name from folder path.

    Creates a name combining:
    - Sanitized folder name (human-readable)
    - Short hash of full path (ensures uniqueness)

    Args:
        folder_path: Absolute path to the folder

    Returns:
        Database name like 'DiagWiki_a1b2c3d4'
    """
    # Normalize path (resolve symlinks, remove trailing slashes)
    normalized_path = os.path.normpath(os.path.abspath(folder_path))

    # Get folder name
    folder_name = os.path.basename(normalized_path)

    # Sanitize folder name: keep only alphanumeric, hyphens, underscores
    sanitized_name = re.sub(r'[^a-zA-Z0-9_-]', '_', folder_name)
    sanitized_name = re.sub(r'_+', '_', sanitized_name)  # Collapse multiple underscores
    sanitized_name = sanitized_name.strip('_')  # Remove leading/trailing underscores

    # Generate short hash of full path for uniqueness (first 8 chars)
    path_hash = hashlib.sha256(normalized_path.encode('utf-8')).hexdigest()[:8]

    # Combine: foldername_hash
    db_name = f"{sanitized_name}_{path_hash}"

    logger.info(f"Generated db_name '{db_name}' for folder '{normalized_path}'")
    return db_name


class LocalEmbeddingProcessor(DataComponent):
    """
    Process documents for local embeddings using sentence-transformers.

    Unlike the Ollama version, this processor can handle batch embedding
    which is much faster for processing many documents.
    """

    def __init__(self) -> None:
        super().__init__()
        self.embedder = Config.get_embedder()
        self.dimension = self.embedder.get_dimension()
        logger.info(f"LocalEmbeddingProcessor initialized with dimension: {self.dimension}")

    def __call__(self, documents: Sequence[Document]) -> Sequence[Document]:
        from utils.repoUtil import RepoUtil
        output = deepcopy(documents)
        logger.info(f"Processing {len(output)} documents for embeddings")

        successful_docs = []
        skipped_large_docs = 0

        # Process in batches for efficiency
        batch_size = 32
        texts_to_embed = []
        docs_to_embed = []

        for i, doc in enumerate(output):
            # Check token count before attempting to embed
            token_count = RepoUtil.token_count(doc.text)
            file_path = getattr(doc, 'meta_data', {}).get('file_path', f'document_{i}')

            logger.debug(f"Processing '{file_path}': {token_count} tokens, {len(doc.text)} chars")

            if token_count > Config.MAX_EMBEDDING_TOKENS:
                logger.warning(
                    f"Document '{file_path}' has {token_count} tokens which exceeds "
                    f"embedding model context limit ({Config.MAX_EMBEDDING_TOKENS}), skipping"
                )
                skipped_large_docs += 1
                continue

            texts_to_embed.append(doc.text)
            docs_to_embed.append((i, doc))

        # Embed in batches
        logger.info(f"Embedding {len(texts_to_embed)} documents in batches of {batch_size}")

        for batch_start in tqdm(range(0, len(texts_to_embed), batch_size),
                                desc="Embedding documents"):
            batch_end = min(batch_start + batch_size, len(texts_to_embed))
            batch_texts = texts_to_embed[batch_start:batch_end]
            batch_docs = docs_to_embed[batch_start:batch_end]

            try:
                # Batch embed
                embeddings = self.embedder.embed_batch(batch_texts)

                for (idx, doc), embedding in zip(batch_docs, embeddings):
                    # Validate embedding size
                    if len(embedding) != self.dimension:
                        file_path = getattr(doc, 'meta_data', {}).get('file_path', f'document_{idx}')
                        logger.warning(
                            f"Document '{file_path}' has inconsistent embedding size "
                            f"{len(embedding)} != {self.dimension}, skipping"
                        )
                        continue

                    # Assign the embedding to the document
                    output[idx].vector = embedding
                    successful_docs.append(output[idx])

            except Exception as e:
                logger.error(f"Error embedding batch starting at {batch_start}: {e}")
                # Try individual embedding as fallback
                for (idx, doc), text in zip(batch_docs, batch_texts):
                    try:
                        embedding = self.embedder.embed(text)
                        output[idx].vector = embedding
                        successful_docs.append(output[idx])
                    except Exception as inner_e:
                        file_path = getattr(doc, 'meta_data', {}).get('file_path', f'document_{idx}')
                        logger.error(f"Error embedding document '{file_path}': {inner_e}")

        if skipped_large_docs > 0:
            logger.info(
                f"Skipped {skipped_large_docs} documents that exceeded token limit "
                f"({Config.MAX_EMBEDDING_TOKENS} tokens)"
            )
        logger.info(
            f"Successfully processed {len(successful_docs)}/{len(output)} documents with embeddings"
        )
        return successful_docs


class DataPipeline:
    """Pipeline for processing documents with text splitting and embedding."""

    def __init__(self, db_name: str = None, embedder_model: str = None, text_splitter_config: dict = None):
        """Initialize DataPipeline with optional custom configuration.

        Args:
            db_name: Optional database name (for process_folder)
            embedder_model: Optional embedder model override (ignored - uses Config)
            text_splitter_config: Optional text splitter config override
        """
        self.db_name = db_name

        # Use local embedding processor
        self.embedder_transformer = LocalEmbeddingProcessor()

        split_config = text_splitter_config if text_splitter_config else Config.get_text_split_config()
        self.splitter = TextSplitter(**split_config)

        self.data_transformer = adal.Sequential(
            self.splitter,
            self.embedder_transformer
        )

    def process_folder(self, folder_path: str, allowed_extensions: list = None, data_dir: str = None):
        """Process all documents in a folder: collect, transform, and save to database.

        This is the main entry point that combines:
        1. Document collection from folder
        2. Text splitting and embedding
        3. Saving to LocalDB

        Args:
            folder_path: Path to folder containing documents
            allowed_extensions: Optional list of file extensions to filter (e.g., ['.py', '.md'])
            data_dir: Optional base directory for databases (defaults to ./data)

        Returns:
            dict: Statistics about the transformation process
        """
        from utils.repoUtil import RepoUtil

        logger.info(f"Processing folder: {folder_path}")

        # Validate folder
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        if not os.path.isdir(folder_path):
            raise ValueError(f"Path is not a directory: {folder_path}")

        # Collect documents
        logger.info("Collecting documents from folder...")
        documents = RepoUtil.collect_documents(folder_path, allowed_extensions)

        if not documents:
            raise ValueError("No documents found in folder")

        logger.info(f"Collected {len(documents)} documents")

        # Determine database directory
        if data_dir is None:
            data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

        db_dir = os.path.join(data_dir, self.db_name)

        # Transform and save
        logger.info("Transforming documents (split + embed)...")
        db = self.transform_and_save(documents, db_dir)
        logger.info("Transformation completed")

        # Get statistics
        transformed_docs = db.get_transformed_data(key="split_and_embed")
        logger.info(f"Created {len(transformed_docs)} document chunks")

        # Calculate statistics
        stats = {
            "status": "success",
            "folder_path": folder_path,
            "database_name": self.db_name,
            "database_path": os.path.join(db_dir, "db.pkl"),
            "original_document_count": len(documents),
            "transformed_chunk_count": len(transformed_docs),
            "chunks_with_embeddings": 0,
            "embedding_dimension": Config.get_embedding_dimension(),
            "embedding_model": Config.EMBEDDING_MODEL,
            "embedding_sizes": {},
            "file_types": {}
        }

        # Analyze chunks
        for doc in transformed_docs:
            if hasattr(doc, 'vector') and doc.vector is not None:
                stats["chunks_with_embeddings"] += 1
                embedding_size = len(doc.vector)
                stats["embedding_sizes"][embedding_size] = \
                    stats["embedding_sizes"].get(embedding_size, 0) + 1

            # Count file types
            if hasattr(doc, 'meta_data') and 'extension' in doc.meta_data:
                ext = doc.meta_data['extension']
                stats["file_types"][ext] = stats["file_types"].get(ext, 0) + 1

        logger.info(
            f"Process completed: {stats['chunks_with_embeddings']}/{stats['transformed_chunk_count']} "
            f"chunks with embeddings"
        )
        return stats

    def transform_and_save(self, documents: Sequence[Document], persist_dir: str) -> LocalDB:
        """Transform documents and save to the specified directory"""
        db = LocalDB()
        db.register_transformer(transformer=self.data_transformer, key="split_and_embed")
        db.load(documents)
        db.transform(key="split_and_embed")

        # Create directory and save to db.pkl file
        os.makedirs(persist_dir, exist_ok=True)
        db_file = os.path.join(persist_dir, "db.pkl")
        db.save_state(filepath=db_file)
        logger.info(f"Saved database to {db_file}")
        return db

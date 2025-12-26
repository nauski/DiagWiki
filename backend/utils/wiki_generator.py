"""
Wiki generation utilities.

This module handles the core logic for generating wiki structures and pages
using RAG (Retrieval-Augmented Generation) with hybrid retrieval.
"""

import os
import json
import logging
from typing import Optional, Dict, List
from datetime import datetime

from utils.rag import RAG
from utils.repoUtil import RepoUtil
from utils.dataPipeline import DataPipeline, generate_db_name
from const.const import Const
from const.prompts import (
    build_wiki_structure_prompt,
    build_wiki_page_prompt,
    STRUCTURE_ANALYSIS_QUERIES,
    build_page_analysis_queries
)
from adalflow.components.model_client.ollama_client import OllamaClient
from adalflow.core.types import ModelType

logger = logging.getLogger(__name__)


class WikiCache:
    """Handles caching of generated wiki content."""
    
    def __init__(self, db_path: str):
        """
        Initialize wiki cache.
        
        Args:
            db_path: Path to the database directory
        """
        self.db_path = db_path
        self.wiki_dir = os.path.join(db_path, "wiki")
        self.structure_file = os.path.join(self.wiki_dir, "structure.xml")
        self.pages_dir = os.path.join(self.wiki_dir, "pages")
        self.metadata_file = os.path.join(self.wiki_dir, "metadata.json")
        
        # Create directories if they don't exist
        os.makedirs(self.pages_dir, exist_ok=True)
    
    def save_structure(self, structure: str) -> str:
        """Save wiki structure to cache."""
        with open(self.structure_file, 'w', encoding='utf-8') as f:
            f.write(structure)
        logger.info(f"Saved wiki structure to: {self.structure_file}")
        return self.structure_file
    
    def load_structure(self) -> Optional[str]:
        """Load wiki structure from cache."""
        if os.path.exists(self.structure_file):
            with open(self.structure_file, 'r', encoding='utf-8') as f:
                return f.read()
        return None
    
    def save_page(self, page_id: str, page_title: str, content: str, metadata: dict = None) -> str:
        """
        Save wiki page to cache.
        
        Args:
            page_id: Unique identifier for the page
            page_title: Title of the page
            content: Markdown content
            metadata: Optional metadata dict
        
        Returns:
            Path to saved file
        """
        # Sanitize filename
        safe_filename = page_id.replace('/', '_').replace('\\', '_')
        page_file = os.path.join(self.pages_dir, f"{safe_filename}.md")
        
        with open(page_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # Save metadata
        if metadata:
            self._update_metadata(page_id, {
                'title': page_title,
                'file': page_file,
                'generated_at': datetime.now().isoformat(),
                **metadata
            })
        
        logger.info(f"Saved wiki page to: {page_file}")
        return page_file
    
    def load_page(self, page_id: str) -> Optional[str]:
        """Load wiki page from cache."""
        safe_filename = page_id.replace('/', '_').replace('\\', '_')
        page_file = os.path.join(self.pages_dir, f"{safe_filename}.md")
        
        if os.path.exists(page_file):
            with open(page_file, 'r', encoding='utf-8') as f:
                return f.read()
        return None
    
    def _update_metadata(self, page_id: str, page_metadata: dict):
        """Update metadata file with page information."""
        metadata = {}
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        
        if 'pages' not in metadata:
            metadata['pages'] = {}
        
        metadata['pages'][page_id] = page_metadata
        
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
    
    def get_metadata(self) -> dict:
        """Get all cached metadata."""
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}


class WikiGenerator:
    """Main class for wiki generation using RAG."""
    
    def __init__(self, root_path: str, data_dir: str):
        """
        Initialize wiki generator.
        
        Args:
            root_path: Root path to the codebase
            data_dir: Base data directory for databases
        """
        self.root_path = root_path
        self.data_dir = data_dir
        self.db_name = generate_db_name(root_path)
        self.db_path = os.path.join(data_dir, self.db_name)
        self.cache = WikiCache(self.db_path)
        self.rag = None
    
    def ensure_database(self):
        """Ensure database exists, create if needed."""
        if not os.path.exists(self.db_path):
            logger.info(f"Database not found for {self.root_path}, creating...")
            pipeline = DataPipeline(
                db_name=self.db_name,
                embedder_model=Const.EMBEDDING_MODEL,
                text_splitter_config=Const.TEXT_SPLIT_CONFIG
            )
            result = pipeline.process_folder(
                folder_path=self.root_path,
                data_dir=self.data_dir
            )
            logger.info(f"Database created: {result}")
    
    def initialize_rag(self):
        """Initialize RAG system."""
        if self.rag is None:
            self.rag = RAG()
            self.rag.load_database(self.db_path)
            logger.info(f"RAG initialized with {len(self.rag.transformed_docs)} documents")
    
    def generate_structure(self, language: str = "en", comprehensive: bool = False, use_cache: bool = True) -> Dict:
        """
        Generate wiki structure using RAG analysis.
        
        Args:
            language: Target language code
            comprehensive: Whether to create comprehensive wiki
            use_cache: Whether to use cached structure if available
        
        Returns:
            Dict with status and structure
        """
        # Check cache first
        if use_cache:
            cached_structure = self.cache.load_structure()
            if cached_structure:
                logger.info("Using cached wiki structure")
                return {
                    "status": "success",
                    "root_path": self.root_path,
                    "comprehensive": comprehensive,
                    "language": language,
                    "wiki_structure": cached_structure,
                    "cached": True
                }
        
        # Ensure database and RAG are ready
        self.ensure_database()
        self.initialize_rag()
        
        # Generate file tree
        file_tree = RepoUtil.build_tree(self.root_path)
        logger.info("File tree generated")
        
        # Read README if exists
        readme_content = ""
        readme_paths = ["README.md", "README.MD", "readme.md", "README.txt", "README"]
        for readme_name in readme_paths:
            readme_path = os.path.join(self.root_path, readme_name)
            if os.path.exists(readme_path):
                try:
                    with open(readme_path, 'r', encoding='utf-8') as f:
                        readme_content = f.read()
                    logger.info(f"README found: {readme_name}")
                    break
                except Exception as e:
                    logger.warning(f"Error reading README: {e}")
        
        # Perform RAG analysis
        logger.info("Querying RAG for codebase analysis...")
        rag_insights = []
        
        for query in STRUCTURE_ANALYSIS_QUERIES:
            try:
                result = self.rag.call(
                    query=query,
                    top_k=5,
                    use_reranking=True
                )
                rag_insights.append({
                    "query": query,
                    "answer": result.answer,
                    "sources": [doc.text[:300] for doc in result.documents[:3]]
                })
                logger.info(f"RAG query completed: {query[:50]}...")
            except Exception as e:
                logger.warning(f"RAG query failed: {e}")
        
        # Build prompt and call LLM
        folder_name = os.path.basename(self.root_path)
        prompt = build_wiki_structure_prompt(
            folder_name=folder_name,
            file_tree=file_tree,
            readme_content=readme_content,
            rag_insights=rag_insights,
            language=language,
            comprehensive=comprehensive
        )
        
        # Call LLM
        model = OllamaClient()
        model_kwargs = {
            "model": Const.GENERATION_MODEL,
            "options": {
                "temperature": 0.7,
                "num_ctx": 8192
            }
        }
        
        api_kwargs = model.convert_inputs_to_api_kwargs(
            input=prompt,
            model_kwargs=model_kwargs,
            model_type=ModelType.LLM
        )
        
        logger.info("Calling LLM to generate wiki structure...")
        response = model.call(api_kwargs=api_kwargs, model_type=ModelType.LLM)
        
        # Extract content from Ollama ChatResponse
        if hasattr(response, 'message') and hasattr(response.message, 'content'):
            wiki_structure = response.message.content
        elif hasattr(response, 'data'):
            wiki_structure = response.data
        elif isinstance(response, dict):
            wiki_structure = response.get('message', {}).get('content', '')
        else:
            logger.warning(f"Unexpected response type: {type(response)}")
            wiki_structure = str(response)
        
        logger.info(f"Wiki structure generated ({len(wiki_structure)} chars)")
        
        # Cache the structure
        self.cache.save_structure(wiki_structure)
        
        return {
            "status": "success",
            "root_path": self.root_path,
            "comprehensive": comprehensive,
            "language": language,
            "wiki_structure": wiki_structure,
            "cached": False
        }
    
    def generate_page(
        self,
        page_title: str,
        page_description: str,
        relevant_files: List[str],
        language: str = "en",
        page_id: str = None,
        use_cache: bool = True
    ) -> Dict:
        """
        Generate wiki page using RAG retrieval.
        
        Args:
            page_title: Title of the page
            page_description: Description of what the page should cover
            relevant_files: List of relevant file paths (hints)
            language: Target language code
            page_id: Optional page ID for caching
            use_cache: Whether to use cached page if available
        
        Returns:
            Dict with status and content
        """
        # Generate page_id if not provided
        if page_id is None:
            page_id = page_title.lower().replace(' ', '_')
        
        # Check cache first
        if use_cache:
            cached_page = self.cache.load_page(page_id)
            if cached_page:
                logger.info(f"Using cached wiki page: {page_title}")
                return {
                    "status": "success",
                    "page_title": page_title,
                    "language": language,
                    "content": cached_page,
                    "cached": True
                }
        
        # Ensure RAG is initialized
        self.initialize_rag()
        
        # Generate RAG queries
        rag_queries = build_page_analysis_queries(page_title, page_description)
        
        # Perform RAG queries
        logger.info(f"Performing {len(rag_queries)} RAG queries for: {page_title}")
        rag_results = []
        all_retrieved_docs = []
        
        for query in rag_queries:
            try:
                result = self.rag.call(
                    query=query,
                    top_k=8,
                    use_reranking=True
                )
                rag_results.append({
                    "query": query,
                    "answer": result.answer,
                    "rationale": result.rationale
                })
                all_retrieved_docs.extend(result.documents)
                logger.info(f"RAG query completed: {query[:60]}...")
            except Exception as e:
                logger.warning(f"RAG query failed for '{query[:50]}...': {e}")
        
        # Deduplicate documents
        seen_paths = {}
        unique_docs = []
        for doc in all_retrieved_docs:
            file_path = doc.meta_data.get('file_path', 'unknown') if hasattr(doc, 'meta_data') else 'unknown'
            if file_path not in seen_paths:
                seen_paths[file_path] = doc
                unique_docs.append(doc)
        
        logger.info(f"Retrieved {len(unique_docs)} unique documents from {len(all_retrieved_docs)} total results")
        
        # Load explicit files
        file_contents = []
        for file_path in relevant_files[:5]:
            full_path = os.path.join(self.root_path, file_path) if not os.path.isabs(file_path) else file_path
            if os.path.exists(full_path) and os.path.isfile(full_path):
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    file_contents.append({
                        "path": file_path,
                        "content": content[:3000]
                    })
                    logger.info(f"Loaded explicit file: {file_path}")
                except Exception as e:
                    logger.warning(f"Error reading file {file_path}: {e}")
        
        # Build context strings
        rag_context = "\n\n".join([
            f"Query: {r['query']}\nAnswer: {r['answer']}\nRationale: {r['rationale']}"
            for r in rag_results
        ])
        
        retrieved_sources = "\n\n".join([
            f"Source {i+1} ({doc.meta_data.get('file_path', 'unknown') if hasattr(doc, 'meta_data') else 'unknown'}):\n{doc.text[:800]}"
            for i, doc in enumerate(unique_docs[:15])
        ])
        
        # Build prompt
        prompt = build_wiki_page_prompt(
            page_title=page_title,
            page_description=page_description,
            rag_context=rag_context,
            retrieved_sources=retrieved_sources,
            file_contents=file_contents,
            unique_docs=unique_docs,
            language=language
        )
        
        # Call LLM
        model = OllamaClient()
        model_kwargs = {
            "model": Const.GENERATION_MODEL,
            "options": {
                "temperature": 0.7,
                "num_ctx": 16384
            }
        }
        
        api_kwargs = model.convert_inputs_to_api_kwargs(
            input=prompt,
            model_kwargs=model_kwargs,
            model_type=ModelType.LLM
        )
        
        logger.info(f"Calling LLM to generate wiki page for: {page_title}")
        response = model.call(api_kwargs=api_kwargs, model_type=ModelType.LLM)
        
        # Extract content from Ollama ChatResponse
        if hasattr(response, 'message') and hasattr(response.message, 'content'):
            page_content = response.message.content
        elif hasattr(response, 'data'):
            page_content = response.data
        elif isinstance(response, dict):
            page_content = response.get('message', {}).get('content', '')
        else:
            logger.warning(f"Unexpected response type: {type(response)}")
            page_content = str(response)
        
        logger.info(f"Wiki page generated ({len(page_content)} chars)")
        
        # Cache the page
        self.cache.save_page(
            page_id=page_id,
            page_title=page_title,
            content=page_content,
            metadata={
                "rag_queries_performed": len(rag_queries),
                "rag_results_count": len(rag_results),
                "unique_sources_retrieved": len(unique_docs),
                "explicit_files_loaded": len(file_contents),
                "language": language
            }
        )
        
        return {
            "status": "success",
            "page_title": page_title,
            "language": language,
            "rag_queries_performed": len(rag_queries),
            "rag_results_count": len(rag_results),
            "unique_sources_retrieved": len(unique_docs),
            "explicit_files_loaded": len(file_contents),
            "content": page_content,
            "cached": False,
            "cache_path": self.cache.pages_dir
        }

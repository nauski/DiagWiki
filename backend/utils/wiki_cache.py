"""
Wiki caching utilities.

This module handles caching of generated wiki content including structure,
pages, diagrams, and metadata. It also manages the wiki RAG database.
"""

import os
import json
import logging
from typing import Optional, Dict
from datetime import datetime

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
        self.diagrams_dir = os.path.join(self.wiki_dir, "diagrams")
        self.metadata_file = os.path.join(self.wiki_dir, "metadata.json")
        self.wiki_db_path = os.path.join(db_path, "wiki_db.pkl")  # Separate DB for wiki content
        
        # Create directories if they don't exist
        os.makedirs(self.pages_dir, exist_ok=True)
        os.makedirs(self.diagrams_dir, exist_ok=True)
    
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
    
    def get_page_metadata(self, page_id: str) -> Optional[Dict]:
        """
        Get metadata for a specific page.
        
        Args:
            page_id: Page identifier
        
        Returns:
            Page metadata dict or None if not found
        """
        all_metadata = self.get_metadata()
        pages = all_metadata.get('pages', {})
        return pages.get(page_id, None)
    
    def add_wiki_content_to_rag(self, content_type: str, content_id: str, content_data: Dict) -> None:
        """
        Add generated wiki content to the wiki RAG database.
        
        This creates searchable documents from diagram explanations, section descriptions,
        and other wiki content for the /askWiki endpoint.
        
        Args:
            content_type: Type of content ('diagram', 'section', 'page')
            content_id: Unique identifier for this content
            content_data: The content data (diagram with explanations, etc.)
        """
        from adalflow.core.types import Document
        from adalflow.core.db import LocalDB
        
        # Build text representation of the content for RAG
        text_parts = []
        metadata = {
            "content_type": content_type,
            "content_id": content_id,
            "source": "wiki"
        }
        
        if content_type == "diagram":
            # Extract meaningful text from diagram
            text_parts.append(f"Section: {content_data.get('section_title', '')}")
            text_parts.append(f"Description: {content_data.get('section_description', '')}")
            
            # Add diagram description
            if 'diagram' in content_data:
                text_parts.append(f"Diagram: {content_data['diagram'].get('description', '')}")
            
            # Add node explanations
            if 'nodes' in content_data:
                text_parts.append("\nComponents:")
                for node_id, node_data in content_data['nodes'].items():
                    text_parts.append(f"- {node_data.get('label', node_id)}: {node_data.get('explanation', '')}")
            
            # Add edge explanations
            if 'edges' in content_data:
                text_parts.append("\nRelationships:")
                for edge_key, edge_data in content_data['edges'].items():
                    text_parts.append(f"- {edge_key}: {edge_data.get('explanation', '')}")
            
            metadata["section_id"] = content_data.get('section_id', '')
            metadata["diagram_type"] = content_data.get('diagram', {}).get('diagram_type', '')
        
        # Create document
        text = "\n".join(text_parts)
        doc = Document(text=text, meta_data=metadata, id=content_id)
        
        # Load or create wiki database
        if os.path.exists(self.wiki_db_path):
            wiki_db = LocalDB.load_state(filepath=self.wiki_db_path)
        else:
            wiki_db = LocalDB()
        
        # Get existing documents or create new list
        try:
            existing_docs = wiki_db.get_transformed_data(key="wiki_content")
        except (ValueError, KeyError):
            existing_docs = []
        
        if existing_docs is None:
            existing_docs = []
        
        # Check if content already exists (by ID), replace if so
        existing_docs = [d for d in existing_docs if d.id != content_id]
        existing_docs.append(doc)
        
        # Store updated documents - directly set transformed_items since these are pre-transformed
        wiki_db.transformed_items["wiki_content"] = existing_docs
        wiki_db.save_state(filepath=self.wiki_db_path)
        
        logger.info(f"Added {content_type} content to wiki RAG: {content_id}")

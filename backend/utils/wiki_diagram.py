"""
Wiki diagram generation utilities.

This module handles the two-step diagram generation process:
1. Identify diagram sections (what to visualize)
2. Generate diagrams with explanations (create visualizations)
"""

import os
import json
import logging
from typing import Dict, List
from adalflow.components.model_client.ollama_client import OllamaClient
from adalflow.core.types import ModelType
from const.const import Const
from const.prompts import (
    build_page_analysis_queries,
    build_diagram_sections_prompt,
    build_single_diagram_prompt
)
from utils.mermaid_parser import parse_mermaid_diagram, validate_mermaid_syntax

logger = logging.getLogger(__name__)


class WikiDiagramGenerator:
    """Handles diagram generation for wiki content."""
    
    def __init__(self, root_path: str, cache, rag_instance):
        """
        Initialize diagram generator.
        
        Args:
            root_path: Root path to the codebase
            cache: WikiCache instance for caching
            rag_instance: Initialized RAG instance for queries
        """
        self.root_path = root_path
        self.cache = cache
        self.rag = rag_instance
    
    def identify_diagram_sections(
        self,
        language: str = "en",
        use_cache: bool = True
    ) -> Dict:
        """
        Step 1: Identify diagram sections for the codebase (Diagram-First Wiki).
        
        This is for a DIAGRAM-FIRST WIKI - diagrams ARE the content, not supplements.
        Analyzes the codebase and identifies diagram sections that together explain it.
        The number of sections is determined by the LLM based on codebase complexity.
        
        Args:
            language: Target language code
            use_cache: Whether to use cached sections if available
        
        Returns:
            Dict with status and identified sections list
        """
        # Use repo name as page_id for caching
        repo_name = os.path.basename(self.root_path)
        page_id = repo_name.lower().replace(' ', '_').replace('/', '_')
        
        # Check cache first
        if use_cache:
            cache_file = os.path.join(self.cache.diagrams_dir, f"{page_id}_sections.json")
            if os.path.exists(cache_file):
                logger.info(f"✅ Using cached diagram sections from: {cache_file}")
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                    cached_data['cached'] = True
                    cached_data['cache_file'] = cache_file
                    return cached_data
        
        # Ensure RAG is initialized
        if self.rag is None:
            raise RuntimeError("RAG not initialized. Call initialize_rag() first.")
        
        # Generate RAG queries
        rag_queries = build_page_analysis_queries(repo_name, "Identify key components and workflows suitable for diagrammatic representation.")
        
        # Perform RAG queries
        logger.info(f"Performing {len(rag_queries)} RAG queries for: {repo_name}")
        rag_results = []
        
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
                logger.info(f"RAG query completed: {query[:60]}...")
            except Exception as e:
                logger.warning(f"RAG query failed for '{query[:50]}...': {e}")
        
        # Build RAG context
        rag_context = "\n\n".join([
            f"Query: {r['query']}\nAnswer: {r['answer']}\nRationale: {r['rationale']}"
            for r in rag_results
        ])
        
        # Step 1: Identify diagram sections
        logger.info("Identifying diagram sections...")
        sections_prompt = build_diagram_sections_prompt(
            repo_name=repo_name,
            rag_context=rag_context,
            language=language
        )
        
        model = OllamaClient()
        model_kwargs = {
            "model": Const.GENERATION_MODEL,
            "format": "json",
            "options": {
                "temperature": 0.7,
                "num_ctx": 8192
            }
        }
        
        api_kwargs = model.convert_inputs_to_api_kwargs(
            input=sections_prompt,
            model_kwargs=model_kwargs,
            model_type=ModelType.LLM
        )
        
        response = model.call(api_kwargs=api_kwargs, model_type=ModelType.LLM)
        
        # Extract content
        if hasattr(response, 'message') and hasattr(response.message, 'content'):
            sections_json = response.message.content
        else:
            sections_json = str(response)
        
        try:
            sections_data = json.loads(sections_json)
            identified_sections = sections_data.get('sections', [])
            logger.info(f"Identified {len(identified_sections)} sections for diagrams")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse sections JSON: {e}")
            identified_sections = []
        
        # Cache the result
        cache_file = os.path.join(self.cache.diagrams_dir, f"{page_id}_sections.json")
        
        result = {
            "status": "success",
            "repo_name": repo_name,
            "language": language,
            "sections": identified_sections,
            "rag_queries_performed": len(rag_queries),
            "cached": False,
            "cache_file": cache_file
        }
        
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Cached sections to: {cache_file}")
        
        return result
    
    def generate_section_diagram(
        self,
        section_id: str,
        section_title: str,
        section_description: str,
        diagram_type: str,
        key_concepts: List[str],
        language: str = "en",
        use_cache: bool = True
    ) -> Dict:
        """
        Step 2: Generate diagram for a single section (Two-Step API - Part 2).
        
        Generates a comprehensive Mermaid diagram with node/edge explanations for one section.
        
        Args:
            section_id: ID of this section
            section_title: Title of this section
            section_description: Description of what this section covers
            diagram_type: Type of Mermaid diagram (flowchart, sequence, etc.)
            key_concepts: List of key concepts to include
            language: Target language code
            use_cache: Whether to use cached diagram if available
        
        Returns:
            Dict with diagram, nodes with explanations, edges with explanations
        """
        # Check cache first
        if use_cache:
            cache_file = os.path.join(self.cache.diagrams_dir, f"diag_{section_id}.json")
            mermaid_file = os.path.join(self.cache.diagrams_dir, f"diag_{section_id}.mmd")
            if os.path.exists(cache_file):
                logger.info(f"✅ Using cached diagram from: {cache_file}")
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                    cached_data['cached'] = True
                    cached_data['cache_file'] = cache_file
                    cached_data['mermaid_file'] = mermaid_file if os.path.exists(mermaid_file) else None
                    return cached_data
        
        # Ensure RAG is initialized
        if self.rag is None:
            raise RuntimeError("RAG not initialized. Call initialize_rag() first.")
        
        # Perform focused RAG queries for this specific section
        rag_results, retrieved_sources = self._perform_section_rag_queries(section_title)
        
        # Build diagram prompt
        logger.info(f"Generating diagram for: {section_title}")
        diagram_prompt = build_single_diagram_prompt(
            section_title=section_title,
            section_description=section_description,
            diagram_type=diagram_type,
            key_concepts=key_concepts,
            rag_context=rag_results,
            retrieved_sources=retrieved_sources,
            language=language
        )
        
        # Call LLM for diagram
        diagram_data = self._generate_diagram_with_llm(diagram_prompt)
        
        # Process the diagram response
        result = self._process_diagram_response(
            diagram_data,
            section_id,
            section_title,
            section_description,
            language,
            len(rag_results.split('\n\n'))  # Approximate query count
        )
        
        # Cache and add to wiki RAG if successful
        if result.get("status") == "success":
            self._cache_diagram_result(result)
        
        return result
    
    def _perform_section_rag_queries(self, section_title: str) -> tuple:
        """Perform RAG queries for a specific section."""
        section_queries = [
            f"How does {section_title} work?",
            f"What are the components involved in {section_title}?",
            f"Explain the implementation of {section_title}"
        ]
        
        logger.info(f"Performing RAG queries for section: {section_title}")
        rag_results = []
        all_retrieved_docs = []
        
        for query in section_queries:
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
        
        # Build RAG context
        rag_context = "\n\n".join([
            f"Query: {r['query']}\nAnswer: {r['answer']}\nRationale: {r['rationale']}"
            for r in rag_results
        ])
        
        # Deduplicate documents for retrieval
        seen_paths = {}
        unique_docs = []
        for doc in all_retrieved_docs:
            file_path = doc.meta_data.get('file_path', 'unknown') if hasattr(doc, 'meta_data') else 'unknown'
            if file_path not in seen_paths:
                seen_paths[file_path] = doc
                unique_docs.append(doc)
        
        retrieved_sources = "\n\n".join([
            f"Source {i+1} ({doc.meta_data.get('file_path', 'unknown') if hasattr(doc, 'meta_data') else 'unknown'}):\n{doc.text[:800]}"
            for i, doc in enumerate(unique_docs[:15])
        ])
        
        return rag_context, retrieved_sources
    
    def _generate_diagram_with_llm(self, diagram_prompt: str) -> str:
        """Generate diagram using LLM."""
        model = OllamaClient()
        model_kwargs = {
            "model": Const.GENERATION_MODEL,
            "format": "json",
            "options": {
                "temperature": 0.7,
                "num_ctx": 16384
            }
        }
        
        api_kwargs = model.convert_inputs_to_api_kwargs(
            input=diagram_prompt,
            model_kwargs=model_kwargs,
            model_type=ModelType.LLM
        )
        
        diagram_response = model.call(api_kwargs=api_kwargs, model_type=ModelType.LLM)
        
        # Extract diagram content
        if hasattr(diagram_response, 'message') and hasattr(diagram_response.message, 'content'):
            return diagram_response.message.content
        else:
            return str(diagram_response)
    
    def _process_diagram_response(
        self,
        diagram_json: str,
        section_id: str,
        section_title: str,
        section_description: str,
        language: str,
        rag_query_count: int
    ) -> Dict:
        """Process the LLM diagram response and validate."""
        try:
            diagram_data = json.loads(diagram_json)
            mermaid_code = diagram_data.get('mermaid_code', '')
            diagram_description = diagram_data.get('diagram_description', '')
            node_explanations = diagram_data.get('node_explanations', {})
            edge_explanations = diagram_data.get('edge_explanations', {})
            
            # Validate and parse mermaid code
            is_valid, validation_msg = validate_mermaid_syntax(mermaid_code)
            
            if is_valid:
                parsed = parse_mermaid_diagram(mermaid_code)
                
                # Combine LLM explanations with parsed structure
                nodes = {}
                for node_id in parsed['node_list']:
                    node_data = parsed['nodes'][node_id]
                    nodes[node_id] = {
                        "label": node_data['label'],
                        "shape": node_data['shape'],
                        "explanation": node_explanations.get(node_id, "")
                    }
                
                edges = {}
                for edge in parsed['edges']:
                    edge_key = edge['key']
                    edges[edge_key] = {
                        "source": edge['source'],
                        "target": edge['target'],
                        "label": edge['label'],
                        "explanation": edge_explanations.get(edge_key, "")
                    }
                
                # Prepare cache file paths
                cache_file = os.path.join(self.cache.diagrams_dir, f"diag_{section_id}.json")
                mermaid_file = os.path.join(self.cache.diagrams_dir, f"diag_{section_id}.mmd")
                
                return {
                    "status": "success",
                    "section_id": section_id,
                    "section_title": section_title,
                    "section_description": section_description,
                    "language": language,
                    "diagram": {
                        "mermaid_code": mermaid_code,
                        "description": diagram_description,
                        "is_valid": True,
                        "diagram_type": parsed['diagram_type']
                    },
                    "nodes": nodes,
                    "edges": edges,
                    "rag_queries_performed": rag_query_count,
                    "cached": False,
                    "cache_file": cache_file,
                    "mermaid_file": mermaid_file
                }
            else:
                return {
                    "status": "error",
                    "section_id": section_id,
                    "section_title": section_title,
                    "error": f"Invalid Mermaid syntax: {validation_msg}",
                    "diagram": {
                        "mermaid_code": mermaid_code,
                        "description": diagram_description,
                        "is_valid": False,
                        "validation_error": validation_msg
                    },
                    "nodes": node_explanations,
                    "edges": edge_explanations
                }
        
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse diagram JSON: {e}")
            return {
                "status": "error",
                "section_id": section_id,
                "section_title": section_title,
                "error": f"JSON parse error: {str(e)}",
                "raw_response": diagram_json[:500]
            }
    
    def _cache_diagram_result(self, result: Dict):
        """Cache the diagram result and add to wiki RAG."""
        if 'cache_file' in result:
            # Save JSON file
            with open(result['cache_file'], 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            logger.info(f"💾 Cached diagram JSON to: {result['cache_file']}")
            
            # Save Mermaid code separately for easy inspection
            if 'mermaid_file' in result and result['diagram'].get('mermaid_code'):
                with open(result['mermaid_file'], 'w', encoding='utf-8') as f:
                    f.write(result['diagram']['mermaid_code'])
                logger.info(f"💾 Cached Mermaid code to: {result['mermaid_file']}")
            
            # Add to wiki RAG database for /askWiki endpoint
            try:
                self.cache.add_wiki_content_to_rag(
                    content_type="diagram",
                    content_id=result['section_id'],
                    content_data=result
                )
                logger.info(f"📚 Added diagram to wiki RAG: {result['section_id']}")
            except Exception as e:
                logger.warning(f"Failed to add diagram to wiki RAG: {e}")

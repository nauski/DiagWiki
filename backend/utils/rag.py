import os
import logging
import weakref
from dataclasses import dataclass, field
from typing import List, Tuple
import adalflow as adal
from adalflow.components.retriever import FAISSRetriever
from adalflow.core.types import Document
from adalflow.core.db import LocalDB
from const.const import Const

logger = logging.getLogger(__name__)

# RAG System Prompt
RAG_SYSTEM_PROMPT = """You are an expert code assistant helping users understand a codebase.
You have access to relevant code snippets from the repository.
Provide clear, accurate, and helpful answers based on the retrieved context.
If you're not sure about something, acknowledge it rather than making assumptions."""

# RAG Template
RAG_TEMPLATE = r"""<SYS>{{system_prompt}}</SYS>
User: {{input_str}}

{% if contexts %}
<CONTEXT>
The following code snippets and documentation from the repository may be relevant to your question:

{% for doc in contexts %}
---
File: {{ doc.meta_data.file_path if doc.meta_data and doc.meta_data.file_path else 'Unknown' }}
{{ doc.text }}
---
{% endfor %}
</CONTEXT>
{% endif %}

Please provide a helpful answer based on the context above.
{{output_format_str}}

Answer:"""


@dataclass
class RAGAnswer(adal.DataClass):
    """RAG answer with rationale and formatted response"""
    rationale: str = field(
        default="",
        metadata={"desc": "Chain of thoughts for the answer."}
    )
    answer: str = field(
        default="",
        metadata={"desc": "Answer to the user query, formatted in markdown. DO NOT include ``` triple backticks at the beginning or end."}
    )

    __output_fields__ = ["rationale", "answer"]


class RAG(adal.Component):
    """RAG component for code repository question answering.
    
    Uses LocalDB for document storage and FAISS for retrieval.
    """

    def __init__(self, model: str = None):
        """
        Initialize the RAG component.

        Args:
            model: Model name to use (defaults to Const.GENERATION_MODEL)
        """
        super().__init__()

        self.model = model or Const.GENERATION_MODEL
        
        # Check if Ollama model exists
        from utils.dataPipeline import check_ollama_model_exists
        if not check_ollama_model_exists(self.model):
            raise Exception(
                f"Ollama model '{self.model}' not found. "
                f"Please run 'ollama pull {self.model}' to install it."
            )

        # Initialize embedder for queries (single string only for Ollama)
        self.embedder = adal.Embedder(
            model_client=Const.EMBEDDING_CONFIG["client"],
            model_kwargs=Const.EMBEDDING_CONFIG["model_kwargs"]
        )
        
        # Create a single-string embedder wrapper for Ollama compatibility
        self_weakref = weakref.ref(self)
        
        def single_string_embedder(query):
            """Embedder that only accepts single strings (Ollama requirement)"""
            if isinstance(query, list):
                if len(query) != 1:
                    raise ValueError("Ollama embedder only supports a single string")
                query = query[0]
            instance = self_weakref()
            assert instance is not None, "RAG instance is no longer available"
            return instance.embedder(input=query)
        
        self.query_embedder = single_string_embedder

        # Set up output parser
        data_parser = adal.DataClassParser(
            data_class=RAGAnswer,
            return_data_class=True
        )

        format_instructions = data_parser.get_output_format_str() + """

FORMATTING RULES:
1. Provide only the final answer, not your thinking process
2. Format answer in markdown for beautiful rendering
3. DO NOT wrap response in ``` code fences
4. Write content directly without escape characters
5. Use plain text for tags and lists"""

        # Set up the generator
        self.generator = adal.Generator(
            template=RAG_TEMPLATE,
            prompt_kwargs={
                "output_format_str": format_instructions,
                "system_prompt": RAG_SYSTEM_PROMPT,
                "contexts": None,
            },
            model_client=Const.EMBEDDING_CONFIG["client"],
            model_kwargs={"model": self.model},
            output_processors=data_parser,
        )

        # Initialize storage
        self.transformed_docs = []
        self.retriever = None
        self.db_path = None

    def load_database(self, db_path: str):
        """
        Load a LocalDB database from disk.

        Args:
            db_path: Path to the db.pkl file or directory containing it
        """
        # Handle both file and directory paths
        if os.path.isdir(db_path):
            db_path = os.path.join(db_path, "db.pkl")
        
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database not found at: {db_path}")

        logger.info(f"Loading database from: {db_path}")
        db = LocalDB.load_state(filepath=db_path)
        
        # Get transformed data with embeddings
        self.transformed_docs = db.get_transformed_data(key="split_and_embed")
        self.db_path = db_path
        
        logger.info(f"Loaded {len(self.transformed_docs)} documents")

        # Validate and filter embeddings
        self.transformed_docs = self._validate_and_filter_embeddings(self.transformed_docs)
        
        if not self.transformed_docs:
            raise ValueError("No valid documents with embeddings found")

        logger.info(f"Using {len(self.transformed_docs)} documents with valid embeddings")

        # Create FAISS retriever
        self._create_retriever()

    def load_from_folder(self, folder_path: str):
        """
        Load database for a folder path (uses auto-generated db_name).

        Args:
            folder_path: Path to folder whose database to load
        """
        from utils.dataPipeline import generate_db_name
        
        db_name = generate_db_name(folder_path)
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        db_path = os.path.join(data_dir, db_name, "db.pkl")
        
        self.load_database(db_path)

    def _validate_and_filter_embeddings(self, documents: List) -> List:
        """
        Validate embeddings and filter out documents with invalid or mismatched sizes.

        Args:
            documents: List of documents with embeddings

        Returns:
            List of documents with valid embeddings of consistent size
        """
        if not documents:
            logger.warning("No documents provided for validation")
            return []

        valid_documents = []
        embedding_sizes = {}

        # First pass: collect embedding sizes
        for i, doc in enumerate(documents):
            if not hasattr(doc, 'vector') or doc.vector is None:
                logger.warning(f"Document {i} has no embedding, skipping")
                continue

            try:
                if isinstance(doc.vector, list):
                    embedding_size = len(doc.vector)
                elif hasattr(doc.vector, 'shape'):
                    embedding_size = doc.vector.shape[0] if len(doc.vector.shape) == 1 else doc.vector.shape[-1]
                elif hasattr(doc.vector, '__len__'):
                    embedding_size = len(doc.vector)
                else:
                    logger.warning(f"Document {i} has invalid embedding type: {type(doc.vector)}")
                    continue

                if embedding_size == 0:
                    logger.warning(f"Document {i} has empty embedding")
                    continue

                embedding_sizes[embedding_size] = embedding_sizes.get(embedding_size, 0) + 1

            except Exception as e:
                logger.warning(f"Error checking embedding for document {i}: {e}")
                continue

        if not embedding_sizes:
            logger.error("No valid embeddings found")
            return []

        # Find most common embedding size (should be correct one)
        target_size = max(embedding_sizes.keys(), key=lambda k: embedding_sizes[k])
        logger.info(f"Target embedding size: {target_size} (found in {embedding_sizes[target_size]} docs)")

        # Log any incorrect sizes
        for size, count in embedding_sizes.items():
            if size != target_size:
                logger.warning(f"Found {count} documents with incorrect size {size}, filtering out")

        # Second pass: filter to target size
        for i, doc in enumerate(documents):
            if not hasattr(doc, 'vector') or doc.vector is None:
                continue

            try:
                if isinstance(doc.vector, list):
                    embedding_size = len(doc.vector)
                elif hasattr(doc.vector, 'shape'):
                    embedding_size = doc.vector.shape[0] if len(doc.vector.shape) == 1 else doc.vector.shape[-1]
                elif hasattr(doc.vector, '__len__'):
                    embedding_size = len(doc.vector)
                else:
                    continue

                if embedding_size == target_size:
                    valid_documents.append(doc)
                else:
                    file_path = doc.meta_data.get('file_path', f'doc_{i}') if hasattr(doc, 'meta_data') else f'doc_{i}'
                    logger.debug(f"Filtering '{file_path}': size {embedding_size} != {target_size}")

            except Exception as e:
                logger.warning(f"Error validating document {i}: {e}")
                continue

        logger.info(f"Validation complete: {len(valid_documents)}/{len(documents)} valid documents")

        if len(valid_documents) < len(documents):
            filtered_count = len(documents) - len(valid_documents)
            logger.warning(f"Filtered out {filtered_count} documents due to embedding issues")

        return valid_documents

    def _create_retriever(self):
        """Create FAISS retriever from loaded documents"""
        if not self.transformed_docs:
            raise ValueError("No documents loaded. Call load_database() first.")

        try:
            # FAISS retriever configuration
            retriever_config = {
                "top_k": 5,
                "dimensions": 768,  # nomic-embed-text dimension
                "metric": "cosine"
            }

            self.retriever = FAISSRetriever(
                **retriever_config,
                embedder=self.query_embedder,
                documents=self.transformed_docs,
                document_map_func=lambda doc: doc.vector,
            )
            logger.info("FAISS retriever created successfully")
            
        except Exception as e:
            logger.error(f"Error creating FAISS retriever: {e}")
            
            # Provide debugging info for embedding size errors
            if "All embeddings should be of the same size" in str(e):
                logger.error("Embedding size mismatch detected")
                sizes = []
                for i, doc in enumerate(self.transformed_docs[:10]):
                    if hasattr(doc, 'vector') and doc.vector is not None:
                        try:
                            size = len(doc.vector) if isinstance(doc.vector, list) else doc.vector.shape[0]
                            sizes.append(f"doc_{i}: {size}")
                        except:
                            sizes.append(f"doc_{i}: error")
                logger.error(f"Sample sizes: {', '.join(sizes)}")
            raise

    def _compute_bm25_scores(self, query: str, doc_indices: List[int]) -> List[float]:
        """
        Compute BM25 scores for documents.
        
        BM25 is a ranking function that scores documents based on term frequency,
        inverse document frequency, and document length normalization.
        
        Args:
            query: User query
            doc_indices: Indices of documents to score
            
        Returns:
            List of BM25 scores for each document
        """
        import re
        import math
        
        # BM25 parameters
        k1 = 1.5  # Term frequency saturation parameter
        b = 0.75  # Length normalization parameter
        
        # Tokenize query (simple whitespace + lowercase)
        stop_words = {'what', 'is', 'the', 'how', 'does', 'can', 'you', 'explain', 'show', 
                     'me', 'a', 'an', 'of', 'to', 'in', 'for', 'on', 'with', 'from', 'at',
                     'by', 'about', 'as', 'are', 'was', 'were', 'be', 'been', 'being'}
        query_terms = [w.lower() for w in re.findall(r'\w+', query) 
                      if w.lower() not in stop_words and len(w) > 2]
        
        if not query_terms:
            return [0.0] * len(doc_indices)
        
        # Get documents and compute stats
        docs = [self.transformed_docs[idx] for idx in doc_indices]
        doc_texts = [doc.text.lower() for doc in docs]
        
        # Compute document lengths
        doc_lengths = [len(re.findall(r'\w+', text)) for text in doc_texts]
        avg_doc_length = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 1.0
        
        # Compute term document frequency (for IDF)
        term_doc_freq = {}
        for term in query_terms:
            term_doc_freq[term] = sum(1 for text in doc_texts if term in text)
        
        # Compute BM25 scores
        scores = []
        num_docs = len(docs)
        
        for i, (doc_text, doc_len) in enumerate(zip(doc_texts, doc_lengths)):
            score = 0.0
            
            for term in query_terms:
                # Term frequency in document
                tf = doc_text.count(term)
                
                if tf == 0:
                    continue
                
                # Inverse document frequency
                df = term_doc_freq[term]
                idf = math.log((num_docs - df + 0.5) / (df + 0.5) + 1.0)
                
                # BM25 formula
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * (doc_len / avg_doc_length))
                score += idf * (numerator / denominator)
            
            # Boost if query terms appear in file path
            file_path = docs[i].meta_data.get('file_path', '').lower() if hasattr(docs[i], 'meta_data') else ''
            for term in query_terms:
                if term in file_path:
                    score += 0.5  # Path match bonus
            
            scores.append(score)
        
        return scores
    
    def _rerank_with_keywords(self, query: str, doc_indices: List[int], initial_top_k: int) -> List[int]:
        """
        Re-rank documents using BM25 keyword scoring combined with semantic similarity.
        
        Uses Reciprocal Rank Fusion (RRF) to combine:
        1. Semantic similarity scores (from FAISS)
        2. BM25 keyword relevance scores
        
        Args:
            query: User query
            doc_indices: Indices of documents (already ranked by semantic similarity)
            initial_top_k: Number of documents retrieved
            
        Returns:
            Re-ranked list of document indices
        """
        # Compute BM25 scores
        bm25_scores = self._compute_bm25_scores(query, doc_indices)
        
        # Reciprocal Rank Fusion (RRF)
        # Combines semantic rank (position in doc_indices) with BM25 scores
        k = 60  # RRF constant (standard value)
        
        # Semantic ranking (position-based)
        semantic_ranks = {idx: rank for rank, idx in enumerate(doc_indices)}
        
        # BM25 ranking
        bm25_ranked = sorted(enumerate(doc_indices), key=lambda x: bm25_scores[x[0]], reverse=True)
        bm25_ranks = {idx: rank for rank, (_, idx) in enumerate(bm25_ranked)}
        
        # Combine scores using RRF
        combined_scores = []
        for idx in doc_indices:
            semantic_rrf = 1.0 / (k + semantic_ranks[idx])
            bm25_rrf = 1.0 / (k + bm25_ranks[idx])
            combined_score = semantic_rrf + bm25_rrf
            combined_scores.append((idx, combined_score))
        
        # Sort by combined score
        combined_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Log re-ranking changes
        if doc_indices[:3] != [idx for idx, _ in combined_scores[:3]]:
            logger.info("Re-ranking changed top 3 results")
            top_files = [self.transformed_docs[idx].meta_data.get('file_path', 'unknown') 
                        for idx, _ in combined_scores[:3] 
                        if hasattr(self.transformed_docs[idx], 'meta_data')]
            logger.info(f"Top 3 after re-ranking: {top_files}")
        
        return [idx for idx, _ in combined_scores]

    def call(self, query: str, top_k: int = 5, use_reranking: bool = True) -> Tuple[RAGAnswer, List[Document]]:
        """
        Process a query using RAG with optional keyword-based re-ranking.

        Args:
            query: User's question
            top_k: Number of documents to retrieve
            use_reranking: Whether to apply keyword-based re-ranking (default: True)

        Returns:
            Tuple of (RAGAnswer, retrieved_documents)
        """
        if self.retriever is None:
            raise ValueError("Retriever not initialized. Call load_database() first.")

        try:
            # Retrieve more documents initially for re-ranking
            initial_top_k = top_k * 3 if use_reranking else top_k
            retrieved_results = self.retriever(query, top_k=initial_top_k)
            
            # Handle both list and RetrieverOutput formats
            if isinstance(retrieved_results, list):
                # List of RetrieverOutput objects
                if len(retrieved_results) > 0:
                    retrieved_result = retrieved_results[0]
                    doc_indices = retrieved_result.doc_indices
                else:
                    doc_indices = []
            else:
                # Single RetrieverOutput object
                doc_indices = retrieved_results.doc_indices
            
            # Apply keyword-based re-ranking if enabled
            if use_reranking and doc_indices:
                doc_indices = self._rerank_with_keywords(query, doc_indices, initial_top_k)
                # Take top_k after re-ranking
                doc_indices = doc_indices[:top_k]
            
            # Get actual documents
            retrieved_docs = [
                self.transformed_docs[doc_index]
                for doc_index in doc_indices
            ]

            # Generate answer
            self.generator.prompt_kwargs["contexts"] = retrieved_docs
            response = self.generator(prompt_kwargs={"input_str": query})

            # Parse response
            if isinstance(response, adal.GeneratorOutput):
                answer = response.data if hasattr(response, 'data') else response
            else:
                answer = response

            logger.info(f"Generated answer for query: {query[:50]}...")
            return answer, retrieved_docs

        except Exception as e:
            logger.error(f"Error in RAG call: {e}", exc_info=True)
            
            # Create error response
            error_response = RAGAnswer(
                rationale="Error occurred while processing the query.",
                answer=f"I apologize, but I encountered an error: {str(e)}. Please try again."
            )
            return error_response, []

    def __call__(self, query: str, top_k: int = 5, use_reranking: bool = True) -> Tuple[RAGAnswer, List[Document]]:
        """Shorthand for call()"""
        return self.call(query, top_k, use_reranking)

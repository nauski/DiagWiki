import os, logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Query, HTTPException, Body
from pydantic import BaseModel, Field
from typing import Optional, List
from utils.repoUtil import RepoUtil
from utils.dataPipeline import check_ollama_model_exists, get_all_ollama_models, DataPipeline, generate_db_name

from const.config import Config, APP_NAME, APP_VERSION
from const.const import Const


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle events for the FastAPI app"""
    # Startup
    logger.info(f"Starting {APP_NAME} v{APP_VERSION}")
    logger.info(f"Environment: {Config.ENVIRONMENT}")
    
    yield
    
    # Shutdown
    logger.info(f"Shutting down {APP_NAME}")


# Create FastAPI app
app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    lifespan=lifespan,
)


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "app": APP_NAME,
        "version": APP_VERSION,
        "environment": Config.ENVIRONMENT
    }

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": f"Welcome to {APP_NAME} API v{APP_VERSION}!"}


# Repo Tree Structure
# accept local path as query param
@app.get("/tree")
async def get_repo_tree(path: str = Query(None, description="Local path to generate repo tree structure")):
    """Get repository tree structure starting from the given local path"""
    import os

    if path is None:
        return {"error": "Path query parameter is required."}

    if not os.path.exists(path):
        return {"error": f"Path '{path}' does not exist."}

    repo_tree = RepoUtil.build_tree(path)
    return repo_tree


@app.get("/value_files")
async def get_valuable_files(path: str = Query(None, description="Local path to scan for valuable files")):
    """Get list of valuable files in the repository starting from the given local path"""

    if path is None:
        return {"error": "Path query parameter is required."}

    if not os.path.exists(path):
        return {"error": f"Path '{path}' does not exist."}

    valuable_files = RepoUtil.repo_filter(path)
    return {"root_path": path, "valuable_files": valuable_files}


@app.get("/file_content")
async def get_file_content(
    root: str = Query(None, description="Root path of the repository"),
    path: str = Query(None, description="File path relative to the root")
):
    """Get content of a specific file given its path relative to root"""

    if root is None or path is None:
        return {"error": "Both root and path query parameters are required."}

    if not os.path.exists(root):
        return {"error": f"Root path '{root}' does not exist."}

    full_path = os.path.join(root, path)
    if not os.path.exists(full_path):
        return {"error": f"File path '{full_path}' does not exist."}

    content = RepoUtil.file_content(root, path)
    if content is None:
        return {"error": f"Could not read content of file '{full_path}'."}

    return {"file_path": full_path, "content": content}

@app.get("/available_models")
async def list_available_models():
    """Get list of all available Ollama models"""
    models = get_all_ollama_models()
    embedding_model = Const.EMBEDDING_CONFIG['model_kwargs'].get('model', 'nomic-embed-text')
    generation_model = Const.GENERATION_MODEL
    
    return {
        "available_models": models,
        "current_embedding_model": embedding_model,
        "current_generation_model": generation_model,
        "embedding_model_available": check_ollama_model_exists(embedding_model),
        "generation_model_available": check_ollama_model_exists(generation_model)
    }

@app.post("/transform")
async def transform_documents(
    folder_path: str = Query(..., description="Path to folder containing documents to process"),
    extensions: str = Query(None, description="Comma-separated file extensions (e.g., '.py,.md')")
):
    """
    Transform documents from a folder: collect files, split text, generate embeddings, and save to LocalDB.
    
    Database name is automatically generated from the folder path for consistency.
    
    Args:
        folder_path: Path to folder containing documents
        extensions: Optional comma-separated file extensions to filter (default: all supported)
    
    Returns:
        Statistics about the transformation process including the generated database name
    """
    try:
        logger.info(f"Starting document transformation for folder: {folder_path}")
        
        # Validate folder exists
        if not os.path.exists(folder_path):
            raise HTTPException(status_code=404, detail=f"Folder not found: {folder_path}")
        
        if not os.path.isdir(folder_path):
            raise HTTPException(status_code=400, detail=f"Path is not a directory: {folder_path}")
        
        # Generate database name from folder path
        db_name = generate_db_name(folder_path)
        
        # Parse extensions if provided
        allowed_extensions = None
        if extensions:
            allowed_extensions = [ext.strip() for ext in extensions.split(',')]
            logger.info(f"Filtering for extensions: {allowed_extensions}")
        
        # Initialize pipeline and process folder
        pipeline = DataPipeline(
            db_name=db_name,
            embedder_model=Const.EMBEDDING_MODEL,
            text_splitter_config=Const.TEXT_SPLIT_CONFIG
        )
        logger.info("Initialized DataPipeline")
        
        # Process folder (collect, transform, save)
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        stats = pipeline.process_folder(
            folder_path=folder_path,
            allowed_extensions=allowed_extensions,
            data_dir=data_dir
        )
        
        logger.info(f"Transform completed: {stats['chunks_with_embeddings']}/{stats['transformed_chunk_count']} chunks with embeddings")
        return stats
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during transformation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Transformation failed: {str(e)}")


@app.post("/initWiki")
async def init_wiki(
    root_path: str = Query(..., description="Root path to the folder for wiki initialization")
):
    """
    Initialize wiki from a local folder.
    
    This endpoint:
    1. Validates the folder exists
    2. Processes documents (transform) if not already done
    3. Initializes RAG system with the database
    4. Returns initialization status
    
    Args:
        root_path: Absolute path to the folder containing documents
        
    Returns:
        Initialization status including RAG readiness and database info
    """
    try:
        logger.info(f"Initializing wiki for folder: {root_path}")
        
        # Validate folder exists
        if not os.path.exists(root_path):
            raise HTTPException(status_code=404, detail=f"Folder not found: {root_path}")
        
        if not os.path.isdir(root_path):
            raise HTTPException(status_code=400, detail=f"Path is not a directory: {root_path}")
        
        # Generate database name
        db_name = generate_db_name(root_path)
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        db_dir = os.path.join(data_dir, db_name)
        db_path = os.path.join(db_dir, "db.pkl")
        
        # Check if database already exists
        db_exists = os.path.exists(db_path)
        
        if not db_exists:
            logger.info("Database not found, processing documents...")
            
            # Collect and process documents
            documents = RepoUtil.collect_documents(root_path, None)
            
            if not documents:
                raise HTTPException(status_code=400, detail="No documents found in folder")
            
            logger.info(f"Collected {len(documents)} documents")
            
            # Transform and save
            pipeline = DataPipeline()
            db = pipeline.transform_and_save(documents, db_dir)
            
            transformed_docs = db.get_transformed_data(key="split_and_embed")
            logger.info(f"Processed {len(transformed_docs)} chunks with embeddings")
        else:
            logger.info(f"Database already exists at: {db_path}")
            # Load existing database to get stats
            from adalflow.core.db import LocalDB
            db = LocalDB.load_state(filepath=db_path)
            transformed_docs = db.get_transformed_data(key="split_and_embed")
        
        # Initialize RAG system
        from utils.rag import RAG
        rag = RAG()
        rag.load_database(db_path)
        
        logger.info(f"RAG system initialized with {len(rag.transformed_docs)} documents")
        
        # Gather statistics
        file_types = {}
        for doc in rag.transformed_docs:
            if hasattr(doc, 'meta_data'):
                ext = doc.meta_data.get('extension', 'unknown')
                file_types[ext] = file_types.get(ext, 0) + 1
        
        return {
            "status": "success",
            "message": "Wiki initialized successfully" if not db_exists else "Wiki loaded from existing database",
            "root_path": root_path,
            "database_name": db_name,
            "database_path": db_path,
            "database_existed": db_exists,
            "rag_ready": True,
            "document_count": len(rag.transformed_docs),
            "embedding_model": Const.EMBEDDING_CONFIG['model_kwargs']['model'],
            "generation_model": Const.GENERATION_MODEL,
            "file_types": file_types
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error initializing wiki: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to initialize wiki: {str(e)}")


@app.post("/query")
async def query_wiki(
    root_path: str = Query(..., description="Root path to the wiki folder"),
    query: str = Query(..., description="Question to ask about the codebase"),
    top_k: int = Query(5, description="Number of documents to retrieve"),
    use_reranking: bool = Query(True, description="Whether to use hybrid retrieval (semantic + BM25)")
):
    """
    Query the wiki using RAG.
    
    This endpoint:
    1. Loads the RAG system for the specified folder
    2. Retrieves relevant document chunks
    3. Generates an answer using the LLM
    4. Returns the answer with source documents
    
    Args:
        root_path: Absolute path to the wiki folder
        query: User's question
        top_k: Number of documents to retrieve (default: 5)
        use_reranking: Enable hybrid retrieval for better results (default: True)
        
    Returns:
        Generated answer with rationale and source documents
    """
    try:
        logger.info(f"Processing query for folder: {root_path}")
        logger.info(f"Query: {query}")
        
        # Validate folder exists
        if not os.path.exists(root_path):
            raise HTTPException(status_code=404, detail=f"Folder not found: {root_path}")
        
        # Generate database name and check if it exists
        db_name = generate_db_name(root_path)
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        db_path = os.path.join(data_dir, db_name, "db.pkl")
        
        if not os.path.exists(db_path):
            raise HTTPException(
                status_code=400, 
                detail=f"Wiki not initialized for this folder. Please call /initWiki first."
            )
        
        # Initialize RAG
        from utils.rag import RAG
        rag = RAG()
        rag.load_database(db_path)
        
        logger.info(f"RAG loaded with {len(rag.transformed_docs)} documents")
        
        # Perform RAG query
        answer, retrieved_docs = rag(query, top_k=top_k, use_reranking=use_reranking)
        
        # Format retrieved documents info
        sources = []
        for i, doc in enumerate(retrieved_docs, 1):
            file_path = doc.meta_data.get('file_path', 'unknown') if hasattr(doc, 'meta_data') else 'unknown'
            sources.append({
                "rank": i,
                "file_path": file_path,
                "text_preview": doc.text[:200] + "..." if len(doc.text) > 200 else doc.text,
                "text_length": len(doc.text)
            })
        
        logger.info(f"Query answered successfully, retrieved {len(sources)} sources")
        
        return {
            "status": "success",
            "query": query,
            "answer": {
                "rationale": answer.rationale,
                "content": answer.answer
            },
            "sources": sources,
            "retrieval_method": "hybrid (semantic + BM25)" if use_reranking else "semantic only",
            "model": {
                "embedding": Const.EMBEDDING_CONFIG['model_kwargs']['model'],
                "generation": Const.GENERATION_MODEL
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process query: {str(e)}")


# Pydantic models for wiki generation
class WikiStructureRequest(BaseModel):
    root_path: str = Field(..., description="Root path to the folder")
    comprehensive: bool = Field(False, description="Whether to create comprehensive wiki (8-12 pages) or concise (4-6 pages)")
    language: str = Field("en", description="Language code (en, ja, zh, es, kr, vi, etc.)")


class WikiPageRequest(BaseModel):
    root_path: str = Field(..., description="Root path to the folder")
    page_title: str = Field(..., description="Title of the page to generate")
    page_description: str = Field(..., description="Description of what the page should cover")
    relevant_files: List[str] = Field(..., description="List of relevant file paths for this page")
    language: str = Field("en", description="Language code")


@app.post("/generateWikiStructure")
async def generate_wiki_structure(request: WikiStructureRequest = Body(...)):
    """
    Generate wiki structure by analyzing the codebase using RAG.
    
    This endpoint:
    1. Initializes RAG for the codebase
    2. Generates file tree structure
    3. Reads README if exists
    4. Uses RAG queries to understand codebase components
    5. Uses LLM to analyze and create wiki structure
    6. Returns structured pages and sections in XML format
    
    Args:
        request: WikiStructureRequest with root_path, comprehensive flag, and language
        
    Returns:
        XML structure defining wiki pages and sections
    """
    try:
        logger.info(f"Generating wiki structure for: {request.root_path}")
        
        # Validate folder
        if not os.path.exists(request.root_path):
            raise HTTPException(status_code=404, detail=f"Folder not found: {request.root_path}")
        
        # Initialize RAG for this folder
        from utils.rag import RAG
        db_name = generate_db_name(request.root_path)
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        db_path = os.path.join(data_dir, db_name)
        
        # Check if database exists, if not create it
        if not os.path.exists(db_path):
            logger.info(f"Database not found for {request.root_path}, creating...")
            pipeline = DataPipeline(
                db_name=db_name,
                embedder_model=Const.EMBEDDING_MODEL,
                text_splitter_config=Const.TEXT_SPLIT_CONFIG
            )
            result = pipeline.process_folder(
                folder_path=request.root_path,
                data_dir=data_dir
            )
            logger.info(f"Database created: {result}")
        
        # Initialize RAG
        from utils.rag import RAG
        rag = RAG()
        rag.load_database(db_path)
        logger.info("RAG initialized for wiki structure generation")
        
        # Generate file tree
        file_tree = RepoUtil.build_tree(request.root_path)
        logger.info("File tree generated")
        
        # Read README if exists
        readme_content = ""
        readme_paths = ["README.md", "README.MD", "readme.md", "README.txt", "README"]
        for readme_name in readme_paths:
            readme_path = os.path.join(request.root_path, readme_name)
            if os.path.exists(readme_path):
                try:
                    with open(readme_path, 'r', encoding='utf-8') as f:
                        readme_content = f.read()
                    logger.info(f"README found: {readme_name}")
                    break
                except Exception as e:
                    logger.warning(f"Error reading README: {e}")
        
        # Use RAG to gather codebase insights
        logger.info("Querying RAG for codebase analysis...")
        rag_insights = []
        
        analysis_queries = [
            "What are the main components, modules, and their purposes in this codebase?",
            "What is the system architecture and how are different parts connected?",
            "What are the key features and core functionality provided?",
            "What APIs, endpoints, or interfaces are exposed?",
            "What data models, schemas, or data structures are used?"
        ]
        
        for query in analysis_queries:
            try:
                result = rag.call(
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
        
        # Get folder name for context
        folder_name = os.path.basename(request.root_path)
        
        # Language mapping
        language_names = {
            'en': 'English',
            'ja': 'Japanese (日本語)',
            'zh': 'Mandarin Chinese (中文)',
            'zh-tw': 'Traditional Chinese (繁體中文)',
            'es': 'Spanish (Español)',
            'kr': 'Korean (한국어)',
            'vi': 'Vietnamese (Tiếng Việt)',
            'pt-br': 'Brazilian Portuguese (Português Brasileiro)',
            'fr': 'Français (French)',
            'ru': 'Русский (Russian)'
        }
        language_name = language_names.get(request.language, 'English')
        
        # Create the prompt
        if request.comprehensive:
            structure_instructions = """
Create a structured wiki with the following main sections:
- Overview (general information about the project)
- System Architecture (how the system is designed)
- Core Features (key functionality)
- Data Management/Flow: If applicable, how data is stored, processed, accessed, and managed
- Frontend Components (UI elements, if applicable)
- Backend Systems (server-side components)
- Model Integration (AI model connections, if applicable)
- Deployment/Infrastructure (how to deploy, infrastructure)
- Extensibility and Customization: If supported, explain how to extend or customize

Each section should contain relevant pages. Return XML with sections and pages.

<wiki_structure>
  <title>[Overall title for the wiki]</title>
  <description>[Brief description of the repository]</description>
  <sections>
    <section id="section-1">
      <title>[Section title]</title>
      <pages>
        <page_ref>page-1</page_ref>
        <page_ref>page-2</page_ref>
      </pages>
      <subsections>
        <section_ref>section-2</section_ref>
      </subsections>
    </section>
  </sections>
  <pages>
    <page id="page-1">
      <title>[Page title]</title>
      <description>[Brief description]</description>
      <importance>high|medium|low</importance>
      <relevant_files>
        <file_path>[Path to relevant file]</file_path>
      </relevant_files>
      <related_pages>
        <related>page-2</related>
      </related_pages>
      <parent_section>section-1</parent_section>
    </page>
  </pages>
</wiki_structure>"""
            page_count = "8-12"
        else:
            structure_instructions = """
Return your analysis in the following XML format:

<wiki_structure>
  <title>[Overall title for the wiki]</title>
  <description>[Brief description of the repository]</description>
  <pages>
    <page id="page-1">
      <title>[Page title]</title>
      <description>[Brief description]</description>
      <importance>high|medium|low</importance>
      <relevant_files>
        <file_path>[Path to relevant file]</file_path>
      </relevant_files>
      <related_pages>
        <related>page-2</related>
      </related_pages>
    </page>
  </pages>
</wiki_structure>"""
            page_count = "4-6"
        
        # Build RAG insights section
        rag_insights_text = ""
        if rag_insights:
            rag_insights_text = "\n\n3. Codebase Analysis (from RAG retrieval):\n<rag_analysis>\n"
            for idx, insight in enumerate(rag_insights, 1):
                rag_insights_text += f"\nQuestion {idx}: {insight['query']}\n"
                rag_insights_text += f"Answer: {insight['answer']}\n"
                if insight['sources']:
                    rag_insights_text += "Key code snippets:\n"
                    for src in insight['sources']:
                        rag_insights_text += f"- {src}\n"
            rag_insights_text += "</rag_analysis>"
        
        prompt = f"""Analyze this folder {folder_name} and create a wiki structure for it.

1. The complete file tree of the project:
<file_tree>
{file_tree}
</file_tree>

2. The README file of the project:
<readme>
{readme_content if readme_content else "No README found"}
</readme>{rag_insights_text}

I want to create a wiki for this codebase. Determine the most logical structure for a wiki based on the content.

IMPORTANT: Use the RAG analysis insights above to understand the codebase's actual components, architecture, and features. The relevant_files in your structure should reference the actual source files mentioned in the RAG analysis.

IMPORTANT: The wiki content will be generated in {language_name} language.

When designing the wiki structure, include pages that would benefit from visual diagrams, such as:
- Architecture overviews
- Data flow descriptions
- Component relationships
- Process workflows
- State machines
- Class hierarchies

{structure_instructions}

IMPORTANT FORMATTING INSTRUCTIONS:
- Return ONLY the valid XML structure specified above
- DO NOT wrap the XML in markdown code blocks (no ``` or ```xml)
- DO NOT include any explanation text before or after the XML
- Ensure the XML is properly formatted and valid
- Start directly with <wiki_structure> and end with </wiki_structure>

IMPORTANT:
1. Create {page_count} pages that would make a {'comprehensive' if request.comprehensive else 'concise'} wiki for this codebase
2. Each page should focus on a specific aspect (e.g., architecture, key features, setup)
3. The relevant_files should be actual files from the codebase that would be used to generate that page
4. Return ONLY valid XML with the structure specified above, with no markdown code block delimiters"""
        
        # Use LLM to generate structure
        from adalflow.components.model_client.ollama_client import OllamaClient
        from adalflow.core.types import ModelType
        
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
        
        return {
            "status": "success",
            "root_path": request.root_path,
            "comprehensive": request.comprehensive,
            "language": request.language,
            "wiki_structure": wiki_structure
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating wiki structure: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate wiki structure: {str(e)}")


@app.post("/generateWikiPage")
async def generate_wiki_page(request: WikiPageRequest = Body(...)):
    """
    Generate detailed wiki page content using RAG-based retrieval.
    
    This endpoint:
    1. Initializes RAG for the codebase
    2. Generates targeted queries from page topic
    3. Uses RAG with hybrid retrieval to get relevant context
    4. Generates comprehensive markdown with diagrams
    5. Returns formatted wiki page content
    
    Args:
        request: WikiPageRequest with page details and relevant files
        
    Returns:
        Markdown content for the wiki page
    """
    try:
        logger.info(f"Generating wiki page: {request.page_title}")
        
        # Validate folder
        if not os.path.exists(request.root_path):
            raise HTTPException(status_code=404, detail=f"Folder not found: {request.root_path}")
        
        # Initialize RAG
        from utils.rag import RAG
        db_name = generate_db_name(request.root_path)
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        db_path = os.path.join(data_dir, db_name)
        
        if not os.path.exists(db_path):
            raise HTTPException(
                status_code=404, 
                detail=f"Database not found for {request.root_path}. Please run /generateWikiStructure first."
            )
        
        rag = RAG()
        rag.load_database(db_path)
        logger.info(f"RAG initialized for page generation: {request.page_title}")
        
        # Generate RAG queries based on page topic and description
        rag_queries = [
            f"What is {request.page_title}? {request.page_description}",
            f"How does {request.page_title} work? Explain the implementation details.",
            f"What are the key components and functions related to {request.page_title}?",
            f"What are the data structures, classes, or APIs for {request.page_title}?",
            f"Show code examples and usage patterns for {request.page_title}."
        ]
        
        # Perform RAG queries to gather comprehensive context
        logger.info(f"Performing {len(rag_queries)} RAG queries for comprehensive context...")
        rag_results = []
        all_retrieved_docs = []
        
        for query in rag_queries:
            try:
                result = rag.call(
                    query=query,
                    top_k=8,  # Get more docs for comprehensive coverage
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
        
        # Deduplicate documents by file path and keep most relevant
        seen_paths = {}
        unique_docs = []
        for doc in all_retrieved_docs:
            file_path = doc.meta_data.get('file_path', 'unknown') if hasattr(doc, 'meta_data') else 'unknown'
            if file_path not in seen_paths:
                seen_paths[file_path] = doc
                unique_docs.append(doc)
        
        logger.info(f"Retrieved {len(unique_docs)} unique documents from {len(all_retrieved_docs)} total results")
        
        # Also load any explicitly requested files
        file_contents = []
        for file_path in request.relevant_files[:5]:  # Limit to 5 explicit files
            full_path = os.path.join(request.root_path, file_path) if not os.path.isabs(file_path) else file_path
            if os.path.exists(full_path) and os.path.isfile(full_path):
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    file_contents.append({
                        "path": file_path,
                        "content": content[:3000]  # Limit each file to 3000 chars
                    })
                    logger.info(f"Loaded explicit file: {file_path}")
                except Exception as e:
                    logger.warning(f"Error reading file {file_path}: {e}")
        
        # Build comprehensive context from RAG results
        rag_context = "\n\n".join([
            f"Query: {r['query']}\nAnswer: {r['answer']}\nRationale: {r['rationale']}"
            for r in rag_results
        ])
        
        retrieved_sources = "\n\n".join([
            f"Source {i+1} ({doc.meta_data.get('file_path', 'unknown') if hasattr(doc, 'meta_data') else 'unknown'}):\n{doc.text[:800]}"
            for i, doc in enumerate(unique_docs[:15])  # Limit to top 15 sources
        ])
        
        # Language mapping
        language_names = {
            'en': 'English',
            'ja': 'Japanese (日本語)',
            'zh': 'Mandarin Chinese (中文)',
            'zh-tw': 'Traditional Chinese (繁體中文)',
            'es': 'Spanish (Español)',
            'kr': 'Korean (한국어)',
            'vi': 'Vietnamese (Tiếng Việt)',
            'pt-br': 'Brazilian Portuguese (Português Brasileiro)',
            'fr': 'Français (French)',
            'ru': 'Русский (Russian)'
        }
        language_name = language_names.get(request.language, 'English')
        
        file_list = '\n'.join([f"- {fc['path']}" for fc in file_contents]) if file_contents else "(Retrieved via RAG)"
        
        # Create detailed prompt
        prompt = f"""You are an expert technical writer and software architect.
Your task is to generate a comprehensive and accurate technical wiki page in Markdown format.

TOPIC: {request.page_title}
DESCRIPTION: {request.page_description}

=== RAG-RETRIEVED CONTEXT (Primary Source) ===

The following information was retrieved using semantic search and hybrid ranking (BM25+RRF) from the codebase:

{rag_context}

=== RETRIEVED SOURCE CODE SNIPPETS ===

{retrieved_sources}

=== ADDITIONAL EXPLICIT FILES ===
{file_list}

CRITICAL STARTING INSTRUCTION:
The very first thing on the page MUST be a <details> block listing the relevant source files.
Format it exactly like this:
<details>
<summary>Relevant source files</summary>

The following files were used as context (retrieved via RAG and hybrid ranking):

{chr(10).join([f"- {doc.meta_data.get('file_path', 'unknown')}" for doc in unique_docs[:10] if hasattr(doc, 'meta_data')])}
</details>

Immediately after the <details> block, the main title should be: # {request.page_title}

Based on the RAG-retrieved context and source code snippets above:

1. **Introduction:** Start with 1-2 paragraphs explaining the purpose and overview.

2. **Detailed Sections:** Break down into logical sections using ## and ### headings:
   - Explain architecture, components, data flow, logic
   - Identify key functions, classes, APIs, configurations

3. **Mermaid Diagrams:**
   - EXTENSIVELY use Mermaid diagrams (flowchart TD, sequenceDiagram, classDiagram, etc.)
   - All diagrams MUST use vertical orientation (graph TD, not LR)
   - For sequence diagrams:
     * Define participants at beginning
     * Use ->> for solid arrow (requests/calls)
     * Use -->> for dotted arrow (responses/returns)
     * Use activation boxes with +/- suffix
     * Use structural elements: loop, alt/else, opt, par, critical, break
   - Provide brief explanation before/after each diagram

4. **Tables:**
   - Summarize features, components, API parameters, config options, data models

5. **Code Snippets (optional):**
   - Include short relevant snippets from source files
   - Well-formatted with language identifiers

6. **Source Citations (CRITICAL):**
   - For EVERY significant piece of information, cite the source file
   - Format: Sources: [filename.ext:start_line-end_line]() or [filename.ext:line_number]()
6. **Source Citations (CRITICAL):**
   - For EVERY significant piece of information, cite the source file from RAG results
   - Format: Sources: [filename.ext]() (line numbers may not be available from RAG)
   - Reference the retrieved sources provided above
   - Multiple files: Sources: [file1.ext](), [file2.ext]()

7. **Technical Accuracy:** All information must be from RAG-retrieved context and source code snippets only.

8. **Clarity:** Use clear, professional, concise technical language.

9. **Conclusion:** End with brief summary if appropriate.

IMPORTANT: Generate the content in {language_name} language.
"""
        
        # Add any explicit file contents if provided
        if file_contents:
            prompt += "\n\n=== EXPLICIT FILE CONTENTS ===\n"
            for fc in file_contents:
                prompt += f"\n{'='*60}\n"
                prompt += f"File: {fc['path']}\n"
                prompt += f"{'='*60}\n"
                prompt += fc['content']
                prompt += f"\n{'='*60}\n\n"
        
        prompt += """\n\nNow generate the comprehensive wiki page in markdown format.

REMEMBER: Base your content primarily on the RAG-retrieved context and source code snippets provided above. 
These have been intelligently retrieved using semantic search and hybrid ranking (BM25+RRF) to find the most relevant information.
The RAG system has already analyzed the query and found the best matching code sections.

Generate the wiki page now:"""
        
        # Use LLM to generate page
        from adalflow.components.model_client.ollama_client import OllamaClient
        from adalflow.core.types import ModelType
        
        model = OllamaClient()
        model_kwargs = {
            "model": Const.GENERATION_MODEL,
            "options": {
                "temperature": 0.7,
                "num_ctx": 16384  # Larger context for comprehensive content
            }
        }
        
        api_kwargs = model.convert_inputs_to_api_kwargs(
            input=prompt,
            model_kwargs=model_kwargs,
            model_type=ModelType.LLM
        )
        
        logger.info(f"Calling LLM to generate wiki page for: {request.page_title}")
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
        
        return {
            "status": "success",
            "page_title": request.page_title,
            "language": request.language,
            "rag_queries_performed": len(rag_queries),
            "rag_results_count": len(rag_results),
            "unique_sources_retrieved": len(unique_docs),
            "explicit_files_loaded": len(file_contents),
            "content": page_content
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating wiki page: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate wiki page: {str(e)}")
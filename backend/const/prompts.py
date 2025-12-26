"""
Prompt templates for LLM interactions.

This module contains all prompt templates used for:
- Wiki structure generation
- Wiki page generation
- RAG queries
"""


def get_language_name(language_code: str) -> str:
    """Get full language name from language code."""
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
    return language_names.get(language_code, 'English')


def build_wiki_structure_prompt(
    folder_name: str,
    file_tree: str,
    readme_content: str,
    rag_insights: list,
    language: str,
    comprehensive: bool
) -> str:
    """
    Build prompt for wiki structure generation.
    
    Args:
        folder_name: Name of the folder/project
        file_tree: Complete file tree structure
        readme_content: README file content
        rag_insights: List of RAG analysis insights
        language: Target language code
        comprehensive: Whether to create comprehensive wiki
    
    Returns:
        Complete prompt string
    """
    language_name = get_language_name(language)
    
    # Build RAG insights section
    rag_insights_text = ""
    if rag_insights:
        rag_insights_text = "\n\n3. Codebase Analysis (from RAG retrieval):\n<rag_analysis>\n"
        for idx, insight in enumerate(rag_insights, 1):
            rag_insights_text += f"\nQuestion {idx}: {insight['query']}\n"
            rag_insights_text += f"Answer: {insight['answer']}\n"
            if insight.get('sources'):
                rag_insights_text += "Key code snippets:\n"
                for src in insight['sources']:
                    rag_insights_text += f"- {src}\n"
        rag_insights_text += "</rag_analysis>"
    
    # Determine structure format and page count
    if comprehensive:
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
1. Create {page_count} pages that would make a {'comprehensive' if comprehensive else 'concise'} wiki for this codebase
2. Each page should focus on a specific aspect (e.g., architecture, key features, setup)
3. The relevant_files should be actual files from the codebase that would be used to generate that page
4. Return ONLY valid XML with the structure specified above, with no markdown code block delimiters"""
    
    return prompt


def build_wiki_page_prompt(
    page_title: str,
    page_description: str,
    rag_context: str,
    retrieved_sources: str,
    file_contents: list,
    unique_docs: list,
    language: str
) -> str:
    """
    Build prompt for wiki page generation.
    
    Args:
        page_title: Title of the page
        page_description: Description of what the page should cover
        rag_context: RAG-retrieved context (answers + rationales)
        retrieved_sources: Retrieved source code snippets
        file_contents: List of explicit file contents
        unique_docs: List of unique documents for citation
        language: Target language code
    
    Returns:
        Complete prompt string
    """
    language_name = get_language_name(language)
    
    file_list = '\n'.join([f"- {fc['path']}" for fc in file_contents]) if file_contents else "(Retrieved via RAG)"
    
    prompt = f"""You are an expert technical writer and software architect.
Your task is to generate a comprehensive and accurate technical wiki page in Markdown format.

TOPIC: {page_title}
DESCRIPTION: {page_description}

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

Immediately after the <details> block, the main title should be: # {page_title}

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
    
    return prompt


# Analysis queries for wiki structure generation
STRUCTURE_ANALYSIS_QUERIES = [
    "What are the main components, modules, and their purposes in this codebase?",
    "What is the system architecture and how are different parts connected?",
    "What are the key features and core functionality provided?",
    "What APIs, endpoints, or interfaces are exposed?",
    "What data models, schemas, or data structures are used?"
]


# Page generation queries template
def build_page_analysis_queries(page_title: str, page_description: str) -> list:
    """
    Build RAG queries for page content generation.
    
    Args:
        page_title: Title of the page
        page_description: Description of the page
    
    Returns:
        List of query strings
    """
    return [
        f"What is {page_title}? {page_description}",
        f"How does {page_title} work? Explain the implementation details.",
        f"What are the key components and functions related to {page_title}?",
        f"What are the data structures, classes, or APIs for {page_title}?",
        f"Show code examples and usage patterns for {page_title}."
    ]

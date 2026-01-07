# Import configuration from config module
# All configurable values are now in config.py and loaded from .env

class Const:
    """
    Application constants that don't require configuration.
    
    For configurable values (model names, timeouts, RAG settings, etc.),
    use Config class from config.py which loads from environment variables.
    """
    
    # File extensions to process
    CODE_EXTENSIONS = [
        ".py", ".js", ".ts", ".java", ".cpp", ".c", ".h", ".hpp", ".go", ".rs",
        ".jsx", ".tsx", ".html", ".css", ".php", ".swift", ".cs", ".svelte", ".rb",
        # Backend & Scripts
        ".sql", ".sh", ".bash", ".pl", ".scala", ".kt", ".kts", ".m", ".mm",
        # Frontend & Modern Web
        ".vue", ".astro", ".less", ".scss", ".sass", ".graphql", ".gql",
        # Config as Code
        ".tf", ".hcl", ".dockerfile"
    ]
    
    DOC_EXTENSIONS = [
        ".md", ".txt", ".rst", ".json", ".yaml", ".yml",
        # Configuration & Schema
        ".toml", ".ini", ".conf", ".cfg", ".xml", ".csv", ".tsv",
        ".env.example", ".lock", ".jsonl", ".proto"
    ]
    
    DIR_SKIP_LIST = [
        "node_modules", "venv", "__pycache__", ".git", "dist", "build", ".venv",
        # IDEs & System
        ".idea", ".vscode", ".vs", ".DS_Store", "thumbs.db",
        # Package Manager Artifacts
        "packages", "vendor", "bower_components", ".npm", ".yarn",
        # Testing & Coverage
        "coverage", ".nyc_output", ".pytest_cache", ".tox",
        # Build & Cache
        "target", "out", "bin", "obj", ".cache", ".next", ".nuxt", ".svelte-kit",
        # Mobile Artifacts
        "Pods", "DerivedData", ".gradle"
    ]
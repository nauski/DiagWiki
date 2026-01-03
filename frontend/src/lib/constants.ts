/**
 * Backend constants utility
 * Fetches and caches constants from the backend API
 */

interface BackendConstants {
	MAX_RAG_CONTEXT_CHARS: number;
	MAX_SOURCES: number;
	MAX_FILE_CHARS: number;
	RAG_TOP_K: number;
	SOURCE_PREVIEW_LENGTH: number;
	DEFAULT_TEMPERATURE: number;
	FOCUSED_TEMPERATURE: number;
	LARGE_CONTEXT_WINDOW: number;
	LLM_TIMEOUT: number;
	GENERATION_MODEL: string;
	EMBEDDING_MODEL: string;
}

// Default values (used as fallback if API call fails)
const DEFAULT_CONSTANTS: BackendConstants = {
	MAX_RAG_CONTEXT_CHARS: 100000,
	MAX_SOURCES: 15,
	MAX_FILE_CHARS: 50000,
	RAG_TOP_K: 20,
	SOURCE_PREVIEW_LENGTH: 600,
	DEFAULT_TEMPERATURE: 0.7,
	FOCUSED_TEMPERATURE: 0.3,
	LARGE_CONTEXT_WINDOW: 16384,
	LLM_TIMEOUT: 180.0,
	GENERATION_MODEL: 'qwen3-coder:30b',
	EMBEDDING_MODEL: 'nomic-embed-text'
};

let cachedConstants: BackendConstants | null = null;
let loadPromise: Promise<BackendConstants> | null = null;

/**
 * Load constants from backend API
 * Results are cached after first successful load
 */
export async function loadConstants(): Promise<BackendConstants> {
	// Return cached constants if available
	if (cachedConstants) {
		return cachedConstants;
	}

	// If already loading, return the existing promise
	if (loadPromise) {
		return loadPromise;
	}

	// Create new load promise
	loadPromise = (async () => {
		try {
			const response = await fetch('http://localhost:8001/constants');
			if (response.ok) {
				const constants = await response.json();
				cachedConstants = constants;
				return constants;
			} else {
				console.error('Failed to load constants from backend, using defaults');
				cachedConstants = DEFAULT_CONSTANTS;
				return DEFAULT_CONSTANTS;
			}
		} catch (err) {
			console.error('Error loading constants:', err);
			cachedConstants = DEFAULT_CONSTANTS;
			return DEFAULT_CONSTANTS;
		} finally {
			loadPromise = null;
		}
	})();

	return loadPromise;
}

/**
 * Get constants synchronously (returns cached or default values)
 * Call loadConstants() first to ensure fresh data from backend
 */
export function getConstants(): BackendConstants {
	return cachedConstants || DEFAULT_CONSTANTS;
}

/**
 * Clear cached constants (useful for testing or forcing reload)
 */
export function clearConstantsCache(): void {
	cachedConstants = null;
	loadPromise = null;
}

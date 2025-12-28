// API Types for backend integration
export interface WikiSection {
	section_id: string;
	section_title: string;
	section_description: string;
	diagram_type: string;
	key_concepts: string[];
}

export interface DiagramData {
	mermaid_code: string;
	description: string;
	is_valid: boolean;
	diagram_type: string;
}

export interface NodeData {
	label: string;
	shape: string;
	explanation: string;
}

export interface EdgeData {
	source: string;
	target: string;
	label: string;
	explanation: string;
}

export interface DiagramSection {
	status: string;
	section_id: string;
	section_title: string;
	section_description: string;
	language: string;
	diagram: DiagramData;
	nodes: Record<string, NodeData>;
	edges: Record<string, EdgeData>;
	cached: boolean;
}

export interface AnalysisResult {
	status: string;
	page_id: string;
	sections: WikiSection[];
	cached: boolean;
	rag_queries_performed: number;
}

export interface ProjectHistory {
	path: string;
	lastAccessed: number;
	diagrams?: string[];
}

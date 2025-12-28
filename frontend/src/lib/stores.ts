import { writable, derived } from 'svelte/store';
import type { ProjectHistory, WikiSection, DiagramSection } from './types';

// Current project path
export const currentProject = writable<string | null>(null);

// Project history (stored in localStorage)
export const projectHistory = writable<ProjectHistory[]>([]);

// Load history from localStorage on init
if (typeof window !== 'undefined') {
	const stored = localStorage.getItem('diagwiki_history');
	if (stored) {
		projectHistory.set(JSON.parse(stored));
	}
}

// Save to localStorage whenever history changes
projectHistory.subscribe((value) => {
	if (typeof window !== 'undefined') {
		localStorage.setItem('diagwiki_history', JSON.stringify(value));
	}
});

// Available diagram sections from analysis
export const availableSections = writable<WikiSection[]>([]);

// Currently open diagram tabs
export const openTabs = writable<DiagramSection[]>([]);

// Active tab index
export const activeTabIndex = writable<number>(0);

// Selected node/edge for explanation panel
export const selectedElement = writable<{
	type: 'node' | 'edge';
	id: string;
	data: any;
} | null>(null);

// Left panel state (tree view)
export const leftPanelOpen = writable<boolean>(true);

// Right panel state (explanation)
export const rightPanelOpen = writable<boolean>(false);

// Loading states
export const isAnalyzing = writable<boolean>(false);
export const isLoadingDiagram = writable<boolean>(false);

// Track which diagrams have been generated (to show loading states)
export const generatedDiagrams = writable<Set<string>>(new Set());

// Active diagram (derived from open tabs and active index)
export const activeDiagram = derived(
	[openTabs, activeTabIndex],
	([$openTabs, $activeTabIndex]) => {
		return $openTabs[$activeTabIndex] || null;
	}
);

// Helper functions
export function addToHistory(path: string) {
	projectHistory.update((history) => {
		const existing = history.find((h) => h.path === path);
		if (existing) {
			existing.lastAccessed = Date.now();
			return [...history.filter((h) => h.path !== path), existing].sort(
				(a, b) => b.lastAccessed - a.lastAccessed
			);
		}
		return [{ path, lastAccessed: Date.now() }, ...history].slice(0, 10); // Keep last 10
	});
}

export function openDiagramTab(diagram: DiagramSection) {
	openTabs.update((tabs) => {
		const existing = tabs.findIndex((t) => t.section_id === diagram.section_id);
		if (existing !== -1) {
			activeTabIndex.set(existing);
			return tabs;
		}
		const newTabs = [...tabs, diagram];
		activeTabIndex.set(newTabs.length - 1);
		return newTabs;
	});
	// Mark as generated (create new Set for reactivity)
	generatedDiagrams.update(set => {
		const newSet = new Set(set);
		newSet.add(diagram.section_id);
		return newSet;
	});
	rightPanelOpen.set(false); // Close right panel when opening new tab
	selectedElement.set(null);
}

export function closeDiagramTab(index: number) {
	openTabs.update((tabs) => {
		const newTabs = tabs.filter((_, i) => i !== index);
		activeTabIndex.update((current) => {
			if (current >= newTabs.length) {
				return Math.max(0, newTabs.length - 1);
			}
			return current;
		});
		return newTabs;
	});
}

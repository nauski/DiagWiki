<script lang="ts">
	import { onMount, onDestroy, afterUpdate } from 'svelte';
	import type { DiagramSection } from '$lib/types';
	import { selectedElement, rightPanelOpen } from '$lib/stores';
	import mermaid from 'mermaid';
	import panzoom from 'panzoom';

	export let diagram: DiagramSection;

	let containerRef: HTMLElement;
	let svgRef: SVGSVGElement | null = null;
	let panzoomInstance: any = null;
	let isRendered = false;
	let currentDiagramId = '';
	let renderCount = 0;
	let zoomLevel = 1;

	onMount(async () => {
		mermaid.initialize({
			startOnLoad: false,
			theme: 'default',
			securityLevel: 'loose',
			flowchart: { useMaxWidth: true, htmlLabels: true }
		});

		await renderDiagram();
	});

	// Re-render when diagram changes
	$: if (diagram && diagram.section_id !== currentDiagramId) {
		currentDiagramId = diagram.section_id;
		renderDiagram();
	}

	async function renderDiagram() {
		if (!containerRef || !diagram?.diagram?.mermaid_code) return;

		// Clean up existing panzoom
		if (panzoomInstance) {
			panzoomInstance.dispose();
			panzoomInstance = null;
		}

		try {
			// Generate unique ID for each render to avoid conflicts
			renderCount++;
			const diagramId = `mermaid-diagram-${renderCount}`;
			
			const { svg } = await mermaid.render(diagramId, diagram.diagram.mermaid_code);
			containerRef.innerHTML = svg;
			isRendered = true;

			// Get the SVG element
			svgRef = containerRef.querySelector('svg');
			if (svgRef) {
				// Initialize panzoom
				panzoomInstance = panzoom(svgRef, {
					maxZoom: 5,
					minZoom: 0.1,
					bounds: true,
					boundsPadding: 0.1,
					zoomSpeed: 0.065,
					onTouch: (e: TouchEvent) => {
						// Allow touch events on interactive elements
						const target = e.target as HTMLElement;
						return !target.closest('.node') && !target.closest('.edgePath');
					}
				});

				// Listen to zoom changes
				panzoomInstance.on('zoom', (e: any) => {
					zoomLevel = e.getTransform().scale;
				});
			}

			// Add click handlers to nodes and edges
			addClickHandlers();
		} catch (error) {
			console.error('Failed to render diagram:', error);
			containerRef.innerHTML = `
				<div class="text-red-600 p-4">
					<p class="font-semibold">Failed to render diagram</p>
					<p class="text-sm">${error instanceof Error ? error.message : 'Unknown error'}</p>
				</div>
			`;
		}
	}

	function resetZoom() {
		if (panzoomInstance) {
			panzoomInstance.moveTo(0, 0);
			panzoomInstance.zoomAbs(0, 0, 1);
			zoomLevel = 1;
		}
	}

	function zoomIn() {
		if (panzoomInstance) {
			panzoomInstance.smoothZoom(0, 0, 1.2);
		}
	}

	function zoomOut() {
		if (panzoomInstance) {
			panzoomInstance.smoothZoom(0, 0, 0.8);
		}
	}

	function addClickHandlers() {
		if (!containerRef) return;

		// Click handlers for nodes - be very specific to avoid multiple selections
		const nodes = containerRef.querySelectorAll('g.node');
		nodes.forEach((node) => {
			// Try to extract node ID from various attributes
			let nodeId = node.id;
			
			// For flowchart nodes, the ID might be in format "flowchart-NodeName-123"
			if (nodeId && nodeId.startsWith('flowchart-')) {
				// Extract the actual node name
				const parts = nodeId.split('-');
				if (parts.length >= 2) {
					nodeId = parts.slice(1, -1).join('-'); // Remove "flowchart-" prefix and number suffix
				}
			}

			// Try to find matching node data
			let nodeData = diagram.nodes[nodeId];
			
			// If not found, try all node keys to find partial match
			if (!nodeData) {
				const matchingKey = Object.keys(diagram.nodes).find(key => 
					nodeId.includes(key) || key.includes(nodeId)
				);
				if (matchingKey) {
					nodeId = matchingKey;
					nodeData = diagram.nodes[matchingKey];
				}
			}

			if (nodeData) {
				// Set cursor only on the specific node group
				(node as HTMLElement).style.cursor = 'pointer';
				
				// Add click handler
				(node as HTMLElement).addEventListener('click', (e) => {
					e.stopPropagation();
					selectedElement.set({
						type: 'node',
						id: nodeId,
						data: nodeData
					});
					rightPanelOpen.set(true);
				});
				
				// Add hover effect only to the actual node rect/shape, not the whole group
				const nodeShape = node.querySelector('rect, circle, ellipse, polygon, path');
				if (nodeShape) {
					const originalOpacity = (nodeShape as SVGElement).style.opacity || '1';
					node.addEventListener('mouseenter', () => {
						(nodeShape as SVGElement).style.opacity = '0.7';
					});
					node.addEventListener('mouseleave', () => {
						(nodeShape as SVGElement).style.opacity = originalOpacity;
					});
				}
			}
		});

		// Click handlers for edges - be specific
		const edges = containerRef.querySelectorAll('g.edgePath');
		edges.forEach((edgeGroup, index) => {
			const edgeKeys = Object.keys(diagram.edges);
			if (edgeKeys[index]) {
				const edgeKey = edgeKeys[index];
				const edgeData = diagram.edges[edgeKey];

				const edgePath = edgeGroup.querySelector('path');
				if (edgePath) {
					(edgePath as SVGPathElement).style.cursor = 'pointer';
					const originalStrokeWidth = (edgePath as SVGPathElement).style.strokeWidth || '2';
					
					edgeGroup.addEventListener('click', (e) => {
						e.stopPropagation();
						selectedElement.set({
							type: 'edge',
							id: edgeKey,
							data: edgeData
						});
						rightPanelOpen.set(true);
					});
					
					// Add hover effect only to the path
					edgeGroup.addEventListener('mouseenter', () => {
						(edgePath as SVGPathElement).style.strokeWidth = '4';
						(edgePath as SVGPathElement).style.opacity = '0.7';
					});
					edgeGroup.addEventListener('mouseleave', () => {
						(edgePath as SVGPathElement).style.strokeWidth = originalStrokeWidth;
						(edgePath as SVGPathElement).style.opacity = '1';
					});
				}
			}
		});
	}

	onDestroy(() => {
		if (panzoomInstance) {
			panzoomInstance.dispose();
			panzoomInstance = null;
		}
		if (containerRef) {
			containerRef.innerHTML = '';
		}
	});
</script>

<div class="h-full w-full flex flex-col bg-white">
	<div class="border-b border-gray-200 p-4 flex items-center justify-between">
		<div class="flex-1">
			<h2 class="text-lg font-semibold text-gray-900">{diagram.section_title}</h2>
			<p class="text-sm text-gray-600">{diagram.section_description}</p>
		</div>
		
		<!-- Zoom controls -->
		<div class="flex items-center gap-2 ml-4">
			<button
				on:click={zoomOut}
				class="p-2 hover:bg-gray-100 rounded border border-gray-300"
				title="Zoom out"
			>
				<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
					<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M20 12H4" />
				</svg>
			</button>
			<span class="text-sm text-gray-600 min-w-[60px] text-center">
				{Math.round(zoomLevel * 100)}%
			</span>
			<button
				on:click={zoomIn}
				class="p-2 hover:bg-gray-100 rounded border border-gray-300"
				title="Zoom in"
			>
				<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
					<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4" />
				</svg>
			</button>
			<button
				on:click={resetZoom}
				class="p-2 hover:bg-gray-100 rounded border border-gray-300"
				title="Reset zoom"
			>
				<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
					<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
				</svg>
			</button>
		</div>
	</div>

	<div class="flex-1 overflow-hidden relative bg-gray-50">
		<div bind:this={containerRef} class="mermaid-container w-full h-full flex items-center justify-center"></div>
		<div class="absolute bottom-4 left-4 text-xs text-gray-500 bg-white px-2 py-1 rounded border border-gray-200">
			Drag to pan • Scroll to zoom • Click elements for details
		</div>
	</div>
</div>

<style>
	:global(.mermaid-container svg) {
		max-width: 100%;
		height: auto;
	}

	/* Removed global hover styles - handled in JavaScript for precision */
</style>

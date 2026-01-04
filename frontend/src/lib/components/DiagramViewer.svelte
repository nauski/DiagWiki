<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import type { DiagramSection } from '$lib/types';
	import { selectedElement, rightPanelOpen, corruptedDiagrams, diagramCache, openTabs, currentProject } from '$lib/stores';
	import { get } from 'svelte/store';
	import mermaid from 'mermaid';
	import panzoom from 'panzoom';
	import { jsPDF } from 'jspdf';
	import { updateDiagram } from '$lib/api';

	export let diagram: DiagramSection;

	let containerRef: HTMLElement;
	let svgRef: SVGSVGElement | null = null;
	let panzoomInstance: any = null;
	let isRendered = false;
	let currentDiagramId = '';
	let currentDiagramCode = ''; // Track diagram code changes
	let renderCount = 0;
	let zoomLevel = 1;
	let showExportModal = false;
	let copySuccess = false;
	let showRawCodeModal = false;
	let editedCode = '';
	let isUpdating = false;
	let updateError = '';
	let updateSuccess = false;

	type ExportFormat = 'svg' | 'png' | 'jpg' | 'pdf';

	onMount(async () => {
		mermaid.initialize({
			startOnLoad: false,
			theme: 'default',
			securityLevel: 'loose',
			flowchart: { useMaxWidth: true, htmlLabels: true }
		});

		// Check if this diagram is already marked as corrupted
		const isAlreadyCorrupted = $corruptedDiagrams.has(diagram.section_id);
		if (isAlreadyCorrupted) {
			const errorMessage = $corruptedDiagrams.get(diagram.section_id) || 'Unknown error';
			containerRef.innerHTML = `
				<div class="text-red-600 p-4 bg-red-50 rounded-lg border border-red-200">
					<p class="font-semibold">❌ Mermaid rendering error</p>
					<p class="text-sm mb-2 font-mono text-xs">${errorMessage}</p>
					<p class="text-sm text-gray-600 mt-3">Click the "Fix" button in the left panel to automatically correct this diagram.</p>
				</div>
			`;
		} else {
			await renderDiagram();
		}
	});

	// Re-render when diagram changes (section ID or content)
	$: if (diagram && (diagram.section_id !== currentDiagramId || diagram.diagram?.mermaid_code !== currentDiagramCode)) {
		currentDiagramId = diagram.section_id;
		currentDiagramCode = diagram.diagram?.mermaid_code || '';
		
		// Update edited code if raw code modal is open
		if (showRawCodeModal) {
			editedCode = unescapeMermaidCode(currentDiagramCode);
			updateError = '';
			updateSuccess = false;
		}
		
		// Check if this diagram is already marked as corrupted
		const isCorrupted = get(corruptedDiagrams).has(diagram.section_id);
		
		if (isCorrupted) {
			const errorMessage = get(corruptedDiagrams).get(diagram.section_id) || 'Unknown error';
			if (containerRef) {
				containerRef.innerHTML = `
					<div class="text-red-600 p-4 bg-red-50 rounded-lg border border-red-200">
						<p class="font-semibold">❌ Mermaid rendering error</p>
						<p class="text-sm mb-2 font-mono text-xs">${errorMessage}</p>
						<p class="text-sm text-gray-600 mt-3">Click the "Fix" button in the left panel to automatically correct this diagram.</p>
					</div>
				`;
			}
		} else {
			renderDiagram();
		}
	}

	async function renderDiagram() {
		if (!containerRef || !diagram?.diagram?.mermaid_code) return;

		// IMPORTANT: Clear container immediately to prevent showing old diagram
		containerRef.innerHTML = '<div class="text-gray-500">Rendering...</div>';

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
			
			// Clear from corrupted diagrams if render succeeds
			corruptedDiagrams.update(map => {
				const newMap = new Map(map);
				newMap.delete(diagram.section_id);
				return newMap;
			});
		} catch (error) {
			console.error('Failed to render diagram:', error);
			const errorMessage = error instanceof Error ? error.message : 'Unknown error';
			
			// Track in corrupted diagrams store
			corruptedDiagrams.update(map => {
				const newMap = new Map(map);
				newMap.set(diagram.section_id, errorMessage);
				return newMap;
			});
			
			// Show error message in viewer
			containerRef.innerHTML = `
				<div class="text-red-600 p-4 bg-red-50 rounded-lg border border-red-200">
					<p class="font-semibold">❌ Mermaid rendering error</p>
					<p class="text-sm mb-2 font-mono text-xs">${errorMessage}</p>
					<p class="text-sm text-gray-600 mt-3">Click the "Fix" button in the left panel to automatically correct this diagram.</p>
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

	function openExportModal() {
		showExportModal = true;
	}

	function closeExportModal() {
		showExportModal = false;
	}

	async function exportDiagram(format: ExportFormat) {
		if (!svgRef) return;

		try {
			const svgClone = svgRef.cloneNode(true) as SVGSVGElement;
			const svgData = new XMLSerializer().serializeToString(svgClone);
			
			switch (format) {
				case 'svg':
					await exportAsSVG(svgData);
					break;
				case 'png':
					await exportAsImage(svgClone, 'png');
					break;
				case 'jpg':
					await exportAsImage(svgClone, 'jpg');
					break;
				case 'pdf':
					await exportAsPDF(svgClone);
					break;
			}
			
			closeExportModal();
		} catch (error) {
			console.error('Failed to export diagram:', error);
			alert(`Failed to export diagram: ${error instanceof Error ? error.message : 'Unknown error'}`);
		}
	}

	async function exportAsSVG(svgData: string) {
		const svgBlob = new Blob([svgData], { type: 'image/svg+xml;charset=utf-8' });
		const url = URL.createObjectURL(svgBlob);
		downloadFile(url, `${diagram.section_id}.svg`);
		URL.revokeObjectURL(url);
	}

	async function exportAsImage(svgElement: SVGSVGElement, format: 'png' | 'jpg') {
		const canvas = document.createElement('canvas');
		const ctx = canvas.getContext('2d');
		if (!ctx) throw new Error('Could not get canvas context');

		// Get SVG dimensions from viewBox or width/height attributes
		const viewBox = svgElement.viewBox.baseVal;
		const width = viewBox.width || svgElement.width.baseVal.value || 800;
		const height = viewBox.height || svgElement.height.baseVal.value || 600;
		
		const scale = 2; // Higher resolution
		canvas.width = width * scale;
		canvas.height = height * scale;

		// Set background - use light gray (matching the bg-gray-50 from UI)
		ctx.fillStyle = '#f9fafb';
		ctx.fillRect(0, 0, canvas.width, canvas.height);

		// Convert SVG to data URL instead of blob URL to avoid CORS issues
		const svgData = new XMLSerializer().serializeToString(svgElement);
		const svgDataUrl = 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(svgData);

		return new Promise<void>((resolve, reject) => {
			const img = new Image();
			img.onload = () => {
				try {
					ctx.scale(scale, scale);
					ctx.drawImage(img, 0, 0, width, height);
					
					canvas.toBlob((blob) => {
						if (blob) {
							const blobUrl = URL.createObjectURL(blob);
							downloadFile(blobUrl, `${diagram.section_id}.${format}`);
							URL.revokeObjectURL(blobUrl);
							resolve();
						} else {
							reject(new Error('Failed to create blob - canvas may be tainted or empty'));
						}
					}, `image/${format === 'jpg' ? 'jpeg' : 'png'}`, 0.95);
				} catch (err) {
					reject(err);
				}
			};
			img.onerror = (e) => {
				reject(new Error(`Failed to load image: ${e}`));
			};
			img.src = svgDataUrl;
		});
	}

	async function exportAsPDF(svgElement: SVGSVGElement) {
		// Get SVG dimensions from viewBox or width/height attributes
		const viewBox = svgElement.viewBox.baseVal;
		const width = viewBox.width || svgElement.width.baseVal.value || 800;
		const height = viewBox.height || svgElement.height.baseVal.value || 600;

		// First convert SVG to canvas
		const canvas = document.createElement('canvas');
		const ctx = canvas.getContext('2d');
		if (!ctx) throw new Error('Could not get canvas context');

		const scale = 2; // High resolution
		canvas.width = width * scale;
		canvas.height = height * scale;

		// Light gray background
		ctx.fillStyle = '#f9fafb';
		ctx.fillRect(0, 0, canvas.width, canvas.height);

		const svgData = new XMLSerializer().serializeToString(svgElement);
		const svgDataUrl = 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(svgData);

		return new Promise<void>((resolve, reject) => {
			const img = new Image();
			img.onload = () => {
				try {
					ctx.scale(scale, scale);
					ctx.drawImage(img, 0, 0, width, height);
					
					// Convert canvas to image data for PDF
					const imgData = canvas.toDataURL('image/png', 0.95);
					
					// Create PDF with proper dimensions (convert px to mm, assuming 96 DPI)
					const pdfWidth = width * 0.264583; // px to mm
					const pdfHeight = height * 0.264583;
					
					// Use landscape or portrait based on aspect ratio
					const orientation = width > height ? 'landscape' : 'portrait';
					const pdf = new jsPDF({
						orientation: orientation,
						unit: 'mm',
						format: [pdfWidth, pdfHeight]
					});
					
					// Add image to PDF (fill entire page)
					pdf.addImage(imgData, 'PNG', 0, 0, pdfWidth, pdfHeight);
					
					// Save the PDF
					pdf.save(`${diagram.section_id}.pdf`);
					resolve();
				} catch (err) {
					reject(err);
				}
			};
			img.onerror = (e) => {
				reject(new Error(`Failed to load image: ${e}`));
			};
			img.src = svgDataUrl;
		});
	}

	function downloadFile(url: string, filename: string) {
		const link = document.createElement('a');
		link.href = url;
		link.download = filename;
		document.body.appendChild(link);
		link.click();
		document.body.removeChild(link);
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

	function unescapeMermaidCode(code: string): string {
		if (!code) return '';
		// Replace escaped newlines with actual newlines
		return code.replace(/\\n/g, '\n').replace(/\\t/g, '\t');
	}

	function toggleRawCodeModal() {
		showRawCodeModal = !showRawCodeModal;
		if (showRawCodeModal) {
			editedCode = unescapeMermaidCode(diagram?.diagram?.mermaid_code || '');
			updateError = '';
			updateSuccess = false;
		} else {
			copySuccess = false;
		}
	}

	async function copyToClipboard() {
		if (!diagram?.diagram?.mermaid_code) return;
		
		try {
			await navigator.clipboard.writeText(diagram.diagram.mermaid_code);
			copySuccess = true;
			setTimeout(() => {
				copySuccess = false;
			}, 2000);
		} catch (err) {
			console.error('Failed to copy:', err);
		}
	}

	async function handleUpdateDiagram() {
		if (!editedCode || !$currentProject) return;
		
		isUpdating = true;
		updateError = '';
		updateSuccess = false;
		
		try {
			const result = await updateDiagram($currentProject, diagram.section_id, editedCode);
			
			if (result.status === 'success') {
				// Update all stores
				diagramCache.update(cache => {
					cache.set(diagram.section_id, result);
					return cache;
				});
				
				openTabs.update(tabs => {
					const index = tabs.findIndex(t => t.section_id === diagram.section_id);
					if (index !== -1) {
						tabs[index] = result;
					}
					return tabs;
				});
				
				// Remove from corrupted diagrams if it was there
				corruptedDiagrams.update(corrupted => {
					corrupted.delete(diagram.section_id);
					return corrupted;
				});
				
				updateSuccess = true;
				setTimeout(() => {
					updateSuccess = false;
					showRawCodeModal = false;
				}, 1500);
			} else {
				updateError = result.error || 'Failed to update diagram';
			}
		} catch (err) {
			updateError = err instanceof Error ? err.message : 'Failed to update diagram';
		} finally {
			isUpdating = false;
		}
	}
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
			<button
				on:click={openExportModal}
				class="px-2.5 py-1.5 hover:bg-gray-100 rounded border-2 border-gray-800 font-semibold text-xs leading-tight"
				title="Export diagram"
			>
				Export
			</button>
		</div>
	</div>

	<div class="flex-1 relative bg-gray-50">
		<div bind:this={containerRef} class="mermaid-container w-full h-full overflow-auto"></div>
		
		<!-- Raw Code button in top-right corner -->
		<button
			on:click={toggleRawCodeModal}
			class="absolute top-4 right-4 px-3 py-1.5 bg-white hover:bg-gray-100 rounded border border-gray-300 shadow-sm transition-colors text-sm font-medium text-gray-700 z-10"
			title="View raw Mermaid code"
		>
			Raw Code
		</button>
		
		<!-- Raw Code floating panel with animation -->
		{#if showRawCodeModal}
			<div 
				class="absolute top-16 right-4 w-[600px] max-w-[calc(100%-2rem)] bg-white rounded-lg shadow-2xl border border-gray-300 flex flex-col z-20 animate-expand"
				style="height: 60vh; max-height: calc(100vh - 10rem);"
			>
				<div class="flex items-center justify-between px-4 py-3 border-b border-gray-200">
					<h3 class="text-sm font-semibold text-gray-900">Raw Mermaid Code</h3>
					<div class="flex items-center gap-2">
						<button
							on:click={copyToClipboard}
							class="p-1.5 hover:bg-gray-100 rounded transition-colors"
							title="Copy to clipboard"
						>
							{#if copySuccess}
								<svg class="w-4 h-4 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
									<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" />
								</svg>
							{:else}
								<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
									<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
								</svg>
							{/if}
						</button>
						<button
							on:click={toggleRawCodeModal}
							class="p-1.5 hover:bg-gray-100 rounded transition-colors"
							title="Close"
						>
							<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
								<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
							</svg>
						</button>
					</div>
				</div>
				
				<div class="flex-1 overflow-auto bg-gray-50">
					<textarea
						bind:value={editedCode}
						class="w-full h-full p-4 text-xs font-mono text-gray-800 bg-gray-50 border-none resize-none focus:outline-none focus:ring-0"
						placeholder="Enter Mermaid code..."
						spellcheck="false"
					></textarea>
				</div>
				
				{#if updateError}
					<div class="px-4 py-2 bg-red-50 border-t border-red-200 text-xs text-red-600">
						{updateError}
					</div>
				{/if}
				
				<div class="px-4 py-3 border-t border-gray-200 flex gap-2">
					<button
						on:click={handleUpdateDiagram}
						disabled={isUpdating || !editedCode || editedCode === diagram?.diagram?.mermaid_code}
						class="flex-1 px-3 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed text-white rounded font-medium text-sm transition-colors flex items-center justify-center gap-2"
					>
						{#if isUpdating}
							<svg class="animate-spin h-4 w-4" fill="none" viewBox="0 0 24 24">
								<circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
								<path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
							</svg>
							Updating...
						{:else if updateSuccess}
							<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
								<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" />
							</svg>
							Updated!
						{:else}
							Update
						{/if}
					</button>
				</div>
			</div>
		{/if}
		
		<div class="absolute bottom-4 left-4 text-xs text-gray-500 bg-white px-2 py-1 rounded border border-gray-200">
			Drag to pan • Scroll to zoom • Click elements for details
		</div>
	</div>
</div>

<!-- Export Format Modal -->
{#if showExportModal}
	<div class="fixed inset-0 flex items-center justify-center z-50 pointer-events-none">
		<div class="bg-white rounded-lg shadow-2xl p-6 max-w-sm w-full mx-4 pointer-events-auto border border-gray-200" on:click|stopPropagation>
			<h3 class="text-lg font-semibold text-gray-900 mb-4">Export Diagram</h3>
			<p class="text-sm text-gray-600 mb-4">Choose export format:</p>
			
			<div class="space-y-2">
				<button
					on:click={() => exportDiagram('svg')}
					class="w-full text-left px-4 py-3 border-2 border-gray-300 rounded-lg hover:border-blue-500 hover:bg-blue-50 transition-colors"
				>
					<div class="font-semibold text-gray-900">SVG</div>
					<div class="text-xs text-gray-500">Vector format, scalable, small file size</div>
				</button>
				
				<button
					on:click={() => exportDiagram('png')}
					class="w-full text-left px-4 py-3 border-2 border-gray-300 rounded-lg hover:border-blue-500 hover:bg-blue-50 transition-colors"
				>
					<div class="font-semibold text-gray-900">PNG</div>
					<div class="text-xs text-gray-500">High quality with light gray background</div>
				</button>
				
				<button
					on:click={() => exportDiagram('jpg')}
					class="w-full text-left px-4 py-3 border-2 border-gray-300 rounded-lg hover:border-blue-500 hover:bg-blue-50 transition-colors"
				>
					<div class="font-semibold text-gray-900">JPG</div>
					<div class="text-xs text-gray-500">Smaller file size with light gray background</div>
				</button>
				
				<button
					on:click={() => exportDiagram('pdf')}
					class="w-full text-left px-4 py-3 border-2 border-gray-300 rounded-lg hover:border-blue-500 hover:bg-blue-50 transition-colors"
				>
					<div class="font-semibold text-gray-900">PDF</div>
					<div class="text-xs text-gray-500">Document format with embedded image</div>
				</button>
			</div>
			
			<button
				on:click={closeExportModal}
				class="mt-4 w-full px-4 py-2 bg-gray-200 hover:bg-gray-300 rounded-lg font-medium text-gray-700 transition-colors"
			>
				Cancel
			</button>
		</div>
	</div>
{/if}

<style>
	:global(.mermaid-container svg) {
		max-width: 100%;
		height: auto;
	}

	.animate-expand {
		animation: expand 0.2s ease-out;
	}

	@keyframes expand {
		from {
			opacity: 0;
			transform: scale(0.95) translateY(-10px);
		}
		to {
			opacity: 1;
			transform: scale(1) translateY(0);
		}
	}

	/* Removed global hover styles - handled in JavaScript for precision */
</style>

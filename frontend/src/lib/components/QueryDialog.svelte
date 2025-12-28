<script lang="ts">
	import { currentProject, availableSections } from '$lib/stores';
	import { generateSectionDiagram } from '$lib/api';
	import type { WikiSection } from '$lib/types';

	export let isOpen = false;
	export let onClose: () => void;

	let prompt = '';
	let selectedSectionId = '';
	let isGenerating = false;
	let error = '';

	async function handleGenerate() {
		if (!prompt.trim() || !$currentProject || isGenerating) return;

		isGenerating = true;
		error = '';

		try {
			// Find the selected section or use the first one
			let section: WikiSection | undefined;
			
			if (selectedSectionId) {
				section = $availableSections.find(s => s.section_id === selectedSectionId);
			}
			
			if (!section) {
				// Create a custom section based on the prompt
				section = {
					section_id: `custom_${Date.now()}`,
					section_title: prompt, // Use full prompt, backend will extract better title
					section_description: prompt,
					diagram_type: 'flowchart',
					key_concepts: []
				};
			}

			const diagram = await generateSectionDiagram($currentProject, section);
			
			// Add the custom section to availableSections if it's new
			if (section.section_id.startsWith('custom_')) {
				availableSections.update(sections => {
					// Check if already exists
					if (!sections.some(s => s.section_id === section.section_id)) {
						// Use the title from the generated diagram if available
						const updatedSection = {
							...section,
							section_title: diagram.section_title || section.section_title
						};
						return [...sections, updatedSection];
					}
					return sections;
				});
			}
			
			// Close dialog and let parent handle opening the tab
			onClose();
			
			// Dispatch custom event with the diagram
			window.dispatchEvent(new CustomEvent('openDiagram', { detail: diagram }));
		} catch (err) {
			error = err instanceof Error ? err.message : 'Failed to generate diagram';
		} finally {
			isGenerating = false;
		}
	}

	function handleKeydown(event: KeyboardEvent) {
		if (event.key === 'Escape') {
			onClose();
		}
	}

	function handleBackdropClick(event: MouseEvent) {
		if (event.target === event.currentTarget) {
			onClose();
		}
	}
</script>

<svelte:window on:keydown={handleKeydown} />

{#if isOpen}
	<!-- svelte-ignore a11y-click-events-have-key-events -->
	<!-- svelte-ignore a11y-no-noninteractive-element-interactions -->
	<div
		class="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50"
		on:click={handleBackdropClick}
		role="dialog"
		aria-modal="true"
		tabindex="-1"
	>
		<div class="bg-white rounded-lg shadow-xl max-w-2xl w-full mx-4 max-h-[80vh] flex flex-col">
			<!-- Header -->
			<div class="flex items-center justify-between p-4 border-b border-gray-200">
				<h2 class="text-xl font-semibold text-gray-900">Generate Custom Diagram</h2>
				<button
					on:click={onClose}
					class="p-1 hover:bg-gray-100 rounded"
					title="Close"
				>
					<svg class="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
						<path
							fill-rule="evenodd"
							d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z"
							clip-rule="evenodd"
						/>
					</svg>
				</button>
			</div>

			<!-- Content -->
			<div class="p-6 overflow-y-auto flex-1">
				<div class="space-y-4">
					<!-- Prompt Input -->
					<div>
						<label for="prompt" class="block text-sm font-medium text-gray-700 mb-2">
							What would you like to visualize?
						</label>
						<textarea
							id="prompt"
							bind:value={prompt}
							rows="4"
							placeholder="E.g., 'Show me the authentication flow' or 'Explain the data processing pipeline'"
							class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm"
							disabled={isGenerating}
						></textarea>
					</div>

					<!-- Section Selector (Optional) -->
					<div>
						<label for="section" class="block text-sm font-medium text-gray-700 mb-2">
							Base on existing section (optional)
						</label>
						<select
							id="section"
							bind:value={selectedSectionId}
							class="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm"
							disabled={isGenerating}
						>
							<option value="">New custom diagram</option>
							{#each $availableSections as section}
								<option value={section.section_id}>{section.section_title}</option>
							{/each}
						</select>
					</div>

					{#if error}
						<div class="p-3 bg-red-50 border border-red-200 rounded-md">
							<p class="text-red-700 text-sm">{error}</p>
						</div>
					{/if}
				</div>
			</div>

			<!-- Footer -->
			<div class="p-4 border-t border-gray-200 flex items-center justify-end gap-3">
				<button
					on:click={onClose}
					class="px-4 py-2 text-gray-700 hover:bg-gray-100 rounded-md transition-colors text-sm font-medium"
					disabled={isGenerating}
				>
					Cancel
				</button>
				<button
					on:click={handleGenerate}
					disabled={!prompt.trim() || isGenerating}
					class="px-6 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors text-sm font-medium"
				>
					{isGenerating ? 'Generating...' : 'Generate Diagram'}
				</button>
			</div>
		</div>
	</div>
{/if}

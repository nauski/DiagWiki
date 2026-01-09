<script lang="ts">
	import { createEventDispatcher } from 'svelte';

	export let node: any;
	export let selectedFiles: string[];
	export let level: number = 0;

	const dispatch = createEventDispatcher();

	let isExpanded = true; // Expand all folders by default

	function toggleExpand() {
		isExpanded = !isExpanded;
	}

	function handleToggle(filePath: string) {
		dispatch('toggle', filePath);
	}

	function handleFolderClick() {
		dispatch('toggleFolder', node);
	}

	$: isSelected = node.type === 'file' && selectedFiles.includes(node.path);
	// A folder is only considered selected if ALL files within it are selected
	$: hasSelectedChildren = node.type === 'folder' && (() => {
		if (!node.children || node.children.length === 0) return false;
		
		// Get all files recursively
		const allFiles: string[] = [];
		function collectAllFiles(n: any) {
			if (n.type === 'file') {
				allFiles.push(n.path);
			} else if (n.type === 'folder' && n.children) {
				n.children.forEach(collectAllFiles);
			}
		}
		collectAllFiles(node);
		
		// Return true only if we have files AND all are selected
		return allFiles.length > 0 && allFiles.every(filePath => selectedFiles.includes(filePath));
	})();
</script>

{#if node}
	<div style="padding-left: {level * 8}px" class="text-xs">
		{#if node.type === 'folder'}
			<div class="flex items-center gap-1">
				<button
					type="button"
					on:click={toggleExpand}
					class="flex items-center gap-1 py-1 hover:bg-gray-100 rounded transition-colors"
					title="Expand/collapse folder"
			>
					<svg class="w-3 h-3 transition-transform {isExpanded ? 'rotate-90' : ''}" fill="currentColor" viewBox="0 0 20 20">
						<path fill-rule="evenodd" d="M7.293 14.707a1 1 0 010-1.414L10.586 10 7.293 6.707a1 1 0 011.414-1.414l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0z" clip-rule="evenodd" />
					</svg>
				</button>
				<button
					type="button"
					on:click={handleFolderClick}
					class="flex items-center gap-1 py-1 hover:bg-gray-100 rounded flex-1 transition-colors {hasSelectedChildren ? 'font-medium' : ''}"
					title="Click to select/deselect all files in this folder"
				>
					<input
						type="checkbox"
						checked={hasSelectedChildren}
						readonly
						class="w-3 h-3 rounded border-gray-300 text-blue-600 focus:ring-blue-500 pointer-events-none"
					/>
					<svg class="w-3 h-3 text-blue-600" fill="currentColor" viewBox="0 0 20 20">
						<path d="M2 6a2 2 0 012-2h5l2 2h5a2 2 0 012 2v6a2 2 0 01-2 2H4a2 2 0 01-2-2V6z" />
					</svg>
					<span class="text-gray-700">{node.name}</span>
				</button>
			</div>
			{#if isExpanded && node.children}
				{#each node.children as child (child.path)}
					<svelte:self 
						node={child} 
						{selectedFiles}
						level={level + 1}
						on:toggle
						on:toggleFolder
					/>
				{/each}
			{/if}
		{:else if node.type === 'file'}
			<button
				type="button"
				on:click={() => handleToggle(node.path)}
				class="flex items-center gap-1 py-1 hover:bg-gray-100 rounded w-full text-left transition-colors"
			>
				<input
					type="checkbox"
					checked={isSelected}
					readonly
					class="w-3 h-3 rounded border-gray-300 text-blue-600 focus:ring-blue-500 pointer-events-none"
				/>
				<svg class="w-3 h-3 text-gray-500" fill="currentColor" viewBox="0 0 20 20">
					<path fill-rule="evenodd" d="M4 4a2 2 0 012-2h4.586A2 2 0 0112 2.586L15.414 6A2 2 0 0116 7.414V16a2 2 0 01-2 2H6a2 2 0 01-2-2V4z" clip-rule="evenodd" />
				</svg>
				<span class="{isSelected ? 'text-blue-700 font-medium' : 'text-gray-600'}">{node.name}</span>
			</button>
		{/if}
	</div>
{/if}

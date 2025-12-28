<script lang="ts">
	export let node: {
		name: string;
		type: 'file' | 'folder';
		path: string;
		children?: any[];
	};
	export let expandedFolders: Set<string>;
	export let depth: number = 0;

	function toggleFolder() {
		if (expandedFolders.has(node.path)) {
			expandedFolders.delete(node.path);
		} else {
			expandedFolders.add(node.path);
		}
		expandedFolders = expandedFolders; // Trigger reactivity
	}

	$: isExpanded = expandedFolders.has(node.path);
	$: hasChildren = node.children && node.children.length > 0;
</script>

<div style="margin-left: {depth * 12}px">
	{#if node.type === 'folder'}
		<button 
			class="flex items-center gap-1 px-2 py-1 hover:bg-gray-100 rounded text-left w-full text-sm group"
			on:click={toggleFolder}
		>
			{#if hasChildren}
				<svg 
					class="w-3 h-3 transform transition-transform {isExpanded ? 'rotate-90' : ''}" 
					fill="currentColor" 
					viewBox="0 0 20 20"
				>
					<path fill-rule="evenodd" d="M7.293 14.707a1 1 0 010-1.414L10.586 10 7.293 6.707a1 1 0 011.414-1.414l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0z" clip-rule="evenodd" />
				</svg>
			{:else}
				<span class="w-3"></span>
			{/if}
			<svg class="w-4 h-4 text-yellow-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
				<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z" />
			</svg>
			<span class="font-bold text-gray-700 truncate">{node.name}</span>
		</button>
	{:else}
		<div class="flex items-center gap-1 px-2 py-1 text-sm">
			<span class="w-3"></span>
			<svg class="w-4 h-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
				<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
			</svg>
			<span class="text-gray-600 truncate">{node.name}</span>
		</div>
	{/if}
</div>

{#if isExpanded && hasChildren}
	{#each node.children as child}
		<svelte:self node={child} {expandedFolders} depth={depth + 1} />
	{/each}
{/if}

<script lang="ts">
	import { currentProject } from '$lib/stores';
	import { queryWithWiki } from '$lib/api';

	let query = '';
	let isQuerying = false;
	let result: any = null;
	let error = '';
	let isExpanded = true;

	async function handleSubmit() {
		if (!query.trim() || !$currentProject || isQuerying) return;

		isQuerying = true;
		error = '';
		result = null;

		try {
			const response = await queryWithWiki($currentProject, query, true);
			result = response;
			isExpanded = true;
		} catch (err) {
			error = err instanceof Error ? err.message : 'Query failed';
		} finally {
			isQuerying = false;
		}
	}

	function handleKeydown(event: KeyboardEvent) {
		if (event.key === 'Enter' && !event.shiftKey) {
			event.preventDefault();
			handleSubmit();
		}
	}

	function toggleExpand() {
		isExpanded = !isExpanded;
	}
</script>

<div class="border-t border-gray-200 bg-white p-4">
	<div class="flex gap-2">
		<input
			type="text"
			bind:value={query}
			on:keydown={handleKeydown}
			placeholder="Ask a question or request modifications..."
			class="flex-1 px-4 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm"
			disabled={!$currentProject || isQuerying}
		/>
		<button
			on:click={handleSubmit}
			disabled={!$currentProject || isQuerying || !query.trim()}
			class="px-6 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors text-sm font-medium"
		>
			{isQuerying ? 'Querying...' : 'Ask'}
		</button>
	</div>

	{#if error}
		<div class="mt-2 p-3 bg-red-50 border border-red-200 rounded-md">
			<p class="text-red-700 text-sm">{error}</p>
		</div>
	{/if}

	{#if result}
		<div class="mt-2 border border-blue-200 rounded-md bg-blue-50 overflow-hidden" style="max-height: 30vh;">
			<button
				on:click={toggleExpand}
				class="w-full flex items-center justify-between px-3 py-2 bg-blue-100 hover:bg-blue-150 transition-colors"
			>
				<span class="text-xs font-semibold text-blue-700 uppercase">Answer</span>
				<svg
					class="w-4 h-4 transform transition-transform {isExpanded ? 'rotate-180' : ''}"
					fill="currentColor"
					viewBox="0 0 20 20"
				>
					<path fill-rule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clip-rule="evenodd" />
				</svg>
			</button>
			{#if isExpanded}
				<div class="p-3 overflow-y-auto" style="max-height: calc(30vh - 40px);">
					<p class="text-sm text-gray-900 whitespace-pre-wrap">{result.answer?.content || result.answer || JSON.stringify(result, null, 2)}</p>
				</div>
			{/if}
		</div>
	{/if}
</div>

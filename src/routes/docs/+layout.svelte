<script lang="ts">
	import { page } from '$app/stores';
	import { base } from '$app/paths';
	import { ChevronLeft, Home, ExternalLink } from 'lucide-svelte';

	$: currentDoc = $page.params.slug;
</script>

<div class="docs-layout">
	<nav class="docs-nav glass">
		<div class="nav-content">
			<a href="{base}/" class="nav-item home-link">
				<Home size={20} />
				<span>Home</span>
			</a>

			<div class="nav-divider"></div>

			<a
				href="https://github.com/m1rl0k/Context-Engine/tree/test/docs"
				class="nav-item github-link"
				target="_blank"
			>
				<ExternalLink size={20} />
				<span>GitHub Docs</span>
			</a>
		</div>
	</nav>

	<main class="docs-content">
		<slot />
	</main>
</div>

<style lang="scss">
	.docs-layout {
		min-height: 100vh;
		background: var(--gradient-bg);
	}

	.docs-nav {
		position: fixed;
		top: 0;
		left: 0;
		right: 0;
		z-index: 100;
		padding: var(--spacing-sm) 0;
		backdrop-filter: blur(20px) saturate(1.8);
		border-bottom: 1px solid rgba(255, 255, 255, 0.1);
	}

	.nav-content {
		display: flex;
		align-items: center;
		gap: var(--spacing-md);
		max-width: 1200px;
		margin: 0 auto;
		padding: 0 var(--spacing-md);
	}

	.nav-item {
		display: flex;
		align-items: center;
		gap: var(--spacing-xs);
		padding: var(--spacing-xs) var(--spacing-sm);
		border-radius: var(--radius-sm);
		text-decoration: none;
		color: var(--text-secondary);
		transition: all 0.2s ease;
		font-size: 0.9rem;
		font-weight: 500;

		&:hover {
			color: var(--text-primary);
			background: rgba(255, 255, 255, 0.1);
		}
	}

	.nav-divider {
		width: 1px;
		height: 20px;
		background: rgba(255, 255, 255, 0.2);
		margin: 0 var(--spacing-sm);
	}

	.docs-content {
		padding-top: calc(60px + var(--spacing-lg));
		min-height: 100vh;
	}
</style>

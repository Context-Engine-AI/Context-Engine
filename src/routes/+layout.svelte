<script lang="ts">
	import '../app.scss';
	import { Megaphone } from 'lucide-svelte';
	import { page } from '$app/stores';
	import { base } from '$app/paths';

	let { children } = $props();

	// Hide banner on contact page only when coming from demo request
	let hideBanner = $derived(
		$page.route.id === '/contact' && $page.url.searchParams.get('type') === 'demo'
	);
</script>

<div class="bg-gradient"></div>
<div class="bg-overlay"></div>

{#if !hideBanner}
	<div class="launch-banner">
		<div class="banner-content">
			<div class="banner-text">
				<span class="banner-icon">
					<Megaphone size={18} />
				</span>
				<span class="banner-message">Context Engine is launching soon!</span>
				<a href="{base}/contact?type=demo" class="banner-cta">Request Demo</a>
			</div>
		</div>
	</div>
{/if}

<div class="app">
	<main>
		{@render children()}
	</main>
</div>

<style lang="scss">
	.launch-banner {
		position: sticky;
		top: 0;
		z-index: 1000;
		background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
		color: white;
		padding: 0.75rem 1rem;
		box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
		animation: slideDown 0.3s ease-out;
	}

	.banner-content {
		display: flex;
		justify-content: space-between;
		align-items: center;
		max-width: 1200px;
		margin: 0 auto;
	}

	.banner-text {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		flex: 1;
	}

	.banner-icon {
		color: rgba(255, 255, 255, 0.9);
		display: inline-flex;
		align-items: center;
		justify-content: center;
		animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
	}

	@keyframes iconPulse {
		0%,
		100% {
			opacity: 0.9;
			transform: scale(1);
		}
		50% {
			opacity: 1;
			transform: scale(1.1);
		}
	}

	.banner-message {
		font-weight: 500;
		font-size: 0.95rem;
	}

	.banner-cta {
		background: rgba(255, 255, 255, 0.9);
		color: #667eea;
		text-decoration: none;
		padding: 0.4rem 1rem;
		border-radius: 1.25rem;
		font-weight: 600;
		font-size: 0.875rem;
		margin-left: 1rem;
		white-space: nowrap;
		transition: all 0.2s ease;

		&:hover {
			background: white;
			transform: translateY(-1px);
			box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
		}

		&:active {
			transform: translateY(0);
		}
	}

	@keyframes slideDown {
		from {
			transform: translateY(-100%);
			opacity: 0;
		}
		to {
			transform: translateY(0);
			opacity: 1;
		}
	}

	@media (max-width: 768px) {
		.banner-message {
			font-size: 0.875rem;
		}

		.banner-cta {
			font-size: 0.8rem;
			padding: 0.35rem 0.75rem;
			margin-left: 0.75rem;
		}

		.banner-content {
			padding: 0 0.5rem;
		}

		.banner-text {
			gap: 0.25rem;
		}
	}

	.app {
		position: relative;
		z-index: 1;
		min-height: 100vh;
	}

	main {
		min-height: 100vh;
		width: 100%;
	}
</style>

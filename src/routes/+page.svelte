<script lang="ts">
	import { Search, Layers, Brain, Plug, BotOff, Zap } from 'lucide-svelte';
	import { onMount } from 'svelte';

	let mounted = false;

	onMount(() => {
		mounted = true;
	});
</script>

<svelte:head>
	<title>Context Engine - MCP Retrieval Stack</title>
	<meta
		name="description"
		content="MCP retrieval stack for AI coding assistants. Hybrid code search, ReFRAG micro-chunking, local LLM enhancement."
	/>
</svelte:head>

<div class="hero-container">
	<div class="container">
		<!-- Content section -->
		<div class="hero-content" class:mounted>
			<div class="hero-badge glass-subtle">
				<span class="pulse">
					<BotOff size={24} />
				</span>
				<span>MCP Retrieval Stack</span>
			</div>

			<h1 class="hero-title">Context Engine</h1>

			<p class="hero-subtitle">
				Hybrid code search with dense + lexical + reranker. ReFRAG micro-chunking, local LLM prompt
				enhancement, and dual SSE/RMCP endpoints. One command deploys Qdrant-powered indexing for
				Cursor, Windsurf, Roo, Cline, Codex, and any MCP client.
			</p>

			<div class="hero-actions">
				<a href="https://github.com/m1rl0k/Context-Engine" class="btn-primary" target="_blank">
					<svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor" class="github-icon">
						<path
							d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"
						/>
					</svg>
					View on GitHub
				</a>

				<a href="#features" class="btn-secondary"> Explore Features </a>
			</div>
		</div>

		<!-- Glass card first on mobile -->
		<div class="hero-visual">
			<div class="glass-card glass-strong">
				<div class="code-preview">
					<div class="code-header">
						<div class="code-dots">
							<span></span>
							<span></span>
							<span></span>
						</div>
						<span class="code-title">Context Engine</span>
					</div>
					<div class="code-content">
						<div class="code-line">
							<span class="code-comment"># Deploy Context Engine</span>
						</div>
						<div class="code-line">
							<span class="code-keyword">docker</span>
							<span class="code-string">compose up -d</span>
						</div>
						<div class="code-line">
							<span class="code-comment"># Index your codebase</span>
						</div>
						<div class="code-line">
							<span class="code-keyword">curl</span>
							<span class="code-string">-X POST /index</span>
						</div>
						<div class="code-line">
							<span class="code-comment"># Ready for MCP clients! <Zap size={16} /></span>
						</div>
					</div>
				</div>
			</div>
		</div>
	</div>
</div>

<section id="features" class="features-section">
	<div class="container">
		<h2 class="section-title">Powerful Features</h2>

		<div class="features-grid">
			<div class="feature-card glass">
				<div class="feature-icon">
					<Search size={32} />
				</div>
				<h3>Hybrid Search</h3>
				<p>Dense + lexical + reranker for precise code discovery</p>
			</div>

			<div class="feature-card glass">
				<div class="feature-icon">
					<Layers size={32} />
				</div>
				<h3>ReFRAG Chunking</h3>
				<p>Micro-chunking for optimal context retrieval</p>
			</div>

			<div class="feature-card glass">
				<div class="feature-icon">
					<Brain size={32} />
				</div>
				<h3>Local LLM</h3>
				<p>Enhanced prompts with local model processing</p>
			</div>

			<div class="feature-card glass">
				<div class="feature-icon">
					<Plug size={32} />
				</div>
				<h3>MCP Compatible</h3>
				<p>Works with Cursor, Windsurf, Roo, Cline, and more</p>
			</div>
		</div>
	</div>
</section>

<style lang="scss">
	.hero-container {
		display: flex;
		align-items: center;
		padding: var(--spacing-xl) 0 0 0; // Remove bottom padding entirely

		.container {
			display: grid;
			grid-template-columns: 1fr;
			gap: var(--spacing-xl);
			align-items: center;
			max-width: 1200px;
			margin: 0 auto;
			padding: 0 var(--spacing-md);

			@media (min-width: 1024px) {
				grid-template-columns: 1fr 1fr;
				gap: var(--spacing-xl);
			}
		}
	}

	.hero-content {
		opacity: 0;
		transform: translateY(30px);
		transition: all 0.8s cubic-bezier(0.25, 0.46, 0.45, 0.94);
		text-align: center;
		max-width: 600px;
		margin: 0 auto;

		&.mounted {
			opacity: 1;
			transform: translateY(0);
		}

		@media (min-width: 1024px) {
			text-align: left;
			margin: 0;
			max-width: none;
		}
	}

	.hero-badge {
		display: inline-flex;
		align-items: center;
		gap: var(--spacing-xs);
		padding: var(--spacing-xs) var(--spacing-sm);
		margin-bottom: var(--spacing-md);
		font-size: 0.9rem;
		font-weight: 500;
		color: var(--text-secondary);
	}

	.hero-title {
		margin-bottom: var(--spacing-md);
		background: linear-gradient(135deg, #fff 0%, #a0a0a0 100%);
		-webkit-background-clip: text;
		-webkit-text-fill-color: transparent;
		background-clip: text;
	}

	.hero-subtitle {
		font-size: 1.2rem;
		margin-bottom: var(--spacing-lg);
		max-width: 600px;
		line-height: 1.6;
		margin-left: auto;
		margin-right: auto;

		@media (min-width: 1024px) {
			margin-left: 0;
			margin-right: 0;
		}
	}

	.hero-actions {
		display: flex;
		gap: var(--spacing-sm);
		align-items: center;
		flex-wrap: wrap;
		justify-content: center;
		margin-top: var(--spacing-lg);

		@media (min-width: 1024px) {
			justify-content: flex-start;
		}
	}

	.github-icon {
		margin-right: var(--spacing-xs);
	}

	.hero-visual {
		display: flex;
		justify-content: center;
		order: -1;

		@media (min-width: 1024px) {
			order: 0;
			justify-content: center;
		}
	}

	.glass-card {
		padding: var(--spacing-md);
		max-width: 400px;
		width: 100%;
		margin: 0;
		position: relative;

		@media (min-width: 1024px) {
			max-width: 450px;
		}
	}

	.code-preview {
		font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
		font-size: 0.9rem;
		color: var(--text-primary);
	}

	.code-header {
		display: flex;
		align-items: center;
		justify-content: space-between;
		padding-bottom: var(--spacing-sm);
		margin-bottom: var(--spacing-sm);
		border-bottom: 1px solid rgba(255, 255, 255, 0.1);
	}

	.code-dots {
		display: flex;
		gap: var(--spacing-xs);

		span {
			width: 8px;
			height: 8px;
			border-radius: 50%;
			background: rgba(255, 255, 255, 0.3);

			&:nth-child(1) {
				background: #ff5f56;
			}
			&:nth-child(2) {
				background: #ffbd2e;
			}
			&:nth-child(3) {
				background: #27ca3f;
			}
		}
	}

	.code-title {
		font-size: 0.8rem;
		color: var(--text-muted);
	}

	.code-line {
		margin-bottom: var(--spacing-xs);
		line-height: 1.5;
	}

	.code-keyword {
		color: #a78bfa; // Brighter purple for better contrast
		font-weight: 600;
	}

	.code-string {
		color: #34d399; // Brighter green for better contrast
		font-weight: 500; // Add slight weight for visibility
	}

	.code-comment {
		color: #cbd5e1; // Much lighter gray for better readability
		font-style: italic;
	}

	.features-section {
		padding: var(--spacing-sm) 0; // Smaller padding
		margin-top: calc(-1 * var(--spacing-xl)); // Pull section up closer
	}

	.section-title {
		text-align: center;
		margin-bottom: var(--spacing-xl);
		color: var(--text-primary);
	}

	.features-grid {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
		gap: var(--spacing-md);
		max-width: 1000px;
		margin: 0 auto;
	}

	.feature-card {
		padding: var(--spacing-md);
		text-align: center;
		transition: all 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94);

		&:hover {
			transform: translateY(-5px);
			background: rgba(255, 255, 255, 0.15);
		}
	}

	.feature-icon {
		font-size: 2rem;
		margin-bottom: var(--spacing-sm);
		display: block;
	}

	.feature-card h3 {
		color: var(--text-primary);
		margin-bottom: var(--spacing-xs);
		font-size: 1.2rem;
	}

	.feature-card p {
		color: var(--text-secondary);
		font-size: 0.95rem;
		margin: 0;
	}

	.pulse {
		display: inline-flex;
		align-items: center;
		justify-content: center;
		animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
	}

	@keyframes pulse {
		0%,
		100% {
			opacity: 1;
			transform: scale(1);
		}
		50% {
			opacity: 0.7;
			transform: scale(1.05);
		}
	}
</style>

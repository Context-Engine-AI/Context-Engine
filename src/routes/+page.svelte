<script lang="ts">
	import {
		Search,
		Layers,
		Brain,
		Plug,
		BotOff,
		Zap,
		Rocket,
		BookOpen,
		Play,
		Code,
		Settings,
		Wrench,
		ExternalLink
	} from 'lucide-svelte';
	import { onMount } from 'svelte';
	import { base } from '$app/paths';

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

<section id="quick-start" class="quick-start-section">
	<div class="container">
		<h2 class="section-title"><Rocket size={24} /> Get Started in Minutes</h2>

		<div class="quick-start-flow">
			<div class="quick-start-card glass">
				<div class="step-number">1</div>
				<h3>Deploy</h3>
				<div class="code-snippet">
					<code>docker compose up -d</code>
				</div>
				<p>Start Context Engine with one command</p>
			</div>

			<div class="quick-start-card glass">
				<div class="step-number">2</div>
				<h3>Configure</h3>
				<div class="code-snippet">
					<code>cp ctx_config.example.json ctx_config.json</code>
				</div>
				<p>Set up your preferences</p>
			</div>

			<div class="quick-start-card glass">
				<div class="step-number">3</div>
				<h3>Index</h3>
				<div class="code-snippet">
					<code>curl -X POST localhost:8000/index</code>
				</div>
				<p>Start indexing your codebase</p>
			</div>
		</div>

		<div class="quick-start-cta">
			<a href="{base}/docs/getting-started" class="btn-primary"> Complete Setup Guide </a>
			<a href="{base}/docs/configuration" class="btn-secondary"> Configuration Reference </a>
		</div>
	</div>
</section>

<section id="documentation" class="docs-section">
	<div class="container">
		<h2 class="section-title"><BookOpen size={24} /> Comprehensive Documentation</h2>
		<p class="docs-intro">Everything you need to master Context Engine</p>

		<div class="docs-grid">
			<div class="docs-category glass">
				<div class="docs-icon"><Play size={28} /></div>
				<h3>Getting Started</h3>
				<div class="docs-links">
					<a href="{base}/docs/getting-started">Quick Start Guide</a>
					<a href="{base}/docs/configuration">Configuration</a>
					<a href="{base}/docs/development">Development Setup</a>
				</div>
			</div>

			<div class="docs-category glass">
				<div class="docs-icon"><Code size={28} /></div>
				<h3>Core Features</h3>
				<div class="docs-links">
					<a href="{base}/docs/architecture">Architecture Overview</a>
					<a href="{base}/docs/mcp-api">MCP API Reference</a>
					<a
						href="https://github.com/m1rl0k/Context-Engine/blob/test/docs/CTX_CLI.md"
						target="_blank">CLI Tool <ExternalLink size={14} /></a
					>
					<a
						href="https://github.com/m1rl0k/Context-Engine/blob/test/docs/IDE_CLIENTS.md"
						target="_blank">IDE Clients <ExternalLink size={14} /></a
					>
				</div>
			</div>

			<div class="docs-category glass">
				<div class="docs-icon"><Settings size={28} /></div>
				<h3>Advanced Topics</h3>
				<div class="docs-links">
					<a
						href="https://github.com/m1rl0k/Context-Engine/blob/test/docs/MULTI_REPO_COLLECTIONS.md"
						target="_blank">Multi-Repo Setup <ExternalLink size={14} /></a
					>
					<a
						href="https://github.com/m1rl0k/Context-Engine/blob/test/docs/commit-indexing/overview.md"
						target="_blank">Commit Indexing <ExternalLink size={14} /></a
					>
					<a
						href="https://github.com/m1rl0k/Context-Engine/blob/test/docs/MEMORY_GUIDE.md"
						target="_blank">Memory Management <ExternalLink size={14} /></a
					>
					<a
						href="https://github.com/m1rl0k/Context-Engine/blob/test/docs/vscode-extension.md"
						target="_blank">VS Code Extension <ExternalLink size={14} /></a
					>
				</div>
			</div>

			<div class="docs-category glass">
				<div class="docs-icon"><Wrench size={28} /></div>
				<h3>Operations</h3>
				<div class="docs-links">
					<a href="{base}/docs/troubleshooting">Troubleshooting</a>
					<a
						href="https://github.com/m1rl0k/Context-Engine/blob/test/docs/BENCHMARKS.md"
						target="_blank">Benchmarks <ExternalLink size={14} /></a
					>
					<a
						href="https://github.com/m1rl0k/Context-Engine/blob/test/docs/OBSERVABILITY.md"
						target="_blank">Observability <ExternalLink size={14} /></a
					>
					<a
						href="https://github.com/m1rl0k/Context-Engine/blob/test/docs/CONFIG_DRIFT.md"
						target="_blank">Config Drift <ExternalLink size={14} /></a
					>
				</div>
			</div>
		</div>

		<div class="docs-footer">
			<a
				href="https://github.com/m1rl0k/Context-Engine/tree/test/docs"
				class="btn-primary"
				target="_blank"
			>
				<ExternalLink size={20} /> Browse All Documentation
			</a>
		</div>
	</div>
</section>

<style lang="scss">
	.hero-container {
		display: flex;
		align-items: center;
		padding: var(--spacing-xl) 0; // Add back bottom padding

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

	// ...existing code...

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

	// Quick Start Section
	.quick-start-section {
		padding: var(--spacing-xl) 0;
		background: linear-gradient(135deg, rgba(139, 92, 246, 0.05) 0%, rgba(59, 130, 246, 0.05) 100%);
	}

	.quick-start-flow {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
		gap: var(--spacing-lg);
		max-width: 1000px;
		margin: 0 auto var(--spacing-lg);
	}

	.quick-start-card {
		padding: var(--spacing-lg);
		text-align: center;
		transition: all 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94);
		position: relative;

		&:hover {
			transform: translateY(-8px);
			background: rgba(255, 255, 255, 0.18);
		}

		h3 {
			color: var(--text-primary);
			margin: var(--spacing-sm) 0;
			font-size: 1.3rem;
		}

		p {
			color: var(--text-secondary);
			margin: var(--spacing-sm) 0 0;
			font-size: 0.95rem;
		}
	}

	.step-number {
		width: 40px;
		height: 40px;
		border-radius: 50%;
		background: linear-gradient(135deg, #8b5cf6, #3b82f6);
		color: white;
		display: flex;
		align-items: center;
		justify-content: center;
		font-weight: 600;
		margin: 0 auto var(--spacing-sm);
		font-size: 1.1rem;
		box-shadow: 0 4px 12px rgba(139, 92, 246, 0.3);
	}

	.code-snippet {
		background: rgba(0, 0, 0, 0.4);
		border-radius: var(--radius-sm);
		padding: var(--spacing-sm);
		margin: var(--spacing-sm) 0;
		border: 1px solid rgba(255, 255, 255, 0.1);

		code {
			font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
			font-size: 0.9rem;
			color: #10b981;
			font-weight: 500;
		}
	}

	.quick-start-cta {
		display: flex;
		gap: var(--spacing-sm);
		justify-content: center;
		flex-wrap: wrap;
		margin-top: var(--spacing-lg);
	}

	// Documentation Section
	.docs-section {
		padding: var(--spacing-xl) 0;
		background: linear-gradient(135deg, rgba(59, 130, 246, 0.03) 0%, rgba(139, 92, 246, 0.03) 100%);
	}

	.docs-intro {
		text-align: center;
		color: var(--text-secondary);
		font-size: 1.1rem;
		margin-bottom: var(--spacing-xl);
		max-width: 600px;
		margin-left: auto;
		margin-right: auto;
	}

	.docs-grid {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
		gap: var(--spacing-lg);
		max-width: 1200px;
		margin: 0 auto var(--spacing-xl);
	}

	.docs-category {
		padding: var(--spacing-lg);
		transition: all 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94);

		&:hover {
			transform: translateY(-5px);
			background: rgba(255, 255, 255, 0.15);
		}

		h3 {
			color: var(--text-primary);
			margin: var(--spacing-sm) 0 var(--spacing-md);
			font-size: 1.3rem;
			display: flex;
			align-items: center;
			gap: var(--spacing-xs);
		}
	}

	.docs-icon {
		font-size: 1.8rem;
		display: block;
		margin-bottom: var(--spacing-sm);
	}

	.docs-links {
		display: flex;
		flex-direction: column;
		gap: var(--spacing-xs);

		a {
			color: var(--text-secondary);
			text-decoration: none;
			padding: var(--spacing-xs) 0;
			border-radius: var(--radius-sm);
			transition: all 0.2s ease;
			font-size: 0.95rem;

			&:hover {
				color: var(--text-primary);
				padding-left: var(--spacing-xs);
				background: rgba(255, 255, 255, 0.05);
			}
		}
	}

	.docs-footer {
		text-align: center;
		margin-top: var(--spacing-lg);
	}
</style>

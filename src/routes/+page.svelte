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
		ExternalLink,
		Star,
		GitFork,
		Activity,
		Shield,
		Database,
		Cpu,
		Target,
		Users
	} from 'lucide-svelte';
	import { onMount } from 'svelte';
	import { base } from '$app/paths';
	import type { PageData } from './$types';

	export let data: PageData;

	let mounted = false;
	let showAllDifferentiators = false;

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
				Self-hosted AI retrieval stack with hybrid search, micro-chunking, and pluggable models. One
				command deploys enterprise-grade code indexing for any MCP client.
			</p>

			<div class="hero-actions">
				<a href="#quick-start" class="btn-primary cta-main">
					<Zap size={20} />
					Get Started
				</a>

				<a href="https://github.com/m1rl0k/Context-Engine" class="btn-secondary" target="_blank">
					<svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor" class="github-icon">
						<path
							d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"
						/>
					</svg>
					View on GitHub
				</a>
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

<!-- Metrics Section -->
<section class="metrics-section">
	<div class="container">
		<div class="metrics-grid">
			<div class="metric-card glass">
				<div class="metric-icon">
					<Star size={24} />
				</div>
				<div class="metric-content">
					<div class="metric-value">{data.metrics.stars}</div>
					<div class="metric-label">GitHub Stars</div>
				</div>
			</div>

			<div class="metric-card glass">
				<div class="metric-icon">
					<GitFork size={24} />
				</div>
				<div class="metric-content">
					<div class="metric-value">{data.metrics.forks}</div>
					<div class="metric-label">Forks</div>
				</div>
			</div>

			<div class="metric-card glass">
				<div class="metric-icon">
					<Code size={24} />
				</div>
				<div class="metric-content">
					<div class="metric-value">{data.metrics.language}</div>
					<div class="metric-label">Primary Language</div>
				</div>
			</div>

			<div class="metric-card glass">
				<div class="metric-icon">
					<Activity size={24} />
				</div>
				<div class="metric-content">
					<div class="metric-value">{data.metrics.status === 'live' ? 'Live' : 'Demo'}</div>
					<div class="metric-label">Data Status</div>
				</div>
			</div>
		</div>

		{#if data.metrics.languages && data.metrics.languages.length > 0}
			<div class="languages-section">
				<h3 class="languages-title">Technology Stack</h3>
				<div class="languages-list">
					{#each data.metrics.languages.slice(0, 4) as language}
						<div class="language-item">
							<div
								class="language-color"
								style="background-color: {language.color || '#gray'}"
							></div>
							<span class="language-name">{language.name}</span>
							<span class="language-percentage">{language.percentage}%</span>
						</div>
					{/each}
				</div>
			</div>
		{/if}
	</div>
</section>

<!-- Key Differentiators Section -->
<section class="differentiators-section">
	<div class="container">
		<div class="differentiators-header">
			<h2 class="section-title">Context‑Engine — Key Differentiators</h2>
			<p class="differentiators-tagline glass-subtle">
				"Self‑hosted, code‑aware retrieval and context compression layer for AI agents - hybrid
				search, deep AST indexing, and pluggable models built for private, enterprise‑grade
				deployment."
			</p>
		</div>

		<div class="differentiators-grid">
			<div class="differentiator-card glass">
				<div class="differentiator-icon security">
					<Shield size={28} />
				</div>
				<h3>Fully Self‑Hosted</h3>
				<p>
					Run on your infra, your vector DB, your LLM, your rules. No vendor lock‑in, complete
					control over your data and deployment.
				</p>
			</div>

			<div class="differentiator-card glass">
				<div class="differentiator-icon performance">
					<Target size={28} />
				</div>
				<h3>Hybrid Retrieval by Design</h3>
				<p>
					Dense + lexical fusion, path priors, symbol boosts, recency weighting, and MMR for
					diversity — precision engineered for code.
				</p>
			</div>

			<div class="differentiator-card glass">
				<div class="differentiator-icon technical">
					<Code size={28} />
				</div>
				<h3>Deep Code‑Aware Indexing</h3>
				<p>
					AST symbols/imports/calls, semantic chunking, optional micro‑chunks (ReFRAG) for tighter
					context windows and precise retrieval.
				</p>
			</div>

			<div class="differentiator-card glass">
				<div class="differentiator-icon innovation">
					<Cpu size={28} />
				</div>
				<h3>Pluggable Models</h3>
				<p>
					Swap embeddings/rerankers per workload or hardware budget — scale from laptop to cluster
					with the same architecture.
				</p>
			</div>
		</div>

		<div class="differentiators-toggle">
			<button
				class="btn-secondary expand-btn"
				on:click={() => (showAllDifferentiators = !showAllDifferentiators)}
			>
				{showAllDifferentiators ? 'Show Less' : 'View All Differentiators'}
			</button>
		</div>

		{#if showAllDifferentiators}
			<div class="differentiators-grid additional-differentiators">
				<div class="differentiator-card glass">
					<div class="differentiator-icon innovation">
						<Brain size={28} />
					</div>
					<h3>Incremental Learning Loop</h3>
					<p>
						Optional in‑process learning reranker with hot‑reloaded weights and convergence tracking
						for continuous improvement.
					</p>
				</div>

				<div class="differentiator-card glass">
					<div class="differentiator-icon technical">
						<Plug size={28} />
					</div>
					<h3>MCP‑Native Architecture</h3>
					<p>
						Acts as a retrieval backbone for tool‑using agents and private MCP deployments — built
						for the agent ecosystem.
					</p>
				</div>

				<div class="differentiator-card glass">
					<div class="differentiator-icon performance">
						<Database size={28} />
					</div>
					<h3>Reproducible Benchmarking</h3>
					<p>
						Built‑in benchmark suite for CoSQA/CoIR/SWE‑bench with full env snapshots — measure what
						matters for your use case.
					</p>
				</div>

				<div class="differentiator-card glass">
					<div class="differentiator-icon security">
						<Users size={28} />
					</div>
					<h3>Enterprise‑Grade Privacy</h3>
					<p>
						Zero telemetry, air‑gapped deployment support, and complete data sovereignty for
						sensitive codebases and compliance requirements.
					</p>
				</div>
			</div>
		{/if}
	</div>
</section>

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
			<div class="quick-start-unified-card glass">
				<div class="quick-start-header">
					<h3>Deploy Context Engine</h3>
					<p>Three simple commands to get running</p>
				</div>

				<div class="terminal-block">
					<div class="terminal-header">
						<div class="terminal-dots">
							<span></span>
							<span></span>
							<span></span>
						</div>
						<span class="terminal-title">Quick Start</span>
					</div>
					<div class="terminal-content">
						<div class="terminal-line">
							<span class="terminal-prompt">$</span>
							<span class="terminal-command">docker compose up -d</span>
							<span class="terminal-comment"># Deploy Context Engine</span>
						</div>
						<div class="terminal-line">
							<span class="terminal-prompt">$</span>
							<span class="terminal-command">cp ctx_config.example.json ctx_config.json</span>
							<span class="terminal-comment"># Configure</span>
						</div>
						<div class="terminal-line">
							<span class="terminal-prompt">$</span>
							<span class="terminal-command">curl -X POST localhost:8000/index</span>
							<span class="terminal-comment"># Start indexing</span>
						</div>
					</div>
				</div>
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
		color: rgba(255, 255, 255, 1);
		background: rgba(139, 92, 246, 0.25);
		padding: var(--spacing-sm);
		border-radius: var(--radius-sm);
		display: inline-flex;
		align-items: center;
		justify-content: center;
		width: 56px;
		height: 56px;
		margin-bottom: var(--spacing-sm);
		font-size: 1.5rem;
		transition: all 0.3s ease;
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
	}

	.quick-start-flow {
		display: flex;
		justify-content: center;
		max-width: 800px;
		margin: 0 auto var(--spacing-lg);
	}

	.quick-start-unified-card {
		padding: var(--spacing-xl);
		transition: all 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94);
		width: 100%;
		max-width: 700px;

		&:hover {
			transform: translateY(-8px);
			background: rgba(255, 255, 255, 0.18);
		}
	}

	.quick-start-header {
		text-align: center;
		margin-bottom: var(--spacing-lg);

		h3 {
			color: var(--text-primary);
			margin: var(--spacing-sm) 0 var(--spacing-xs);
			font-size: 1.4rem;
		}

		p {
			color: var(--text-secondary);
			margin: 0;
			font-size: 1rem;
		}
	}

	.terminal-block {
		margin-top: var(--spacing-md);
	}

	.terminal-header {
		background: rgba(0, 0, 0, 0.3);
		padding: var(--spacing-sm);
		border-radius: var(--radius-sm) var(--radius-sm) 0 0;
		display: flex;
		align-items: center;
		justify-content: space-between;
		border: 1px solid rgba(255, 255, 255, 0.1);
		border-bottom: none;
	}

	.terminal-dots {
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

	.terminal-title {
		font-size: 0.8rem;
		color: var(--text-muted);
		font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
	}

	.terminal-content {
		background: rgba(0, 0, 0, 0.5);
		padding: var(--spacing-md);
		border-radius: 0 0 var(--radius-sm) var(--radius-sm);
		border: 1px solid rgba(255, 255, 255, 0.1);
		border-top: none;
		font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
	}

	.terminal-line {
		display: flex;
		align-items: center;
		margin-bottom: var(--spacing-xs);
		line-height: 1.4;

		&:last-child {
			margin-bottom: 0;
		}
	}

	.terminal-prompt {
		color: #22d3ee;
		margin-right: var(--spacing-xs);
		font-weight: 600;
	}

	.terminal-command {
		color: #10b981;
		font-weight: 500;
		white-space: nowrap;
		margin-right: var(--spacing-sm);
	}

	.terminal-comment {
		color: #94a3b8;
		font-style: italic;
		font-size: 0.9rem;
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
		color: rgba(255, 255, 255, 1);
		background: rgba(139, 92, 246, 0.25);
		padding: var(--spacing-sm);
		border-radius: var(--radius-sm);
		display: inline-flex;
		align-items: center;
		justify-content: center;
		width: 56px;
		height: 56px;
		margin-bottom: var(--spacing-sm);
		font-size: 1.5rem;
		transition: all 0.3s ease;
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

	// Key Differentiators Section
	.differentiators-section {
		padding: var(--spacing-xl) 0;
	}

	.differentiators-header {
		text-align: center;
		margin-bottom: var(--spacing-xl);
	}

	.differentiators-tagline {
		display: inline-block;
		padding: var(--spacing-md) var(--spacing-lg);
		margin-top: var(--spacing-md);
		font-style: italic;
		color: var(--text-secondary);
		font-size: 1rem;
		line-height: 1.5;
		max-width: 800px;
		border-left: 3px solid rgba(139, 92, 246, 0.5);
	}

	.differentiators-grid {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
		gap: var(--spacing-lg);
		max-width: 1400px;
		margin: 0 auto;
	}

	.differentiator-card {
		padding: var(--spacing-lg);
		transition: all 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94);
		border-left: 3px solid transparent;
		opacity: 0;
		transform: translateY(20px);
		animation: fadeInUp 0.6s ease forwards;

		&:nth-child(1) {
			animation-delay: 0.1s;
		}
		&:nth-child(2) {
			animation-delay: 0.2s;
		}
		&:nth-child(3) {
			animation-delay: 0.3s;
		}
		&:nth-child(4) {
			animation-delay: 0.4s;
		}

		&:hover {
			transform: translateY(-8px) scale(1.02);
			background: rgba(255, 255, 255, 0.12);
			border-left-color: rgba(139, 92, 246, 0.6);
			box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
		}

		h3 {
			color: var(--text-primary);
			margin: var(--spacing-sm) 0 var(--spacing-sm);
			font-size: 1.2rem;
			font-weight: 600;
		}

		p {
			color: var(--text-secondary);
			font-size: 0.95rem;
			line-height: 1.6;
			margin: 0;
		}
	}

	@keyframes fadeInUp {
		to {
			opacity: 1;
			transform: translateY(0);
		}
	}

	.differentiator-icon {
		color: rgba(255, 255, 255, 1);
		background: rgba(139, 92, 246, 0.25);
		padding: var(--spacing-sm);
		border-radius: var(--radius-sm);
		display: inline-flex;
		align-items: center;
		justify-content: center;
		width: 56px;
		height: 56px;
		margin-bottom: var(--spacing-sm);
		transition: all 0.3s ease;

		&.security {
			color: rgba(34, 197, 94, 1);
			background: rgba(34, 197, 94, 0.25);
		}

		&.performance {
			color: rgba(96, 165, 250, 1);
			background: rgba(59, 130, 246, 0.25);
		}

		&.technical {
			color: rgba(196, 145, 251, 1);
			background: rgba(168, 85, 247, 0.25);
		}

		&.innovation {
			color: rgba(251, 146, 60, 1);
			background: rgba(251, 146, 60, 0.25);
		}
	}

	.differentiators-toggle {
		text-align: center;
		margin-top: var(--spacing-lg);
	}

	.expand-btn {
		padding: var(--spacing-sm) var(--spacing-lg);
		transition: all 0.3s ease;

		&:hover {
			transform: translateY(-2px);
			box-shadow: 0 8px 25px rgba(139, 92, 246, 0.2);
		}
	}

	.additional-differentiators {
		margin-top: var(--spacing-lg);
		animation: slideIn 0.5s cubic-bezier(0.25, 0.46, 0.45, 0.94);
	}

	@keyframes slideIn {
		from {
			opacity: 0;
			transform: translateY(20px);
		}
		to {
			opacity: 1;
			transform: translateY(0);
		}
	}

	.cta-main {
		background: linear-gradient(135deg, #8b5cf6, #3b82f6);
		color: white;
		box-shadow: 0 8px 25px rgba(139, 92, 246, 0.3);
		font-weight: 600;
		gap: var(--spacing-xs);
		display: inline-flex;
		align-items: center;

		&:hover {
			background: linear-gradient(135deg, #7c3aed, #2563eb);
			box-shadow: 0 12px 30px rgba(139, 92, 246, 0.4);
			transform: translateY(-3px);
		}
	}

	// Metrics Section
	.metrics-section {
		padding: var(--spacing-lg) 0;
	}

	.metrics-grid {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
		gap: var(--spacing-lg);
		margin-bottom: var(--spacing-xl);
	}

	.metric-card {
		padding: var(--spacing-lg);
		display: flex;
		align-items: center;
		gap: var(--spacing-md);
		transition: all 0.3s ease;

		&:hover {
			transform: translateY(-3px);
			background: rgba(255, 255, 255, 0.15);
		}
	}

	.metric-icon {
		color: rgba(255, 255, 255, 0.9);
		background: rgba(139, 92, 246, 0.2);
		padding: var(--spacing-sm);
		border-radius: var(--radius-sm);
		display: flex;
		align-items: center;
		justify-content: center;
		min-width: 48px;
		height: 48px;
	}

	.metric-content {
		flex: 1;
	}

	.metric-value {
		font-size: 1.5rem;
		font-weight: 700;
		color: var(--text-primary);
		line-height: 1.2;
	}

	.metric-label {
		font-size: 0.85rem;
		color: var(--text-secondary);
		margin-top: 2px;
	}

	.languages-section {
		max-width: 600px;
		margin: 0 auto;
		text-align: center;
	}

	.languages-title {
		color: var(--text-primary);
		margin-bottom: var(--spacing-md);
		font-size: 1.2rem;
		font-weight: 600;
	}

	.languages-list {
		display: flex;
		flex-wrap: wrap;
		gap: var(--spacing-md);
		justify-content: center;
	}

	.language-item {
		display: flex;
		align-items: center;
		gap: var(--spacing-xs);
		padding: var(--spacing-xs) var(--spacing-sm);
		background: rgba(255, 255, 255, 0.05);
		border-radius: var(--radius-sm);
		border: 1px solid rgba(255, 255, 255, 0.1);
	}

	.language-color {
		width: 12px;
		height: 12px;
		border-radius: 50%;
	}

	.language-name {
		color: var(--text-primary);
		font-size: 0.9rem;
		font-weight: 500;
	}

	.language-percentage {
		color: var(--text-muted);
		font-size: 0.8rem;
	}

	@media (max-width: 768px) {
		.metrics-grid {
			grid-template-columns: repeat(2, 1fr);
			gap: var(--spacing-md);
		}

		.metric-card {
			padding: var(--spacing-md);
			gap: var(--spacing-sm);
		}

		.metric-value {
			font-size: 1.2rem;
		}

		.languages-list {
			justify-content: center;
		}

		.differentiators-grid {
			grid-template-columns: 1fr;
			gap: var(--spacing-md);
		}

		.differentiator-card {
			padding: var(--spacing-md);
		}

		.differentiators-tagline {
			font-size: 0.9rem;
			padding: var(--spacing-sm) var(--spacing-md);
		}
	}
</style>

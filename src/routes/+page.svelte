<script lang="ts">
	import { onMount } from 'svelte';
	import {
		Search,
		GitGraph,
		Globe,
		Database,
		Cpu,
		Server,
		Zap,
		FileCode,
		Package,
		BookOpen,
		GitBranch,
		Bug,
		Github,
		Brain,
		Leaf
	} from 'lucide-svelte';

	// Form state
	let heroEmail = $state('');
	let demoEmail = $state('');
	let heroSubmitted = $state(false);
	let demoSubmitted = $state(false);

	// Features data
	const features = [
		{
			icon: Search,
			title: 'Semantic Search',
			desc: 'Intent-aware code discovery with repo_search. Find code by meaning, not just keywords.'
		},
		{
			icon: GitGraph,
			title: 'Symbol Graph',
			desc: 'Navigate callers, callees, and definitions with symbol_graph. Understand code relationships instantly.'
		},
		{
			icon: Globe,
			title: 'Cross-Repo Search',
			desc: 'Trace API boundaries across repositories. Follow the data flow from frontend to backend.'
		},
		{
			icon: Database,
			title: 'Persistent Memory',
			desc: 'Store and recall context with memory_store and memory_find. Your AI remembers everything.'
		},
		{
			icon: Zap,
			title: 'IDE Agnostic',
			desc: 'Works with Claude, Cursor, Windsurf, Copilot, Kiro, and any MCP-compatible client.'
		},
		{
			icon: Server,
			title: 'Self-Hosted',
			desc: 'Your code never leaves your network. Run Singular mode for complete data sovereignty.'
		}
	];

	// Graph data
	const sources = [
		{ icon: FileCode, label: 'Code Files' },
		{ icon: Package, label: 'Dependencies' },
		{ icon: BookOpen, label: 'Documentation' },
		{ icon: GitBranch, label: 'Git History' },
		{ icon: Bug, label: 'Issues' }
	];

	const processors = [
		{ icon: Database, label: 'Qdrant Index' },
		{ icon: Cpu, label: 'Embeddings' },
		{ icon: GitGraph, label: 'Symbol Graph' },
		{ icon: Database, label: 'Memory Store' }
	];

	const tools = [
		{ label: 'repo_search' },
		{ label: 'symbol_graph' },
		{ label: 'context_answer' },
		{ label: 'memory_find' }
	];

	let graphCanvas: SVGSVGElement;
	let graphContainer: HTMLDivElement;

	function drawConnections() {
		if (!graphCanvas || !graphContainer) return;

		const rect = graphContainer.getBoundingClientRect();
		graphCanvas.setAttribute('width', String(rect.width));
		graphCanvas.setAttribute('height', String(rect.height));
		graphCanvas.innerHTML = '';

		const sourceNodes = graphContainer.querySelectorAll('.sources .graph-node');
		const processorNodes = graphContainer.querySelectorAll('.processors .graph-node');
		const toolNodes = graphContainer.querySelectorAll('.tools .graph-node');

		const getCenter = (el: Element) => {
			const r = el.getBoundingClientRect();
			return {
				x: r.left + r.width / 2 - rect.left,
				y: r.top + r.height / 2 - rect.top
			};
		};

		const drawLine = (
			from: { x: number; y: number },
			to: { x: number; y: number },
			color: string,
			delay: number
		) => {
			const dx = to.x - from.x;
			const cp1x = from.x + dx * 0.4;
			const cp2x = from.x + dx * 0.6;
			const d = `M${from.x},${from.y} C${cp1x},${from.y} ${cp2x},${to.y} ${to.x},${to.y}`;

			// Invisible hit area (wider, for easier hover)
			const hitArea = document.createElementNS('http://www.w3.org/2000/svg', 'path');
			hitArea.setAttribute('d', d);
			hitArea.setAttribute('fill', 'none');
			hitArea.setAttribute('stroke', 'transparent');
			hitArea.setAttribute('stroke-width', '20');
			hitArea.classList.add('graph-line-hit');

			// Visible line
			const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
			path.setAttribute('d', d);
			path.setAttribute('fill', 'none');
			path.setAttribute('stroke', color);
			path.setAttribute('stroke-width', '1.5');
			path.setAttribute('stroke-opacity', '0.3');
			path.classList.add('graph-line');
			path.dataset.color = color;
			const length = path.getTotalLength();
			path.style.strokeDasharray = String(length);
			path.style.strokeDashoffset = String(length);
			path.style.animation = `drawLine 1.5s ease forwards ${delay}s`;

			// Link hit area to visible path for hover effect
			hitArea.addEventListener('mouseenter', () => path.classList.add('hovered'));
			hitArea.addEventListener('mouseleave', () => path.classList.remove('hovered'));

			graphCanvas.appendChild(path);
			graphCanvas.appendChild(hitArea);
		};

		sourceNodes.forEach((s, i) => {
			processorNodes.forEach((p, j) => {
				drawLine(getCenter(s), getCenter(p), '#2d67ff', i * 0.1 + j * 0.05);
			});
		});

		processorNodes.forEach((p, i) => {
			toolNodes.forEach((t, j) => {
				drawLine(getCenter(p), getCenter(t), '#00b894', 0.5 + i * 0.1 + j * 0.05);
			});
		});
	}

	onMount(() => {
		// Draw connections after mount
		setTimeout(drawConnections, 100);
		window.addEventListener('resize', drawConnections);

		return () => {
			window.removeEventListener('resize', drawConnections);
		};
	});

	async function handleHeroSubmit(e: SubmitEvent) {
		e.preventDefault();
		try {
			const response = await fetch('https://formspree.io/f/xojjvnkd', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ email: heroEmail, subject: 'Early Access Request' })
			});
			if (response.ok) {
				heroSubmitted = true;
			}
		} catch (error) {
			console.error('Form submission error:', error);
		}
	}

	async function handleDemoSubmit(e: SubmitEvent) {
		e.preventDefault();
		try {
			const response = await fetch('https://formspree.io/f/xojjvnkd', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ email: demoEmail, subject: 'Request Invite' })
			});
			if (response.ok) {
				demoSubmitted = true;
			}
		} catch (error) {
			console.error('Form submission error:', error);
		}
	}
</script>

<svelte:head>
	<title>Context Engine - Your Codebase, Instantly Understood</title>
	<meta
		name="description"
		content="AI code context platform. Semantic search, symbol graphs, persistent memory across every IDE."
	/>
</svelte:head>

<!-- Hero Section -->
<section class="hero">
	<div class="hero-content">
		<span class="kicker">AI Code Context Platform</span>
		<h1 class="hero-title">
			Your Codebase,<br /><span class="gradient-text">Instantly Understood</span>
		</h1>
		<p class="hero-subtitle">
			Semantic search, symbol graphs, and persistent memory across every IDE. Stop explaining your
			codebase to AI. Start coding.
		</p>

		{#if heroSubmitted}
			<div class="hero-form" style="color: var(--accent); font-size: 16px;">
				Thanks! We'll be in touch at {heroEmail}
			</div>
		{:else}
			<form class="hero-form" onsubmit={handleHeroSubmit}>
				<input
					type="email"
					class="input"
					placeholder="you@company.com"
					required
					bind:value={heroEmail}
				/>
				<button type="submit" class="btn btn-primary btn-lg">Request Early Access</button>
			</form>
		{/if}

		<div class="hero-actions">
			<a
				href="https://github.com/Context-Engine-AI/Context-Engine"
				class="btn btn-secondary"
				target="_blank"
			>
				<Github size={18} />
				View on GitHub
			</a>
			<a
				href="https://www.npmjs.com/package/@context-engine-bridge/context-engine-mcp-bridge"
				class="btn btn-ghost"
				target="_blank"
			>
				<svg viewBox="0 0 24 24" fill="currentColor" width="18" height="18">
					<path
						d="M0 7.334v8h6.666v1.332H12v-1.332h12v-8H0zm6.666 6.664H5.334v-4H3.999v4H1.335V8.667h5.331v5.331zm4 0v1.336H8.001V8.667h5.334v5.332h-2.669v-.001zm12.001 0h-1.33v-4h-1.336v4h-1.335v-4h-1.33v4h-2.671V8.667h8.002v5.331zM10.665 10H12v2.667h-1.335V10z"
					/>
				</svg>
				NPM Bridge
			</a>
		</div>

		<div class="hero-stats">
			<div class="stat">
				<span class="stat-value">&lt;100ms</span>
				<span class="stat-label">Search Latency</span>
			</div>
			<div class="stat">
				<span class="stat-value">16</span>
				<span class="stat-label">IDE Integrations</span>
			</div>
			<div class="stat">
				<span class="stat-value">32</span>
				<span class="stat-label">Languages</span>
			</div>
		</div>
	</div>

	<div class="hero-visual">
		<div class="code-window">
			<div class="code-header">
				<span class="dot red"></span>
				<span class="dot yellow"></span>
				<span class="dot green"></span>
				<span class="code-title">mcp_tools.py</span>
			</div>
			<pre class="code-content"><span class="c-comment"># Find authentication flow</span>
result = <span class="c-fn">repo_search</span>(
    query=<span class="c-str">"user authentication"</span>,
    limit=<span class="c-num">5</span>,
    compact=<span class="c-key">True</span>
)
<span class="c-comment"># 4,456 sources -> 5 relevant in 47ms</span></pre>
		</div>

		<div class="hero-visual-features">
			<div class="visual-feature">
				<div class="visual-feature-icon">
					<Brain size={20} />
				</div>
				<div class="visual-feature-text">
					<span class="visual-feature-title">Self-Learning</span>
					<span class="visual-feature-desc">Gets smarter as you search</span>
				</div>
			</div>
			<div class="visual-feature">
				<div class="visual-feature-icon green">
					<Leaf size={20} />
				</div>
				<div class="visual-feature-text">
					<span class="visual-feature-title">Green Energy</span>
					<span class="visual-feature-desc">80% less compute per query</span>
				</div>
			</div>
		</div>
	</div>
</section>

<!-- Graph Section -->
<section class="graph-section">
	<h2 class="section-title">How It <span class="gradient-text">Works</span></h2>
	<p class="section-subtitle">
		From raw code to curated context in milliseconds. Every query flows through our semantic
		understanding layer.
	</p>

	<div class="graph-container">
		<div class="graph-labels">
			<span class="graph-label">Raw Sources</span>
			<span class="graph-label">Semantic Processing</span>
			<span class="graph-label">MCP Tools</span>
		</div>

		<div class="graph-content" bind:this={graphContainer}>
			<div class="graph-column sources">
				{#each sources as source}
					{@const Icon = source.icon}
					<div class="graph-node source">
						<span class="node-icon">
							<Icon size={18} />
						</span>
						{source.label}
					</div>
				{/each}
			</div>

			<div class="graph-column processors">
				{#each processors as processor}
					{@const Icon = processor.icon}
					<div class="graph-node process">
						<span class="node-icon">
							<Icon size={18} />
						</span>
						{processor.label}
					</div>
				{/each}
			</div>

			<div class="graph-column tools">
				{#each tools as tool}
					<div class="graph-node tool">
						<span class="node-icon">
							<Zap size={18} />
						</span>
						{tool.label}
					</div>
				{/each}
			</div>

			<svg class="graph-canvas" bind:this={graphCanvas}></svg>
		</div>

		<div class="graph-footer">
			<div class="graph-stats">4,456 sources → 682 relevant spans</div>
			<div class="graph-progress">
				<div class="graph-progress-bar"></div>
			</div>
		</div>
	</div>
</section>

<!-- Features Section -->
<section class="features-section">
	<h2 class="section-title">Built for <span class="gradient-text">Developers</span></h2>
	<p class="section-subtitle">
		Everything you need to give your AI deep understanding of your codebase.
	</p>

	<div class="features-grid">
		{#each features as feature}
			{@const Icon = feature.icon}
			<div class="feature-card">
				<div class="feature-icon">
					<Icon size={22} />
				</div>
				<h3 class="feature-title">{feature.title}</h3>
				<p class="feature-desc">{feature.desc}</p>
			</div>
		{/each}
	</div>
</section>

<!-- Demo Section -->
<section class="demo-section" id="demo">
	<h2 class="section-title">Get <span class="gradient-text">Early Access</span></h2>
	<p class="section-subtitle">Join the beta and give your AI the context it deserves.</p>

	{#if demoSubmitted}
		<div class="demo-form" style="color: var(--accent); font-size: 16px; justify-content: center;">
			Thanks! We'll be in touch at {demoEmail}
		</div>
	{:else}
		<form class="demo-form" onsubmit={handleDemoSubmit}>
			<input
				type="email"
				class="input"
				placeholder="you@company.com"
				required
				bind:value={demoEmail}
			/>
			<button type="submit" class="btn btn-primary btn-lg">Request Invite</button>
		</form>
	{/if}
</section>

<!-- Footer -->
<footer class="footer">
	<div class="footer-content">
		<div class="footer-links">
			<a
				href="https://github.com/Context-Engine-AI/Context-Engine"
				class="footer-link"
				target="_blank">GitHub</a
			>
			<a
				href="https://www.npmjs.com/package/@context-engine-bridge/context-engine-mcp-bridge"
				class="footer-link"
				target="_blank">NPM</a
			>
			<a
				href="https://marketplace.visualstudio.com/items?itemName=context-engine.context-engine-uploader"
				class="footer-link"
				target="_blank">VS Code Extension</a
			>
			<a href="mailto:support@context-engine.ai" class="footer-link">support@context-engine.ai</a>
		</div>
		<div class="footer-copy">© {new Date().getFullYear()} Context Engine. All rights reserved.</div>
	</div>
</footer>

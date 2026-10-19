<script lang="ts">
	import { onMount } from 'svelte';
	import { base } from '$app/paths';
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
		Play,
		X,
		Volume2,
		VolumeX
	} from 'lucide-svelte';

	// Video lightbox state
	let videoOpen = $state(false);
	let videoMuted = $state(true);
	let videoEl: HTMLVideoElement = $state(null!);

	function openVideo() {
		videoOpen = true;
		document.body.style.overflow = 'hidden';
	}

	function closeVideo() {
		videoOpen = false;
		document.body.style.overflow = '';
		if (videoEl) {
			videoEl.pause();
			videoEl.currentTime = 0;
		}
	}

	function toggleMute() {
		videoMuted = !videoMuted;
		if (videoEl) videoEl.muted = videoMuted;
	}

	function handleLightboxKeydown(e: KeyboardEvent) {
		if (e.key === 'Escape') closeVideo();
	}

	function handleOverlayClick(e: MouseEvent) {
		if ((e.target as HTMLElement).classList.contains('video-lightbox')) {
			closeVideo();
		}
	}

	const SIGNUP_URL = 'https://dev.context-engine.ai/auth/signup';
	const plans = [
		{ name: 'Starter', monthly: 5, annual: 48, members: 1, repos: 2, calls: '500' },
		{ name: 'Individual', monthly: 10, annual: 96, members: 1, repos: 5, calls: '3,500' },
		{ name: 'Team', monthly: 29, annual: 278, members: 5, repos: 10, calls: '50,000' },
		{ name: 'Business', monthly: 79, annual: 758, members: 15, repos: 30, calls: '150,000' },
		{ name: 'Organization', monthly: 149, annual: 1430, members: 25, repos: 50, calls: '250,000' }
	];

	// Features data
	const features = [
		{
			icon: Search,
			title: 'Semantic Search',
			desc: 'Find the implementation behind an idea, not just a matching keyword. Give your agent relevant source evidence instead of an entire repository.'
		},
		{
			icon: GitGraph,
			title: 'Symbol Graph',
			desc: 'Navigate definitions, callers, callees and dependencies. Understand what a change touches before your agent writes it.'
		},
		{
			icon: Globe,
			title: 'Cross-Repo Intelligence',
			desc: 'Connect repositories across your business. Trace APIs, shared types and dependencies from the frontend through services and libraries.'
		},
		{
			icon: Database,
			title: 'Organization Knowledge',
			desc: 'Build a shared ontology: a map of services, concepts, relationships and decisions grounded in your code and documentation. Keep useful team knowledge in persistent memory.'
		},
		{
			icon: Zap,
			title: 'Any Agent. Any IDE. Any Model.',
			desc: 'One MCP connection for Claude, Codex, Cursor, Copilot and other compatible clients. Choose your model through your agent or IDE; keep your context portable.'
		},
		{
			icon: Server,
			title: 'Your Knowledge, Your Control',
			desc: 'Workspace permissions keep organization knowledge scoped to the right people. Choose hosted access or self-host on your own infrastructure.'
		}
	];

	// Graph data
	const sources = [
		{ icon: FileCode, label: 'Code Files' },
		{ icon: Package, label: 'Dependencies' },
		{ icon: BookOpen, label: 'Documentation' },
		{ icon: GitBranch, label: 'Git History' },
		{ icon: Bug, label: 'Team Knowledge' }
	];

	const processors = [
		{ icon: Database, label: 'Helion Index' },
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
</script>

<svelte:head>
	<title>Context Engine — Codebase Intelligence for Any Agent, IDE or Model</title>
	<meta
		name="description"
		content="One MCP server for codebase intelligence across agents, IDEs and any model. Connect repositories and organization knowledge across your business. Plans from $5/month."
	/>
</svelte:head>

<!-- Hero Section -->
<section class="hero">
	<div class="hero-content">
		<div class="hero-brand">
			<img src="{base}/logo-white.svg" alt="Context Engine" class="hero-logo hero-logo-dark" />
			<img src="{base}/logo.svg" alt="Context Engine" class="hero-logo hero-logo-light" />
			<a
				href="https://www.nvidia.com/en-us/startups/"
				target="_blank"
				rel="noopener noreferrer"
				class="nvidia-badge-link"
			>
				<img
					src="{base}/nvidia-inception-badge.png"
					alt="NVIDIA Inception Program"
					class="nvidia-badge"
				/>
			</a>
		</div>
		<h1 class="hero-title">
			Codebase intelligence.<br /><span class="gradient-text">Beyond one repository.</span>
		</h1>
		<p class="hero-subtitle">
			One MCP server for codebase intelligence across agents, IDEs, and any model. Connect your
			repositories and organization knowledge so every workflow starts with the right context. Bring
			your own factory; keep your intelligence connected.
		</p>

		<div class="hero-form">
			<a href={SIGNUP_URL} class="btn btn-primary btn-lg">Get started</a>
			<a href="#pricing" class="btn btn-secondary btn-lg">Plans from $5/month</a>
		</div>

		<div class="hero-actions">
			<button class="btn btn-demo" onclick={openVideo}>
				<span class="btn-demo-icon"><Play size={18} /></span>
				Watch Demo
			</button>
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
				<span class="stat-value">MCP</span>
				<span class="stat-label">One open protocol</span>
			</div>
			<div class="stat">
				<span class="stat-value">Any model</span>
				<span class="stat-label">Through your agent or IDE</span>
			</div>
			<div class="stat">
				<span class="stat-value">Cross-repo</span>
				<span class="stat-label">Organization-wide context</span>
			</div>
		</div>
	</div>

	<div class="hero-visual">
		<div class="code-window">
			<div class="code-header">
				<span class="dot red"></span>
				<span class="dot yellow"></span>
				<span class="dot green"></span>
				<span class="code-title">MCP · cross_repo_search</span>
			</div>
			<pre class="code-content"><span class="c-comment">// Your agent asks across the business</span
				>
<span class="c-fn">cross_repo_search</span>({'{'}
  <span class="c-key">"query"</span>: <span class="c-str">"Who consumes CheckoutCreated?"</span>,
  <span class="c-key">"target_repos"</span>: [<span class="c-str">"storefront"</span>, <span
					class="c-str">"billing"</span
				>, <span class="c-str">"shared"</span>],
  <span class="c-key">"trace_boundary"</span>: <span class="c-key">true</span>
{'}'})

<span class="c-comment">// Illustrative source-backed context</span>
<span class="c-str">shared/events.ts</span>       <span class="c-comment">→ event contract</span>
<span class="c-str">storefront/checkout.ts</span> <span class="c-comment">→ publisher</span>
<span class="c-str">billing/invoices.ts</span>    <span class="c-comment">→ consumer</span></pre>
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
					<GitGraph size={20} />
				</div>
				<div class="visual-feature-text">
					<span class="visual-feature-title">Shared Knowledge</span>
					<span class="visual-feature-desc">Context that travels across tools</span>
				</div>
			</div>
		</div>
	</div>
</section>

<!-- Graph Section -->
<section class="features-section" aria-labelledby="evaluation-title">
	<p class="open-source-label">
		Measured with real agent tasks · internal evaluation · October 5, 2026
	</p>
	<h2 class="section-title" id="evaluation-title">
		Higher accuracy. <span class="gradient-text">Fewer calls. Fewer tokens.</span>
	</h2>
	<div class="evaluation-results">
		<div><strong>23% higher</strong><span>file-and-symbol accuracy</span></div>
		<div><strong>14% fewer</strong><span>tokens per task, on average</span></div>
		<div><strong>14% fewer</strong><span>tool calls per task, on average</span></div>
	</div>
	<details class="evaluation-method">
		<summary>About the evaluation</summary>
		<p>
			Claude Sonnet with and without Context Engine: 80 paired comparisons across 40 code-navigation
			tasks in two repositories. Accuracy measures correct-file and correct-symbol recall, improving
			from 52.2% to 64.3%. Average tokens decreased from 1.01M to 0.88M and tool calls from 25.5 to
			21.8.
		</p>
		<p>
			Public retrieval-benchmark reference:
			<a href="https://github.com/noodlbox/tokens-to-gold" target="_blank" rel="noopener noreferrer"
				>TokensToGold methodology and scoring code ↗</a
			>. The Context Engine figures above come from our separate agent A/B evaluation.
		</p>
	</details>
</section>

<section class="graph-section">
	<h2 class="section-title">How It <span class="gradient-text">Works</span></h2>
	<p class="section-subtitle">
		Code, documentation, relationships and memory become grounded context your AI tools can use.
		Connect through MCP without tying your business knowledge to a single model.
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
			<div class="graph-stats">Connected sources → relevant code and organization context</div>
			<div class="graph-progress">
				<div class="graph-progress-bar"></div>
			</div>
		</div>
	</div>
</section>

<!-- Features Section -->
<section class="features-section" id="capabilities">
	<h2 class="section-title">
		Intelligence for your <span class="gradient-text">whole codebase</span>
	</h2>
	<p class="section-subtitle">
		From one repository to knowledge across your business. Keep context portable as your agents,
		IDEs and models evolve.
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

<section
	class="features-section integrations"
	id="integrations"
	aria-labelledby="integrations-title"
>
	<h2 class="section-title" id="integrations-title">
		Your workflow. <span class="gradient-text">One MCP connection.</span>
	</h2>
	<p class="section-subtitle">
		Bring Context Engine to your agents and editors. Your organization’s context stays connected
		when your tools change.
	</p>
	<div class="integration-links">
		<a
			href="https://marketplace.visualstudio.com/items?itemName=context-engine.context-engine-uploader"
			target="_blank"
			rel="noopener noreferrer">VS Code Marketplace ↗</a
		>
		<a
			href="https://open-vsx.org/extension/context-engine/context-engine-uploader"
			target="_blank"
			rel="noopener noreferrer">Open VSX ↗</a
		>
		<a
			href="https://www.npmjs.com/package/@context-engine-bridge/context-engine-mcp-bridge"
			target="_blank"
			rel="noopener noreferrer">MCP bridge on npm ↗</a
		>
		<a href="https://docs.context-engine.ai" target="_blank" rel="noopener noreferrer"
			>MCP setup &amp; documentation ↗</a
		>
	</div>
	<div class="factory-note">
		<p class="open-source-label">BYOF — Bring Your Own Factory</p>
		<h3>Your agents run the factory. Context Engine supplies shared intelligence.</h3>
		<p>
			Give planning, coding, review and verification agents a common view of your repositories,
			dependencies and team knowledge through MCP-compatible runtimes. Choose your orchestration
			layer instead of rebuilding your context for every agent.
		</p>
		<a href="https://github.com/Agent-Field/SWE-AF" target="_blank" rel="noopener noreferrer"
			>Explore SWE-AF, an independent AgentField software factory ↗</a
		>
	</div>
</section>

<section class="features-section" id="pricing" aria-labelledby="pricing-title">
	<h2 class="section-title" id="pricing-title">
		Start with <span class="gradient-text">$5/month</span>
	</h2>
	<p class="section-subtitle">
		Every hosted plan requires a subscription. Choose the workspace size that fits your team.
	</p>
	<div class="features-grid">
		{#each plans as plan}
			<div class="feature-card">
				<h3 class="feature-title">{plan.name}</h3>
				<p class="plan-price"><strong>${plan.monthly}</strong>/month</p>
				<p class="feature-desc">Or ${plan.annual}/year, billed annually.</p>
				<p class="feature-desc">
					{plan.members}
					{plan.members === 1 ? 'member' : 'members'} · {plan.repos} repositories · {plan.calls} tool
					calls/month
				</p>
				<p class="feature-desc">
					Semantic search, symbol graphs, cross-repo intelligence and persistent memory.
				</p>
				<a href={SIGNUP_URL} class="btn btn-primary">Get started</a>
			</div>
		{/each}
	</div>
	<p class="section-subtitle">
		Need a business-wide knowledge platform, custom capacity or an enterprise agreement? <a
			href="{base}/contact">Talk to us</a
		>.
	</p>
</section>

<!-- Video Lightbox -->
{#if videoOpen}
	<!-- svelte-ignore a11y_no_noninteractive_element_interactions -->
	<div
		class="video-lightbox"
		role="dialog"
		aria-modal="true"
		aria-label="Demo video"
		tabindex="-1"
		onkeydown={handleLightboxKeydown}
		onclick={handleOverlayClick}
	>
		<div class="video-lightbox-inner">
			<div class="video-lightbox-controls">
				<button
					class="video-control-btn"
					onclick={toggleMute}
					aria-label={videoMuted ? 'Unmute' : 'Mute'}
				>
					{#if videoMuted}
						<VolumeX size={20} />
					{:else}
						<Volume2 size={20} />
					{/if}
				</button>
				<button class="video-control-btn" onclick={closeVideo} aria-label="Close video">
					<X size={20} />
				</button>
			</div>
			<!-- svelte-ignore a11y_media_has_caption -->
			<video
				bind:this={videoEl}
				autoplay
				muted={videoMuted}
				loop
				playsinline
				class="video-lightbox-player"
			>
				<source src="{base}/context_engine_1.webm" type="video/webm" />
				<source src="{base}/context_engine_1.mp4" type="video/mp4" />
			</video>
		</div>
	</div>
{/if}

<!-- Footer -->
<footer class="footer">
	<div class="open-source-foundations">
		<p class="open-source-label">Open-source building blocks from the Context Engine team</p>
		<div class="open-source-links">
			<a
				href="https://github.com/Context-Engine-AI/helion-db"
				target="_blank"
				rel="noopener noreferrer"
			>
				<strong>Helion DB ↗</strong>
				<span
					>Rust graph + vector database, forked from HelixDB, with optional LSM storage on
					S3-compatible object stores. GPL-3.0.</span
				>
			</a>
			<a
				href="https://github.com/Context-Engine-AI/Context-Engine-Reviewer"
				target="_blank"
				rel="noopener noreferrer"
			>
				<strong>Context Engine Reviewer ↗</strong>
				<span
					>A standalone, multi-provider AI pull-request reviewer with optional Context Engine
					intelligence. AGPL-3.0.</span
				>
			</a>
		</div>
	</div>
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
			<a
				href="https://open-vsx.org/extension/context-engine/context-engine-uploader"
				class="footer-link"
				target="_blank"
				rel="noopener noreferrer">Open VSX</a
			>
			<a href="mailto:support@context-engine.ai" class="footer-link">support@context-engine.ai</a>
		</div>
		<div class="footer-copy">© {new Date().getFullYear()} Context Engine. All rights reserved.</div>
	</div>
	<div class="footer-legal">
		© 2025 NVIDIA, the NVIDIA logo, and NVIDIA Inception Program are trademarks and/or registered
		trademarks of NVIDIA Corporation in the U.S. and other countries.
	</div>
</footer>

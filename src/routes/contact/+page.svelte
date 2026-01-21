<script lang="ts">
	import { fade } from 'svelte/transition';
	import {
		Mail,
		Github,
		MapPin,
		MessageSquare,
		Send,
		CheckCircle,
		AlertCircle
	} from 'lucide-svelte';
	// Import Discord icon from a custom SVG since it's not in lucide
	import { base } from '$app/paths';

	let formState: 'idle' | 'submitting' | 'success' | 'error' = $state('idle');
	let formData = $state({
		name: '',
		email: '',
		subject: '',
		message: ''
	});

	async function handleSubmit(event: SubmitEvent) {
		event.preventDefault();
		formState = 'submitting';

		try {
			const response = await fetch('https://formspree.io/f/xojjvnkd', {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json'
				},
				body: JSON.stringify(formData)
			});

			if (response.ok) {
				formState = 'success';
				formData = { name: '', email: '', subject: '', message: '' };
			} else {
				formState = 'error';
			}
		} catch (error) {
			formState = 'error';
		}
	}
</script>

<svelte:head>
	<title>Contact | Context Engine</title>
	<meta
		name="description"
		content="Get in touch with the Context Engine team - MCP Retrieval Stack"
	/>
</svelte:head>

<div class="contact-container" in:fade={{ duration: 300 }}>
	<div class="container">
		<header class="contact-header">
			<h1 class="hero-title">Get In Touch</h1>
			<p class="hero-subtitle">
				Let's discuss your MCP integration <span class="divider">//</span> Questions about Context Engine?
			</p>
		</header>

		<div class="contact-content">
			<div class="contact-info glass">
				<h2>Let's Connect</h2>
				<p>
					Whether you're looking to implement Context Engine for your AI coding assistant, want to
					contribute to the project, or have questions about the MCP retrieval stack, we'd love to
					hear from you.
				</p>

				<div class="info-items">
					<!-- <div class="info-item glass-subtle">
						<span class="icon">
							<Mail size={24} />
						</span>
						<div>
							<h3>Email</h3>
							<a href="mailto:contact@context-engine.ai">contact@context-engine.ai</a>
						</div>
					</div> -->

					<div class="info-item glass-subtle">
						<span class="icon">
							<Github size={24} />
						</span>
						<div>
							<h3>GitHub</h3>
							<a
								href="https://github.com/m1rl0k/Context-Engine"
								target="_blank"
								rel="noopener noreferrer"
							>
								github.com/m1rl0k/Context-Engine
							</a>
						</div>
					</div>

					<div class="info-item glass-subtle">
						<span class="icon discord-icon">
							<svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
								<path
									d="M20.317 4.37a19.791 19.791 0 0 0-4.885-1.515a.074.074 0 0 0-.079.037c-.210.375-.444.864-.608 1.25a18.27 18.27 0 0 0-5.487 0a12.64 12.64 0 0 0-.617-1.25a.077.077 0 0 0-.079-.037A19.736 19.736 0 0 0 3.677 4.37a.07.07 0 0 0-.032.027C.533 9.046-.32 13.58.099 18.057a.082.082 0 0 0 .031.057a19.9 19.9 0 0 0 5.993 3.03a.078.078 0 0 0 .084-.028a14.09 14.09 0 0 0 1.226-1.994a.076.076 0 0 0-.041-.106a13.107 13.107 0 0 1-1.872-.892a.077.077 0 0 1-.008-.128a10.2 10.2 0 0 0 .372-.292a.074.074 0 0 1 .077-.010c3.928 1.793 8.18 1.793 12.062 0a.074.074 0 0 1 .078.01c.120.098.246.198.373.292a.077.077 0 0 1-.006.127a12.299 12.299 0 0 1-1.873.892a.077.077 0 0 0-.041.107c.36.698.772 1.362 1.225 1.993a.076.076 0 0 0 .084.028a19.839 19.839 0 0 0 6.002-3.03a.077.077 0 0 0 .032-.054c.5-5.177-.838-9.674-3.549-13.66a.061.061 0 0 0-.031-.03zM8.02 15.33c-1.183 0-2.157-1.085-2.157-2.419c0-1.333.956-2.419 2.157-2.419c1.210 0 2.176 1.096 2.157 2.42c0 1.333-.956 2.418-2.157 2.418zm7.975 0c-1.183 0-2.157-1.085-2.157-2.419c0-1.333.955-2.419 2.157-2.419c1.210 0 2.176 1.096 2.157 2.42c0 1.333-.946 2.418-2.157 2.418z"
								/>
							</svg>
						</span>
						<div>
							<h3>Discord Community</h3>
							<a
								href="https://discord.com/channels/1452237497315622924"
								target="_blank"
								rel="noopener noreferrer"
							>
								Join our Discord server
							</a>
						</div>
					</div>

					<div class="info-item glass-subtle">
						<span class="icon">
							<MapPin size={24} />
						</span>
						<div>
							<h3>Open Source</h3>
							<p>Built for the global AI developer community</p>
						</div>
					</div>
				</div>
			</div>

			<form class="contact-form glass" onsubmit={handleSubmit}>
				<div class="form-header">
					<h2>Send a Message</h2>
				</div>

				<div class="form-group">
					<label for="name">Name</label>
					<input
						type="text"
						id="name"
						bind:value={formData.name}
						required
						disabled={formState === 'submitting'}
						placeholder="Your name"
					/>
				</div>

				<div class="form-group">
					<label for="email">Email</label>
					<input
						type="email"
						id="email"
						bind:value={formData.email}
						required
						disabled={formState === 'submitting'}
						placeholder="your.email@example.com"
					/>
				</div>

				<div class="form-group">
					<label for="subject">Subject</label>
					<input
						type="text"
						id="subject"
						bind:value={formData.subject}
						required
						disabled={formState === 'submitting'}
						placeholder="What's this about?"
					/>
				</div>

				<div class="form-group">
					<label for="message">Message</label>
					<textarea
						id="message"
						bind:value={formData.message}
						rows="6"
						required
						disabled={formState === 'submitting'}
						placeholder="Tell us about your use case, questions, or how you'd like to contribute..."
					></textarea>
				</div>

				{#if formState === 'success'}
					<div class="alert alert-success" in:fade>
						<CheckCircle size={20} />
						<span>Message sent successfully! We'll get back to you within 24-48 hours.</span>
					</div>
				{/if}

				{#if formState === 'error'}
					<div class="alert alert-error" in:fade>
						<AlertCircle size={20} />
						<span>Something went wrong. Please try again or email us directly.</span>
					</div>
				{/if}

				<button
					type="submit"
					class="submit-button glass-button"
					disabled={formState === 'submitting'}
				>
					{#if formState === 'submitting'}
						<span class="loading-spinner"></span>
						<span>Sending...</span>
					{:else}
						<Send size={20} />
						<span>Send Message</span>
					{/if}
				</button>
			</form>
		</div>

		<div class="back-link">
			<a href="{base}/" class="btn-secondary"> ← Back to Home </a>
		</div>
	</div>
</div>

<style>
	.contact-container {
		min-height: 100vh;
		padding: var(--spacing-xl) 0;
	}

	.container {
		max-width: 1200px;
		margin: 0 auto;
		padding: 0 var(--spacing-md);
	}

	.contact-header {
		text-align: center;
		margin-bottom: var(--spacing-xl);
		padding-bottom: var(--spacing-lg);
		position: relative;
	}

	.contact-header::after {
		content: '';
		position: absolute;
		bottom: 0;
		left: 50%;
		transform: translateX(-50%);
		width: 100px;
		height: 2px;
		background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.5), transparent);
	}

	.hero-title {
		font-size: clamp(2.5rem, 5vw, 4rem);
		font-weight: 800;
		background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
		background-clip: text;
		-webkit-background-clip: text;
		-webkit-text-fill-color: transparent;
		margin-bottom: var(--spacing-md);
		letter-spacing: -0.02em;
	}

	.hero-subtitle {
		font-size: 1.25rem;
		color: var(--text-secondary);
		max-width: 600px;
		margin: 0 auto;
		line-height: 1.6;
	}

	.hero-subtitle .divider {
		color: rgba(167, 139, 250, 0.9);
		margin: 0 var(--spacing-xs);
		font-weight: 700;
	}

	.contact-content {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: var(--spacing-xl);
		align-items: start;
		margin-bottom: var(--spacing-xl);
	}

	.glass {
		background: var(--glass-primary);
		backdrop-filter: blur(20px) saturate(1.8);
		border: 1px solid var(--glass-border);
		border-radius: var(--radius-lg);
		padding: var(--spacing-lg);
	}

	.glass-subtle {
		background: var(--glass-secondary);
		backdrop-filter: blur(10px) saturate(1.2);
		border: 1px solid rgba(255, 255, 255, 0.1);
		border-radius: var(--radius-md);
	}

	.glass-button {
		background: linear-gradient(135deg, rgba(99, 102, 241, 0.9) 0%, rgba(139, 92, 246, 0.9) 100%);
		border: 1px solid rgba(99, 102, 241, 1);
		color: white;
		transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
		font-weight: 700;
		box-shadow:
			0 8px 32px rgba(99, 102, 241, 0.5),
			0 4px 16px rgba(139, 92, 246, 0.3),
			inset 0 1px 0 rgba(255, 255, 255, 0.3);
	}

	.glass-button:hover:not(:disabled) {
		background: linear-gradient(135deg, rgba(99, 102, 241, 1) 0%, rgba(139, 92, 246, 1) 100%);
		border-color: rgba(167, 139, 250, 1);
		transform: translateY(-6px) scale(1.02);
		box-shadow:
			0 20px 60px rgba(99, 102, 241, 0.8),
			0 12px 32px rgba(139, 92, 246, 0.6),
			0 6px 16px rgba(167, 139, 250, 0.4),
			inset 0 1px 0 rgba(255, 255, 255, 0.4);
	}

	.contact-info h2 {
		font-size: 2rem;
		margin-bottom: var(--spacing-md);
		color: var(--text-primary);
		font-weight: 700;
	}

	.contact-info > p {
		line-height: 1.7;
		margin-bottom: var(--spacing-lg);
		color: var(--text-secondary);
		font-size: 1.1rem;
	}

	.info-items {
		display: flex;
		flex-direction: column;
		gap: var(--spacing-md);
	}

	.info-item {
		display: flex;
		align-items: center;
		gap: var(--spacing-md);
		padding: var(--spacing-md);
		transition: all 0.3s ease;
	}

	.info-item:hover {
		transform: translateX(8px);
		border-color: rgba(139, 92, 246, 0.5);
	}

	/* Icon hover effects */
	.info-item:nth-child(1):hover .icon {
		box-shadow: 0 8px 32px rgba(59, 130, 246, 0.5);
		transform: scale(1.05);
	}

	.info-item:nth-child(2):hover .icon {
		box-shadow: 0 8px 32px rgba(55, 65, 81, 0.6);
		transform: scale(1.05);
	}

	.info-item:nth-child(3):hover .discord-icon {
		box-shadow: 0 8px 32px rgba(88, 101, 242, 0.6) !important;
		transform: scale(1.05);
	}

	.info-item:nth-child(4):hover .icon {
		box-shadow: 0 8px 32px rgba(16, 185, 129, 0.5);
		transform: scale(1.05);
	}

	.icon {
		flex-shrink: 0;
		display: flex;
		align-items: center;
		justify-content: center;
		width: 48px;
		height: 48px;
		border-radius: var(--radius-sm);
		transition: all 0.3s ease;
	}

	/* Email icon - Blue theme */
	.info-item:nth-child(1) .icon {
		color: white;
		background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
		box-shadow: 0 4px 20px rgba(59, 130, 246, 0.3);
	}

	/* GitHub icon - Dark theme */
	.info-item:nth-child(2) .icon {
		color: white;
		background: linear-gradient(135deg, #374151 0%, #1f2937 100%);
		box-shadow: 0 4px 20px rgba(55, 65, 81, 0.4);
	}

	/* Discord icon - Discord theme */
	.discord-icon {
		color: white !important;
		background: linear-gradient(135deg, #5865f2 0%, #4338ca 100%) !important;
		box-shadow: 0 4px 20px rgba(88, 101, 242, 0.4) !important;
	}

	/* Open Source icon - Green theme */
	.info-item:nth-child(4) .icon {
		color: white;
		background: linear-gradient(135deg, #10b981 0%, #059669 100%);
		box-shadow: 0 4px 20px rgba(16, 185, 129, 0.3);
	}

	.info-item h3 {
		font-size: 1.125rem;
		margin-bottom: 0.25rem;
		color: var(--text-primary);
		font-weight: 600;
	}

	.info-item a {
		color: rgba(167, 139, 250, 1);
		text-decoration: none;
		transition: color 0.2s ease;
		font-weight: 500;
	}

	.info-item a:hover {
		color: rgba(196, 181, 253, 1);
	}

	.info-item p {
		margin: 0;
		color: var(--text-muted);
	}

	.contact-form {
		/* Form styling without sticky positioning */
	}

	.discord-icon {
		color: #5865f2; /* Discord brand color */
	}

	.form-header h2 {
		font-size: 1.75rem;
		margin-bottom: var(--spacing-lg);
		color: var(--text-primary);
		font-weight: 700;
	}

	.form-group {
		margin-bottom: var(--spacing-md);
	}

	label {
		display: block;
		margin-bottom: var(--spacing-xs);
		color: var(--text-primary);
		font-weight: 600;
		font-size: 0.95rem;
	}

	input,
	textarea {
		width: 100%;
		padding: var(--spacing-sm);
		background: rgba(255, 255, 255, 0.05);
		border: 1px solid rgba(255, 255, 255, 0.2);
		border-radius: var(--radius-sm);
		color: var(--text-primary);
		font-family: var(--font-primary);
		font-size: 1rem;
		transition: all 0.3s ease;
	}

	input::placeholder,
	textarea::placeholder {
		color: var(--text-muted);
	}

	input:focus,
	textarea:focus {
		outline: none;
		border-color: rgba(139, 92, 246, 0.8);
		background: rgba(255, 255, 255, 0.08);
		box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.2);
	}

	input:disabled,
	textarea:disabled {
		opacity: 0.6;
		cursor: not-allowed;
	}

	textarea {
		resize: vertical;
		min-height: 120px;
		font-family: var(--font-primary);
	}

	.submit-button {
		width: 100%;
		padding: var(--spacing-lg);
		border: none;
		border-radius: var(--radius-md);
		font-weight: 700;
		font-size: 1.125rem;
		cursor: pointer;
		font-family: inherit;
		display: flex;
		align-items: center;
		justify-content: center;
		gap: var(--spacing-xs);
		text-transform: uppercase;
		letter-spacing: 0.5px;
		position: relative;
		overflow: hidden;
	}

	.submit-button::before {
		content: '';
		position: absolute;
		top: 0;
		left: -100%;
		width: 100%;
		height: 100%;
		background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
		transition: left 0.6s ease;
	}

	.submit-button:hover:not(:disabled)::before {
		left: 100%;
	}

	.submit-button:disabled {
		opacity: 0.6;
		cursor: not-allowed;
	}

	.loading-spinner {
		width: 20px;
		height: 20px;
		border: 2px solid rgba(255, 255, 255, 0.3);
		border-top: 2px solid var(--text-primary);
		border-radius: 50%;
		animation: spin 1s linear infinite;
	}

	@keyframes spin {
		0% {
			transform: rotate(0deg);
		}
		100% {
			transform: rotate(360deg);
		}
	}

	.alert {
		padding: var(--spacing-md);
		border-radius: var(--radius-sm);
		margin-bottom: var(--spacing-md);
		font-weight: 500;
		display: flex;
		align-items: center;
		gap: var(--spacing-xs);
	}

	.alert-success {
		background: rgba(34, 197, 94, 0.15);
		border: 1px solid rgba(34, 197, 94, 0.3);
		color: rgba(34, 197, 94, 1);
	}

	.alert-error {
		background: rgba(239, 68, 68, 0.15);
		border: 1px solid rgba(239, 68, 68, 0.3);
		color: rgba(239, 68, 68, 1);
	}

	.back-link {
		text-align: center;
		margin-top: var(--spacing-xl);
	}

	.btn-secondary {
		display: inline-flex;
		align-items: center;
		gap: var(--spacing-xs);
		padding: var(--spacing-sm) var(--spacing-md);
		background: var(--glass-secondary);
		color: var(--text-secondary);
		text-decoration: none;
		border: 1px solid var(--glass-border);
		border-radius: var(--radius-sm);
		font-weight: 500;
		transition: all 0.3s ease;
	}

	.btn-secondary:hover {
		color: var(--text-primary);
		background: var(--glass-primary);
		transform: translateY(-2px);
	}

	@media (max-width: 1024px) {
		.contact-content {
			gap: var(--spacing-lg);
		}
	}

	@media (max-width: 768px) {
		.container {
			padding: 0 var(--spacing-sm);
		}

		.contact-content {
			grid-template-columns: 1fr;
			gap: var(--spacing-lg);
		}

		.hero-title {
			font-size: 2.5rem;
		}

		.hero-subtitle {
			font-size: 1.125rem;
		}

		.glass {
			padding: var(--spacing-md);
		}

		.info-item {
			padding: var(--spacing-sm);
		}
	}

	@media (prefers-reduced-motion: reduce) {
		*,
		*::before,
		*::after {
			animation-duration: 0.01ms !important;
			transition-duration: 0.01ms !important;
		}
	}
</style>

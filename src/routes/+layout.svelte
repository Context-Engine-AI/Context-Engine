<script lang="ts">
	import '../app.scss';
	import { onMount } from 'svelte';
	import { page } from '$app/stores';
	import { base } from '$app/paths';
	import { Sun, Moon, Mail, Github, Layers, Menu, X, Key, LogIn } from 'lucide-svelte';

	let { children } = $props();

	let theme = $state('dark');
	let mobileMenuOpen = $state(false);

	onMount(() => {
		const saved = localStorage.getItem('theme');
		if (saved) {
			theme = saved;
			document.documentElement.setAttribute('data-theme', saved);
		}

		// Global Escape key handler for mobile menu
		const handleEscape = (e: KeyboardEvent) => {
			if (e.key === 'Escape' && mobileMenuOpen) {
				closeMobileMenu();
			}
		};
		document.addEventListener('keydown', handleEscape);

		// Close mobile menu when resizing past breakpoint
		const handleResize = () => {
			if (window.innerWidth > 1024 && mobileMenuOpen) {
				closeMobileMenu();
			}
		};
		window.addEventListener('resize', handleResize);

		return () => {
			document.removeEventListener('keydown', handleEscape);
			window.removeEventListener('resize', handleResize);
		};
	});

	function toggleTheme() {
		theme = theme === 'dark' ? 'light' : 'dark';
		document.documentElement.setAttribute('data-theme', theme);
		localStorage.setItem('theme', theme);
	}

	function toggleMobileMenu() {
		mobileMenuOpen = !mobileMenuOpen;
	}

	function closeMobileMenu() {
		mobileMenuOpen = false;
	}

	// Check if we're on the contact page
	let isContactPage = $derived($page.route.id === '/contact');
</script>

<header class="header">
	<div class="header-left">
		<a href="{base}/" class="logo">
			<span class="logo-icon">
				<Layers size={18} />
			</span>
			Context Engine
		</a>
		<span class="byok-pill" data-tooltip="Bring Your Own LLM Key">
			<Key size={12} />
			<span>BYOK</span>
		</span>
		<nav class="nav">
			<a
				href="https://www.npmjs.com/package/@context-engine-bridge/context-engine-mcp-bridge"
				class="nav-link"
				target="_blank"
			>
				<svg viewBox="0 0 24 24" fill="currentColor" width="16" height="16">
					<path
						d="M0 7.334v8h6.666v1.332H12v-1.332h12v-8H0zm6.666 6.664H5.334v-4H3.999v4H1.335V8.667h5.331v5.331zm4 0v1.336H8.001V8.667h5.334v5.332h-2.669v-.001zm12.001 0h-1.33v-4h-1.336v4h-1.335v-4h-1.33v4h-2.671V8.667h8.002v5.331zM10.665 10H12v2.667h-1.335V10z"
					/>
				</svg>
				NPM Bridge
			</a>
			<a
				href="https://github.com/Context-Engine-AI/Context-Engine"
				class="nav-link"
				target="_blank"
			>
				<Github size={16} />
				GitHub
			</a>
			<a
				href="https://marketplace.visualstudio.com/items?itemName=context-engine.context-engine-uploader"
				class="nav-link"
				target="_blank"
			>
				<svg viewBox="0 0 24 24" fill="currentColor" width="16" height="16">
					<path
						d="M17.583.063a1.5 1.5 0 00-1.032.392 1.5 1.5 0 00-.001 0L7.332 9.057 2.983 5.602a1 1 0 00-1.283.082L.313 7.012A1 1 0 000 7.748v8.504a1 1 0 00.313.736l1.387 1.328a1 1 0 001.283.082l4.349-3.455 9.218 8.602a1.5 1.5 0 001.033.392 1.5 1.5 0 00.417-.063l4.5-1.5a1.5 1.5 0 001-1.415V1.541a1.5 1.5 0 00-1-1.415l-4.5-1.5a1.5 1.5 0 00-.417-.063zM18 6.927v10.146L10.893 12z"
					/>
				</svg>
				VS Code <span style="font-size:11px;opacity:0.7">(OSS)</span>
			</a>
		</nav>
	</div>
	<div class="header-right">
		<a href="mailto:support@context-engine.ai" class="btn btn-ghost desktop-only">
			<Mail size={16} />
			Support
		</a>
		<button class="icon-btn" onclick={toggleTheme} aria-label="Toggle theme">
			{#if theme === 'dark'}
				<span class="sun-icon"><Sun size={18} /></span>
			{:else}
				<span class="moon-icon"><Moon size={18} /></span>
			{/if}
		</button>
		<a href="#demo" class="btn btn-primary desktop-only">Request Demo</a>
		<a href="https://dev.context-engine.ai/login" class="btn btn-login desktop-only">
			<LogIn size={16} />
			Login / Sign Up
		</a>
		<button
			class="icon-btn mobile-menu-btn"
			onclick={toggleMobileMenu}
			aria-label={mobileMenuOpen ? 'Close menu' : 'Open menu'}
			aria-expanded={mobileMenuOpen}
			aria-controls="mobile-menu"
		>
			{#if mobileMenuOpen}
				<X size={24} />
			{:else}
				<Menu size={24} />
			{/if}
		</button>
	</div>
</header>

<!-- Mobile Menu Overlay -->
{#if mobileMenuOpen}
	<div class="mobile-menu-overlay" onclick={closeMobileMenu} role="none"></div>
	<nav class="mobile-menu" id="mobile-menu">
		<a
			href="https://www.npmjs.com/package/@context-engine-bridge/context-engine-mcp-bridge"
			class="mobile-nav-link"
			target="_blank"
			rel="noopener noreferrer"
			onclick={closeMobileMenu}
		>
			<svg viewBox="0 0 24 24" fill="currentColor" width="20" height="20">
				<path
					d="M0 7.334v8h6.666v1.332H12v-1.332h12v-8H0zm6.666 6.664H5.334v-4H3.999v4H1.335V8.667h5.331v5.331zm4 0v1.336H8.001V8.667h5.334v5.332h-2.669v-.001zm12.001 0h-1.33v-4h-1.336v4h-1.335v-4h-1.33v4h-2.671V8.667h8.002v5.331zM10.665 10H12v2.667h-1.335V10z"
				/>
			</svg>
			NPM Bridge
		</a>
		<a
			href="https://github.com/Context-Engine-AI/Context-Engine"
			class="mobile-nav-link"
			target="_blank"
			rel="noopener noreferrer"
			onclick={closeMobileMenu}
		>
			<Github size={20} />
			GitHub
		</a>
		<a
			href="https://marketplace.visualstudio.com/items?itemName=context-engine.context-engine-uploader"
			class="mobile-nav-link"
			target="_blank"
			rel="noopener noreferrer"
			onclick={closeMobileMenu}
		>
			<svg viewBox="0 0 24 24" fill="currentColor" width="20" height="20">
				<path
					d="M17.583.063a1.5 1.5 0 00-1.032.392 1.5 1.5 0 00-.001 0L7.332 9.057 2.983 5.602a1 1 0 00-1.283.082L.313 7.012A1 1 0 000 7.748v8.504a1 1 0 00.313.736l1.387 1.328a1 1 0 001.283.082l4.349-3.455 9.218 8.602a1.5 1.5 0 001.033.392 1.5 1.5 0 00.417-.063l4.5-1.5a1.5 1.5 0 001-1.415V1.541a1.5 1.5 0 00-1-1.415l-4.5-1.5a1.5 1.5 0 00-.417-.063zM18 6.927v10.146L10.893 12z"
				/>
			</svg>
			VS Code Extension
		</a>
		<a href="mailto:support@context-engine.ai" class="mobile-nav-link" onclick={closeMobileMenu}>
			<Mail size={20} />
			Support
		</a>
		<a
			href="https://dev.context-engine.ai/login"
			class="mobile-nav-link mobile-login-link"
			onclick={closeMobileMenu}
		>
			<LogIn size={20} />
			Login / Sign Up
		</a>
		<div class="mobile-menu-cta">
			<a href="#demo" class="btn btn-primary btn-lg" onclick={closeMobileMenu}>Request Demo</a>
		</div>
	</nav>
{/if}

<main>
	{@render children()}
</main>

<style>
	main {
		position: relative;
		z-index: 1;
		min-height: calc(100vh - 64px);
	}
</style>

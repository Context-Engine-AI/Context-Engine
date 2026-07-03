import { error } from '@sveltejs/kit';
import { marked } from 'marked';
import hljs from 'highlight.js';
import { base } from '$app/paths';

export const prerender = true;

// Configure marked with syntax highlighting
marked.setOptions({
	breaks: true,
	gfm: true
});

// Import markdown files at build time
const markdownFiles = import.meta.glob(['/docs/*.md', '/README.md'], { query: '?raw', import: 'default', eager: true });

// Map of slug to document info and file path
const docMap: Record<string, { title: string; description: string; filePath: string }> = {
	'readme': {
		title: 'README',
		description: 'Context Engine overview and quick start',
		filePath: '/README.md'
	},
	'getting-started': {
		title: 'Getting Started',
		description: 'Quick start guide for Context Engine',
		filePath: '/docs/GETTING_STARTED.md'
	},
	'configuration': {
		title: 'Configuration', 
		description: 'Configure Context Engine for your needs',
		filePath: '/docs/CONFIGURATION.md'
	},
	'architecture': {
		title: 'Architecture',
		description: 'Understanding Context Engine\'s architecture',
		filePath: '/docs/ARCHITECTURE.md'
	},
	'troubleshooting': {
		title: 'Troubleshooting',
		description: 'Solve common issues and problems',
		filePath: '/docs/TROUBLESHOOTING.md'
	},
	'development': {
		title: 'Development',
		description: 'Development setup and contributing',
		filePath: '/docs/DEVELOPMENT.md'
	},
	'mcp-api': {
		title: 'MCP API',
		description: 'Model Context Protocol API reference',
		filePath: '/docs/MCP_API.md'
	},
	'ide-clients': {
		title: 'IDE Clients',
		description: 'Configure IDE clients for Context Engine',
		filePath: '/docs/IDE_CLIENTS.md'
	},
	'ctx-cli': {
		title: 'ctx CLI',
		description: 'Command line interface documentation',
		filePath: '/docs/CTX_CLI.md'
	},
	'memory-guide': {
		title: 'Memory Guide',
		description: 'Understanding Context Engine memory systems',
		filePath: '/docs/MEMORY_GUIDE.md'
	},
	'multi-repo': {
		title: 'Multi-Repo Collections',
		description: 'Working with multiple repositories',
		filePath: '/docs/MULTI_REPO_COLLECTIONS.md'
	},
	'observability': {
		title: 'Observability',
		description: 'Monitoring and observability features',
		filePath: '/docs/OBSERVABILITY.md'
	},
	'vscode-extension': {
		title: 'VS Code Extension',
		description: 'VS Code extension setup and usage',
		filePath: '/docs/vscode-extension.md'
	}
};

// Create reverse mapping: filename -> slug
const fileToSlug: Record<string, string> = {};
Object.entries(docMap).forEach(([slug, info]) => {
	const filename = info.filePath.split('/').pop() || '';
	fileToSlug[filename] = slug;
});

// Custom renderer to transform GitHub links to website routes
const renderer = new marked.Renderer();

// Function to generate slug from heading text (matches GitHub's behavior)
function generateSlug(text: string): string {
	return text
		.toLowerCase()
		.trim()
		// Handle parentheses: remove them but keep the content
		.replace(/\s*\(\s*/g, ' ')
		.replace(/\s*\)\s*/g, ' ')
		// Handle ampersand (&) - GitHub converts & to --
		.replace(/\s*&\s*/g, '--')
		// Handle forward slash (/) - GitHub converts / to --
		.replace(/\s*\/\s*/g, '--')
		// Handle dots in compound words (like llama.cpp -> llamacpp)
		.replace(/\.(?=[a-z])/g, '')
		// Replace sequences of non-alphanumeric characters with single hyphen
		.replace(/[^a-z0-9-]+/g, '-')
		// Clean up multiple consecutive hyphens (but preserve -- from & and /)
		.replace(/-{3,}/g, '--')
		// Remove leading/trailing hyphens
		.replace(/^-+|-+$/g, '');
}

// Custom heading renderer to generate proper IDs
renderer.heading = function({ tokens, depth }) {
	// Extract text from tokens
	let text = '';
	if (tokens && tokens.length > 0) {
		text = tokens.map(token => 'text' in token ? token.text : '').join('');
	}
	
	const id = generateSlug(text);
	return `<h${depth} id="${id}">${text}</h${depth}>`;
};

// Custom link renderer
renderer.link = function({ href, title, tokens }) {
	// Extract text from tokens
	let text = '';
	if (tokens && tokens.length > 0) {
		text = tokens.map(token => 'text' in token ? token.text : '').join('');
	}
	
	// Guard against null/undefined href
	if (!href) {
		return text;
	}
	
	// Check if this is a link to a markdown file
	if (href.endsWith('.md')) {
		const filename = href.split('/').pop() || '';
		const slug = fileToSlug[filename];
		
		if (slug) {
			// Transform to website route with base path
			href = `${base}/docs/${slug}`;
		}
	}
	
	// Handle relative links to other docs (including ../README.md)
	if (href.startsWith('../') && href.endsWith('.md')) {
		const filename = href.split('/').pop() || '';
		
		const slug = fileToSlug[filename];
		if (slug) {
			href = `${base}/docs/${slug}`;
		}
	}
	
	// Build the link HTML
	const titleAttr = title ? ` title="${title}"` : '';
	return `<a href="${href}"${titleAttr}>${text}</a>`;
};

marked.setOptions({ renderer });

export async function load({ params }) {
	const { slug } = params;
	
	if (!docMap[slug]) {
		throw error(404, 'Documentation not found');
	}

	const docInfo = docMap[slug];
	
	try {
		// Get the markdown content from the imported files
		const markdownContent = markdownFiles[docInfo.filePath] as string;
		
		if (!markdownContent) {
			throw new Error(`Markdown file not found: ${docInfo.filePath}`);
		}
		
		// Convert markdown to HTML with syntax highlighting
		const html = marked(markdownContent);
		
		return {
			slug,
			title: docInfo.title,
			description: docInfo.description,
			content: html,
			filename: docInfo.filePath.replace('/docs/', '')
		};
	} catch (err) {
		console.error(`Error processing ${slug}:`, err);
		throw error(500, 'Failed to load documentation');
	}
}

// Export entries for prerendering
export function entries() {
	return Object.keys(docMap).map(slug => ({ slug }));
}
import adapter from '@sveltejs/adapter-static';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

/** @type {import('@sveltejs/kit').Config} */
const config = {
	// Consult https://svelte.dev/docs/kit/integrations
	// for more information about preprocessors
	preprocess: vitePreprocess(),

	kit: {
		// adapter-static for GitHub Pages
		adapter: adapter({
			pages: 'build',
			assets: 'build',
			fallback: undefined,
			precompress: false,
			strict: false
		}),
		paths: {
			base: process.argv.includes('dev') ? '' : process.env.BASE_PATH || '/Context-Engine'
		},
		prerender: {
			handleHttpError: ({ path, referrer, message }) => {
				// Ignore all HTTP errors during prerender
				console.warn(`Ignoring HTTP error: ${path} (${message})`);
				return;
			},
			handleMissingId: 'ignore',
			crawl: false
		}
	}
};

export default config;

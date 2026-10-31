import { expect, test } from '@playwright/test';

test('home page explains model-agnostic intelligence and paid Starter pricing', async ({
	page
}) => {
	await page.setViewportSize({ width: 1440, height: 1000 });
	await page.goto('/');
	await expect(page.locator('h1')).toContainText('Codebase intelligence');
	await expect(page.locator('.hero-subtitle')).toContainText('agents, IDEs, and any model');
	await expect(page.locator('.code-window')).toContainText('Who consumes CheckoutCreated?');
	await expect(page.locator('.code-window')).toContainText('Illustrative source-backed context');
	await expect(
		page.getByRole('heading', { name: 'Fewer calls. Fewer turns. Lower cost.' })
	).toBeVisible();
	await page.getByText('About the evaluation', { exact: true }).click();
	await expect(page.locator('.evaluation-method')).toContainText(
		'80 paired runs (40 code-navigation tasks'
	);
	await expect(page.locator('.evaluation-method')).toContainText('60 paired audit tasks');
	await expect(page.locator('.evaluation-results').first()).toContainText('26% fewer');
	await expect(page.locator('.evaluation-results').nth(1)).toContainText('25% less');
	await expect(page.getByRole('link', { name: 'CoIR code-retrieval benchmark ↗' })).toHaveAttribute(
		'href',
		'https://github.com/CoIR-team/coir'
	);
	await expect(
		page.getByRole('link', { name: 'TokensToGold methodology and scoring code ↗' })
	).toHaveAttribute('href', 'https://github.com/noodlbox/tokens-to-gold');
	await expect(
		page.getByRole('heading', { name: 'Organization Knowledge', exact: true })
	).toBeVisible();
	await expect(
		page.getByRole('heading', { name: 'Cross-Repo Intelligence', exact: true })
	).toBeVisible();
	const starter = page.locator('#pricing .feature-card').filter({
		has: page.getByRole('heading', { name: 'Starter', exact: true })
	});
	await expect(starter).toContainText('$5');
	await expect(starter).toContainText('$48/year');
	await expect(starter).toContainText('2 repositories');
	await expect(
		page.getByRole('link', { name: 'Get started', exact: true }).first()
	).toHaveAttribute('href', 'https://dev.context-engine.ai/auth/signup');
	await expect(page.getByText(/start free|try for free|join the beta/i)).toHaveCount(0);
	await expect(page.getByRole('link', { name: 'Helion DB ↗' })).toHaveAttribute(
		'href',
		'https://github.com/Context-Engine-AI/helion-db'
	);
	await expect(page.getByRole('link', { name: 'Context Engine Reviewer ↗' })).toHaveAttribute(
		'href',
		'https://github.com/Context-Engine-AI/Context-Engine-Reviewer'
	);
	await expect(page.getByRole('link', { name: 'Open VSX ↗', exact: true })).toHaveAttribute(
		'href',
		'https://open-vsx.org/extension/context-engine/context-engine-uploader'
	);
	await expect(page.getByText('BYOF — Bring Your Own Factory')).toBeVisible();
	await expect(page.getByRole('link', { name: /Explore SWE-AF/ })).toHaveAttribute(
		'href',
		'https://github.com/Agent-Field/SWE-AF'
	);
	await page.screenshot({ path: 'test-results/paid-homepage-desktop.png', fullPage: true });
});

for (const width of [320, 390, 768]) {
	test(`paid-plan homepage fits ${width}px and keeps signup accessible`, async ({ page }) => {
		await page.setViewportSize({ width, height: 844 });
		await page.goto('/');
		await expect(
			page.getByRole('link', { name: 'Get started', exact: true }).first()
		).toBeVisible();
		await expect(page.locator('#pricing')).toBeVisible();
		expect(
			await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)
		).toBe(true);
		await page.screenshot({ path: `test-results/paid-homepage-${width}.png`, fullPage: true });
	});
}

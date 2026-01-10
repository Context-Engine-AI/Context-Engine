<script lang="ts">
	import '../app.scss';

	let { children } = $props();
</script>

<div class="bg-gradient"></div>
<div class="bg-overlay"></div>

<div class="app">
	<main>
		{@render children()}
	</main>
</div>

<style lang="scss">
	.app {
		position: relative;
		z-index: 1;
		min-height: 100vh;
	}
	
	main {
		min-height: 100vh;
		width: 100%;
	}
</style>

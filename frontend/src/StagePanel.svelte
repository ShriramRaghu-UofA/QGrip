<script lang="ts">
  import type { Snippet } from 'svelte';

  interface Props {
    title: string;
    description: string;
    active: boolean;
    children: Snippet;
  }

  let { title, description, active, children }: Props = $props();
</script>

<section class={['card border border-base-300 bg-base-200 shadow-xl', active && 'ring-2 ring-primary']} aria-labelledby={`stage-${title}`}>
  <div class="card-body gap-4">
    <div>
      <div class="badge badge-primary badge-outline mb-2">{active ? 'Current stage' : 'Workflow stage'}</div>
      <h2 id={`stage-${title}`} class="card-title text-2xl">{title}</h2>
      <p class="text-base-content/70">{description}</p>
    </div>
    {@render children()}
  </div>
</section>

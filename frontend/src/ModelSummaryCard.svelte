<script lang="ts">
  import type { ModelSummary } from './api';

  let { summary, loading = false }: { summary?: ModelSummary | null; loading?: boolean } = $props();

  const integer = new Intl.NumberFormat();
  const configDocument = $derived(summary ? JSON.stringify(summary.model_config, null, 2) : '');
</script>

<section class="card border border-base-300 bg-base-100" aria-label="Model summary">
  <div class="card-body gap-4">
    <div class="flex flex-wrap items-start justify-between gap-3">
      <div>
        <h3 class="card-title text-lg">Model architecture</h3>
        {#if summary}
          <p class="font-mono text-sm">{summary.model_class}</p>
        {:else if !loading}
          <p class="text-sm text-base-content/60">Model details are unavailable.</p>
        {/if}
      </div>
      {#if loading}
        <span class="loading loading-spinner loading-sm" aria-label="Loading model summary"></span>
      {:else if summary}
        <div class="flex gap-2">
          <span class="badge badge-primary uppercase">{summary.model_name}</span>
          <span class="badge badge-outline">{summary.source === 'checkpoint' ? 'Checkpoint' : 'Preset preview'}</span>
        </div>
      {/if}
    </div>

    {#if summary}
      <div class="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <div class="rounded-box bg-base-300 p-3">
          <div class="text-xs text-base-content/60">Parameters</div>
          <div class="text-lg font-semibold tabular-nums">{integer.format(summary.parameter_count)}</div>
          <div class="text-xs">{integer.format(summary.trainable_parameter_count)} trainable</div>
        </div>
        <div class="rounded-box bg-base-300 p-3">
          <div class="text-xs text-base-content/60">Input</div>
          <div class="text-lg font-semibold tabular-nums">{summary.window_size.toLocaleString()} × {summary.channels}</div>
          <div class="text-xs">samples × channels · {summary.sample_rate_hz.toLocaleString()} Hz</div>
        </div>
        <div class="rounded-box bg-base-300 p-3">
          <div class="text-xs text-base-content/60">Output</div>
          <div class="text-lg font-semibold">{summary.labels.length} classes</div>
          <div class="text-xs">{summary.proportional ? 'Proportional activation head' : 'Discrete classification'}</div>
        </div>
        <div class="rounded-box bg-base-300 p-3">
          <div class="text-xs text-base-content/60">Normalization</div>
          <div class="text-lg font-semibold">{summary.normalization.replace(/_/g, ' ')}</div>
          {#if summary.validation_accuracy != null}
            <div class="text-xs">Validation {(summary.validation_accuracy * 100).toFixed(1)}% · loss {summary.validation_loss?.toFixed(4) ?? '—'}</div>
          {:else}
            <div class="text-xs">Profile-derived preview</div>
          {/if}
        </div>
      </div>

      <details class="collapse collapse-arrow border border-base-300 bg-base-200">
        <summary class="collapse-title font-semibold">Architecture details</summary>
        <div class="collapse-content grid gap-4 lg:grid-cols-2">
          <div class="min-w-0">
            <h4 class="mb-2 text-sm font-semibold">Canonical model config</h4>
            <pre class="max-h-96 overflow-auto rounded-box bg-base-300 p-3 text-xs"><code>{configDocument}</code></pre>
          </div>
          <div class="min-w-0">
            <h4 class="mb-2 text-sm font-semibold">Torch module tree</h4>
            <pre class="max-h-96 overflow-auto rounded-box bg-base-300 p-3 text-xs"><code>{summary.module_tree}</code></pre>
          </div>
        </div>
      </details>
    {/if}
  </div>
</section>

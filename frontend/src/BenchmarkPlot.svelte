<script lang="ts">
  import type { BenchmarkResult } from './api';

  let { results = [] }: { results?: BenchmarkResult[] } = $props();

  const maximumLatency = $derived(
    Math.max(1, ...results.flatMap((result) => [result.median_ms, result.p95_ms, result.p99_ms])),
  );
  const maximumThroughput = $derived(Math.max(1, ...results.map((result) => result.throughput_hz)));

  function runtimeLabel(result: BenchmarkResult): string {
    return `${result.backend.toUpperCase()} · ${result.device.toUpperCase()}`;
  }

  function barWidth(value: number, maximum: number): string {
    return `${Math.max(1, (value / maximum) * 100)}%`;
  }
</script>

{#if results.length}
  <div class="alert bg-base-200 text-sm" aria-label="Benchmark run summary">
    <span>
      Each runtime measured <strong>{results[0].iterations.toLocaleString()}</strong> timed batch-1 windows after
      <strong>{results[0].warmup.toLocaleString()}</strong> warmup windows. Window shape:
      <strong>{results[0].window_size.toLocaleString()} × {results[0].channels}</strong>.
    </span>
  </div>

  <div class="grid gap-4 xl:grid-cols-2">
    <figure class="rounded-box bg-base-300 p-4" aria-label="Benchmark latency percentile bars">
      <figcaption class="mb-4">
        <h4 class="font-semibold">Latency percentiles</h4>
        <p class="text-xs text-base-content/60">Grouped horizontal bars · lower is better</p>
      </figcaption>
      <div class="mb-3 flex flex-wrap gap-3 text-xs" aria-hidden="true">
        <span class="flex items-center gap-1"><span class="size-2 rounded-full bg-success"></span>Median</span>
        <span class="flex items-center gap-1"><span class="size-2 rounded-full bg-warning"></span>p95</span>
        <span class="flex items-center gap-1"><span class="size-2 rounded-full bg-error"></span>p99</span>
      </div>
      <div class="space-y-5">
        {#each results as result (`${result.backend}-${result.device}`)}
          <div class="grid gap-2 sm:grid-cols-[8rem_minmax(16rem,1fr)]">
            <div class="self-center text-sm font-semibold leading-tight">{runtimeLabel(result)}</div>
            <div class="space-y-1.5">
              <div class="flex items-center gap-2 text-xs">
                <span class="w-12 shrink-0">Median</span>
                <div class="h-3 min-w-0 flex-1 overflow-hidden rounded-full bg-base-100"><div class="h-full rounded-full bg-success" style:width={barWidth(result.median_ms, maximumLatency)}></div></div>
                <span class="w-20 shrink-0 text-right tabular-nums">{result.median_ms.toFixed(3)} ms</span>
              </div>
              <div class="flex items-center gap-2 text-xs">
                <span class="w-12 shrink-0">p95</span>
                <div class="h-3 min-w-0 flex-1 overflow-hidden rounded-full bg-base-100"><div class="h-full rounded-full bg-warning" style:width={barWidth(result.p95_ms, maximumLatency)}></div></div>
                <span class="w-20 shrink-0 text-right tabular-nums">{result.p95_ms.toFixed(3)} ms</span>
              </div>
              <div class="flex items-center gap-2 text-xs">
                <span class="w-12 shrink-0">p99</span>
                <div class="h-3 min-w-0 flex-1 overflow-hidden rounded-full bg-base-100"><div class="h-full rounded-full bg-error" style:width={barWidth(result.p99_ms, maximumLatency)}></div></div>
                <span class="w-20 shrink-0 text-right tabular-nums">{result.p99_ms.toFixed(3)} ms</span>
              </div>
            </div>
          </div>
        {/each}
      </div>
    </figure>

    <figure class="rounded-box bg-base-300 p-4" aria-label="Benchmark throughput bars">
      <figcaption class="mb-4">
        <h4 class="font-semibold">Throughput</h4>
        <p class="text-xs text-base-content/60">Horizontal bars · higher is better</p>
      </figcaption>
      <div class="space-y-5">
        {#each results as result (`${result.backend}-${result.device}`)}
          <div class="grid gap-2 sm:grid-cols-[8rem_minmax(16rem,1fr)]">
            <div class="self-center text-sm font-semibold leading-tight">{runtimeLabel(result)}</div>
            <div class="flex items-center gap-2">
              <div class="h-6 min-w-0 flex-1 overflow-hidden rounded-full bg-base-100"><div class="h-full rounded-full bg-info" style:width={barWidth(result.throughput_hz, maximumThroughput)}></div></div>
              <span class="w-24 shrink-0 text-right text-xs tabular-nums">{result.throughput_hz.toFixed(1)}/s</span>
            </div>
          </div>
        {/each}
      </div>
    </figure>
  </div>
{/if}

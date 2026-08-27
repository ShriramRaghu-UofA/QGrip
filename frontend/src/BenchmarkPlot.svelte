<script lang="ts">
  import type { Attachment } from 'svelte/attachments';
  import uPlot from 'uplot';

  import type { BenchmarkResult } from './api';

  let { results = [] }: { results?: BenchmarkResult[] } = $props();

  const labels = $derived(results.map((result) => `${result.backend.toUpperCase()} · ${result.device.toUpperCase()}`));

  /** Attach and clean up a categorical latency-percentile comparison chart. */
  const renderLatency: Attachment<HTMLDivElement> = (container) => {
    const x = results.map((_, index) => index);
    const instance = new uPlot(
      {
        width: Math.max(360, container.clientWidth),
        height: 260,
        title: 'Latency percentiles · lower is better',
        scales: { x: { time: false } },
        axes: [
          { values: (_plot, ticks) => ticks.map((tick) => labels[Math.round(tick)] ?? '') },
          { label: 'milliseconds' },
        ],
        series: [
          {},
          { label: 'Median', stroke: '#50fa7b', width: 3, points: { show: true } },
          { label: 'p95', stroke: '#ffb86c', width: 3, points: { show: true } },
          { label: 'p99', stroke: '#ff5555', width: 3, points: { show: true } },
        ],
      },
      [
        x,
        results.map((result) => result.median_ms),
        results.map((result) => result.p95_ms),
        results.map((result) => result.p99_ms),
      ],
      container,
    );
    return () => instance.destroy();
  };

  /** Attach and clean up a throughput comparison chart. */
  const renderThroughput: Attachment<HTMLDivElement> = (container) => {
    const x = results.map((_, index) => index);
    const instance = new uPlot(
      {
        width: Math.max(360, container.clientWidth),
        height: 220,
        title: 'Throughput · higher is better',
        scales: { x: { time: false } },
        axes: [
          { values: (_plot, ticks) => ticks.map((tick) => labels[Math.round(tick)] ?? '') },
          { label: 'predictions / second' },
        ],
        series: [
          {},
          { label: 'Throughput', stroke: '#8be9fd', fill: '#8be9fd33', width: 3, points: { show: true } },
        ],
      },
      [x, results.map((result) => result.throughput_hz)],
      container,
    );
    return () => instance.destroy();
  };
</script>

<div class="grid gap-4 xl:grid-cols-2">
  <div class="overflow-x-auto rounded-box bg-base-300 p-3" {@attach renderLatency} aria-label="Benchmark latency percentile plot"></div>
  <div class="overflow-x-auto rounded-box bg-base-300 p-3" {@attach renderThroughput} aria-label="Benchmark throughput plot"></div>
</div>

<style>
  div :global(.uplot) {
    font-family: inherit;
  }
</style>

<script lang="ts">
  import type { Attachment } from 'svelte/attachments';
  import uPlot from 'uplot';

  const renderPlot: Attachment<HTMLDivElement> = (container) => {
    const time = Array.from({ length: 80 }, (_, index) => index / 20);
    const confidence = time.map((value) => 0.55 + Math.sin(value * 4) * 0.12);
    const activation = time.map((value) => 0.35 + Math.cos(value * 3) * 0.18);
    const instance = new uPlot(
      {
        width: Math.max(320, container.clientWidth),
        height: 220,
        title: 'Prediction history',
        scales: { y: { range: [0, 1] } },
        series: [
          {},
          { label: 'Confidence', stroke: '#50fa7b', width: 2 },
          { label: 'Activation', stroke: '#bd93f9', width: 2 },
        ],
      },
      [time, confidence, activation],
      container,
    );
    return () => instance.destroy();
  };
</script>

<div
  class="plot-shell overflow-x-auto rounded-box bg-base-300 p-3"
  {@attach renderPlot}
  aria-label="Confidence and activation history plot"
></div>

<style>
  .plot-shell :global(.uplot) {
    font-family: inherit;
  }
</style>

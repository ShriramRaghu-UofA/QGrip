<script lang="ts">
  import type { Attachment } from 'svelte/attachments';
  import uPlot from 'uplot';

  import type { Prediction } from './api';

  let { history = [] }: { history?: Prediction[] } = $props();

  const renderPlot: Attachment<HTMLDivElement> = (container) => {
    const values = history.length ? history : [{ confidence: 0, activation: 0 }];
    const time = values.map((_, index) => index);
    const confidence = values.map((value) => value.confidence);
    const activation = values.map((value) => value.activation);
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

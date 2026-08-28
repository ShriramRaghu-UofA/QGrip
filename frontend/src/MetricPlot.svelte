<script lang="ts">
  import type { Attachment } from 'svelte/attachments';
  import uPlot from 'uplot';

  import type { Prediction } from './api';

  /** Most recent accepted predictions, plotted in arrival order. */
  let { history = [] }: { history?: Prediction[] } = $props();

  /** Attach and clean up a compact uPlot confidence/activation visualization. */
  const renderPlot: Attachment<HTMLDivElement> = (container) => {
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
      [[0], [0], [0]],
      container,
    );

    // Keep the third-party widget mounted while live predictions arrive. Reading
    // `history` in the attachment itself would make Svelte destroy and recreate
    // the plot for every sample, briefly collapsing its height and disturbing the
    // page's scroll position.
    $effect(() => {
      const values = history.length ? history : [{ confidence: 0, activation: 0 }];
      instance.setData([
        values.map((_, index) => index),
        values.map((value) => value.confidence),
        values.map((value) => value.activation),
      ]);
    });

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

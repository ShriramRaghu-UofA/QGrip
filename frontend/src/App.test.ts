import { cleanup, render, screen, waitFor } from "@testing-library/svelte";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, expect, test, vi } from "vitest";
import App from "./App.svelte";

const { plotConstructor } = vi.hoisted(() => ({
  plotConstructor: vi.fn(),
}));

vi.mock("uplot", () => ({
  default: class {
    constructor() {
      plotConstructor();
    }

    setData(): void {}
    destroy(): void {}
  },
}));

function modelSummary(path: string): object {
  const checkpoint = path.includes("/checkpoints/");
  const dense = path.includes("dense") || checkpoint;
  return {
    source: checkpoint ? "checkpoint" : "preset",
    model_name: dense ? "dense" : "transformer",
    model_class: dense ? "DenseEMGClassifier" : "TransformerEMGClassifier",
    model_config: {
      n_classes: 3,
      n_channels: 8,
      predict_activation: true,
    },
    labels: ["rest", "open", "close"],
    window_size: 200,
    channels: 8,
    sample_rate_hz: 200,
    normalization: "window_zscore",
    proportional: true,
    parameter_count: dense ? 1234 : 5678,
    trainable_parameter_count: dense ? 1234 : 5678,
    module_tree: dense
      ? "DenseEMGClassifier(...)"
      : "TransformerEMGClassifier(...)\n",
    checkpoint: checkpoint ? "C:/data/model.pt" : null,
    validation_loss: checkpoint ? 0.1234 : null,
    validation_accuracy: checkpoint ? 0.91 : null,
  };
}

beforeEach(() => {
  plotConstructor.mockClear();
  localStorage.clear();
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL) => {
      const path = String(input);
      let body: object = {
        api_version: 1,
        profile: "synthetic.json",
        device: "synthetic",
        gestures: ["rest", "open", "close"],
        models: ["transformer", "dense"],
        proportional: true,
        activation_tolerance: 0.1,
      };
      if (path.includes("/api/v1/artifacts"))
        body = { artifacts: [], calibration_ready: false };
      if (path.includes("/summary")) body = modelSummary(path);
      if (path.includes("/api/v1/doctor"))
        body = {
          ready: true,
          kind: "synthetic",
          sample_rate_hz: 200,
          channels: 8,
        };
      return new Response(JSON.stringify(body), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    }),
  );
});

afterEach(cleanup);

test("all workflow stages are keyboard-accessible", async () => {
  const user = userEvent.setup();
  render(App);
  for (const stage of ["Setup", "Collect", "Train", "Validate"]) {
    await user.click(screen.getByRole("button", { name: stage }));
    expect(
      screen.getByRole("heading", { name: stage, level: 2 }),
    ).toBeInTheDocument();
  }
  expect(
    screen.queryByRole("button", { name: "Handi" }),
  ).not.toBeInTheDocument();
});

test("training exposes a compact model summary with expandable details", async () => {
  const user = userEvent.setup();
  render(App);
  await user.click(screen.getByRole("button", { name: "Train" }));
  await waitFor(() =>
    expect(screen.getByText("TransformerEMGClassifier")).toBeInTheDocument(),
  );
  expect(screen.getByText("5,678")).toBeInTheDocument();
  const details = screen.getByText("Architecture details").closest("details");
  expect(details).not.toHaveAttribute("open");

  await user.selectOptions(
    screen.getByRole("combobox", { name: "Model preset" }),
    "dense",
  );
  await waitFor(() =>
    expect(screen.getByText("DenseEMGClassifier")).toBeInTheDocument(),
  );
  expect(screen.getByText("1,234")).toBeInTheDocument();
});

test("theme selection persists", async () => {
  const user = userEvent.setup();
  render(App);
  await user.selectOptions(
    screen.getByRole("combobox", { name: "Theme" }),
    "nord",
  );
  expect(localStorage.getItem("qgrip-theme")).toBe("nord");
});

test("subject can be applied before connecting a device", async () => {
  const user = userEvent.setup();
  render(App);
  const subjectInput = screen.getByRole("textbox", { name: "Subject" });
  await user.clear(subjectInput);
  await user.type(subjectInput, "alice");
  await user.click(
    screen.getByRole("button", { name: "Continue to collection" }),
  );

  expect(screen.getByText("Calibrate activation")).toBeInTheDocument();
  expect(vi.mocked(fetch)).not.toHaveBeenCalledWith(
    "/api/v1/doctor",
    expect.anything(),
  );
  expect(
    vi
      .mocked(fetch)
      .mock.calls.some(([input]) => String(input).includes("subject=alice")),
  ).toBe(true);
});

test("proportional collection makes calibration the required next step", async () => {
  const user = userEvent.setup();
  render(App);
  await user.click(screen.getByRole("button", { name: "Connect device" }));
  const continueButton = screen.getByRole("button", {
    name: "Continue to collection",
  });
  await waitFor(() => expect(continueButton).toBeEnabled());
  await user.click(continueButton);
  expect(screen.getByText("Calibrate activation")).toBeInTheDocument();
  expect(screen.getByText("Locked")).toBeInTheDocument();
  expect(
    screen.getByRole("button", { name: "Start required calibration" }),
  ).toBeEnabled();
  expect(
    screen.queryByRole("button", { name: "Start training data collection" }),
  ).not.toBeInTheDocument();
});

test("discrete collection skips calibration", async () => {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL) => {
      const path = String(input);
      let body: object = {
        api_version: 1,
        profile: "synthetic.json",
        device: "synthetic",
        gestures: ["rest", "open"],
        models: ["dense"],
        proportional: false,
        activation_tolerance: 0.1,
      };
      if (path.includes("/api/v1/artifacts"))
        body = { artifacts: [], calibration_ready: false };
      if (path.includes("/summary")) body = modelSummary(path);
      if (path.includes("/api/v1/doctor"))
        body = {
          ready: true,
          kind: "synthetic",
          sample_rate_hz: 200,
          channels: 8,
        };
      if (path.includes("/api/v1/sgt/start"))
        body = { state: "running", kind: "sgt" };
      return new Response(JSON.stringify(body), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    }),
  );
  const user = userEvent.setup();
  render(App);
  await user.click(screen.getByRole("button", { name: "Connect device" }));
  const continueButton = screen.getByRole("button", {
    name: "Continue to collection",
  });
  await waitFor(() => expect(continueButton).toBeEnabled());
  await user.click(continueButton);
  expect(screen.getByText("No calibration needed")).toBeInTheDocument();
  await user.click(
    screen.getByRole("button", { name: "Start training data collection" }),
  );
  const startCall = vi
    .mocked(fetch)
    .mock.calls.find(([input]) => String(input).includes("/api/v1/sgt/start"));
  expect(startCall).toBeDefined();
  expect(JSON.parse(String(startCall?.[1]?.body))).toMatchObject({
    discrete: true,
  });
});

test("SGT activation guidance remains visible during preparation", async () => {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL) => {
      const path = String(input);
      let body: object = {
        api_version: 1,
        profile: "synthetic.json",
        device: "synthetic",
        gestures: ["rest", "open"],
        models: ["dense"],
        proportional: true,
        activation_tolerance: 0.1,
      };
      if (path.includes("/api/v1/artifacts"))
        body = { artifacts: [], calibration_ready: true };
      if (path.includes("/summary")) body = modelSummary(path);
      if (path.includes("/api/v1/doctor"))
        body = {
          ready: true,
          kind: "synthetic",
          sample_rate_hz: 200,
          channels: 8,
        };
      if (path.includes("/api/v1/sgt/start"))
        body = {
          state: "running",
          kind: "sgt",
          stage: "preparation",
          gesture: "open",
          trial: 2,
          total_trials: 4,
          progress: 0.5,
          activation: 0.75,
          measured_activation: 0.42,
          duration_seconds: 2,
        };
      return new Response(JSON.stringify(body), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    }),
  );
  const user = userEvent.setup();
  render(App);
  await user.click(screen.getByRole("button", { name: "Connect device" }));
  const continueButton = screen.getByRole("button", {
    name: "Continue to collection",
  });
  await waitFor(() => expect(continueButton).toBeEnabled());
  await user.click(continueButton);
  await user.click(
    screen.getByRole("button", { name: "Start training data collection" }),
  );
  await waitFor(() =>
    expect(screen.getByText("Next target")).toBeInTheDocument(),
  );
  expect(screen.getByText("75%")).toBeInTheDocument();
  expect(screen.getByText("42%")).toBeInTheDocument();
  expect(screen.getByText("Session progress")).toBeInTheDocument();
  expect(screen.getByText("50%")).toBeInTheDocument();
  expect(screen.getByText("Get ready")).toBeInTheDocument();
});

test("live inference renders backend predictions", async () => {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL) => {
      const path = String(input);
      let body: object = {
        api_version: 1,
        profile: "synthetic.json",
        device: "synthetic",
        gestures: ["rest", "open", "close"],
        models: ["dense"],
        proportional: true,
        activation_tolerance: 0.1,
      };
      if (path.includes("/api/v1/artifacts"))
        body = { artifacts: ["C:/data/model.pt"], calibration_ready: false };
      if (path.includes("/summary")) body = modelSummary(path);
      if (path.includes("/api/v1/inference/start"))
        body = { state: "running", kind: "inference" };
      if (path.includes("/api/v1/inference/status"))
        body = {
          state: "running",
          kind: "inference",
          prediction: {
            gesture: "open",
            confidence: 0.91,
            activation: 0.64,
            latency_ms: 3.2,
          },
          health: {
            severity: "warning",
            warnings: ["device samples lost"],
            missing_values: 2,
            lost_samples: 3,
            malformed_packets: 0,
            misaligned_packets: 0,
            consumer_overruns: 1,
          },
        };
      return new Response(JSON.stringify(body), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    }),
  );
  const user = userEvent.setup();
  render(App);
  await user.click(screen.getByRole("button", { name: "Validate" }));
  await waitFor(() =>
    expect(screen.getByText(/Validation 91\.0%/)).toBeInTheDocument(),
  );
  expect(screen.getByText(/loss 0\.1234/)).toBeInTheDocument();
  const start = screen.getByRole("button", { name: "Start live inference" });
  await waitFor(() => expect(start).toBeEnabled());
  await user.click(start);
  await waitFor(() => expect(screen.getByText("91%")).toBeInTheDocument(), {
    timeout: 2000,
  });
  expect(plotConstructor).toHaveBeenCalledTimes(1);
  expect(screen.getByText("open")).toBeInTheDocument();
  expect(screen.getByText("64%")).toBeInTheDocument();
  expect(screen.getByText(/Device loss: 3/)).toBeInTheDocument();
  expect(screen.getByText(/consumer overruns: 1/)).toBeInTheDocument();
});

test("benchmark renders CPU and available GPU comparisons", async () => {
  let benchmarkBody: { iterations: number; warmup: number } | undefined;
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const path = String(input);
      let body: object = {
        api_version: 1,
        profile: "synthetic.json",
        device: "synthetic",
        device_preference: "gpu",
        gestures: ["rest", "open"],
        models: ["dense"],
        proportional: true,
        activation_tolerance: 0.1,
      };
      if (path.includes("/api/v1/artifacts"))
        body = { artifacts: ["C:/data/model.pt"], calibration_ready: false };
      if (path.includes("/summary")) body = modelSummary(path);
      if (path.includes("/api/v1/benchmark")) {
        benchmarkBody = JSON.parse(String(init?.body)) as {
          iterations: number;
          warmup: number;
        };
        body = {
          results: [
            {
              backend: "onnx",
              device: "cpu",
              model_name: "dense",
              iterations: benchmarkBody.iterations,
              warmup: benchmarkBody.warmup,
              window_size: 200,
              channels: 8,
              mean_ms: 1.2,
              median_ms: 1.1,
              p95_ms: 1.5,
              p99_ms: 1.6,
              min_ms: 1,
              max_ms: 1.7,
              stdev_ms: 0.1,
              throughput_hz: 833.3,
            },
            {
              backend: "onnx",
              device: "gpu",
              model_name: "dense",
              iterations: benchmarkBody.iterations,
              warmup: benchmarkBody.warmup,
              window_size: 200,
              channels: 8,
              mean_ms: 0.4,
              median_ms: 0.3,
              p95_ms: 0.5,
              p99_ms: 0.6,
              min_ms: 0.2,
              max_ms: 0.7,
              stdev_ms: 0.05,
              throughput_hz: 2500,
            },
          ],
        };
      }
      return new Response(JSON.stringify(body), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    }),
  );
  const user = userEvent.setup();
  render(App);
  await user.click(screen.getByRole("button", { name: "Validate" }));
  const benchmark = screen.getByRole("button", { name: "Run benchmark" });
  await waitFor(() => expect(benchmark).toBeEnabled());
  const iterations = screen.getByRole("spinbutton", {
    name: "Timed inference windows",
  });
  const warmup = screen.getByRole("spinbutton", { name: "Warmup windows" });
  expect(iterations).toHaveValue(1000);
  expect(warmup).toHaveValue(20);
  await user.clear(iterations);
  await user.type(iterations, "25");
  await user.clear(warmup);
  await user.type(warmup, "5");
  await user.click(benchmark);
  await waitFor(() => expect(screen.getAllByText("2500.0/s")).toHaveLength(2));
  expect(benchmarkBody).toMatchObject({ iterations: 25, warmup: 5 });
  expect(
    screen.getByLabelText("Benchmark latency percentile bars"),
  ).toBeInTheDocument();
  expect(screen.getByLabelText("Benchmark run summary")).toHaveTextContent(
    /25.*timed batch-1 windows/,
  );
  expect(screen.getAllByText(/gpu/i).length).toBeGreaterThan(0);
  expect(
    screen.getByText("Raw benchmark results").closest("details"),
  ).not.toHaveAttribute("open");
});

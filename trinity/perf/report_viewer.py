import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import streamlit as st

try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx
except ImportError:  # pragma: no cover - fallback for streamlit runtime layout changes
    from streamlit.runtime.scriptrunner_utils.script_run_context import (
        get_script_run_ctx,
    )

STEP_METRIC_PREFIXES_BY_MODULE: dict[str, list[str]] = {
    "explorer": [
        "rollout/time/run_execution/mean",
        "rollout/time/task_execution/mean",
        "rollout/prompt_length/mean",
        "rollout/response_length/mean",
        "rollout/api_call_prompt_tokens_per_second/mean",
        "rollout/api_call_response_tokens_per_second/mean",
        "experience_pipeline/experience_count",
    ],
    "trainer": [],
}

MEMORY_SERIES_KEY = "memory_rss_mb"


class PerfReportViewer:
    @staticmethod
    def run_viewer(report_path: str, port: int) -> None:
        """Start the Streamlit perf report viewer."""
        from streamlit.web import cli

        viewer_path = Path(__file__)
        sys.argv = [
            "streamlit",
            "run",
            str(viewer_path.resolve()),
            "--server.port",
            str(port),
            "--server.fileWatcherType",
            "none",
            "--",
            "--report",
            report_path,
        ]
        sys.exit(cli.main())


def launch_report_viewer(report_path: str, port: int) -> None:
    """Launch the Streamlit perf report viewer from another CLI entrypoint."""
    PerfReportViewer.run_viewer(report_path, port)


def has_streamlit_context() -> bool:
    return get_script_run_ctx() is not None


def configure_streamlit_page() -> None:
    if has_streamlit_context():
        st.set_page_config(page_title="Trinity Performance Report", layout="wide")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trinity Performance Report Viewer")
    parser.add_argument("--report", type=str, required=True, help="Path to the perf report JSON.")
    parser.add_argument(
        "--port",
        type=int,
        default=8503,
        help="Port used when auto-launching the Streamlit report viewer.",
    )
    return parser.parse_args()


def load_report(report_path: str) -> dict[str, Any]:
    report_file = Path(report_path)
    if not report_file.exists():
        raise FileNotFoundError(f"Report file not found: {report_path}")
    with report_file.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def infer_module_name(report: dict[str, Any]) -> str:
    run_meta = report.get("run_meta", {})
    return str(run_meta.get("module"))


def get_step_metric_prefixes(report: dict[str, Any]) -> list[str]:
    module_name = infer_module_name(report)
    return STEP_METRIC_PREFIXES_BY_MODULE.get(module_name, [])


def format_timestamp(timestamp: Optional[float]) -> str:
    if timestamp is None:
        return "N/A"
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")


def format_metric_value(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def metric_label(metric_name: str) -> str:
    return metric_name.replace("_", " ").title()


def gpu_series_label(gpu_payload: dict[str, Any]) -> str:
    gpu_id = gpu_payload.get("gpu_id", "?")
    gpu_name = gpu_payload.get("name")
    if gpu_name:
        return f"GPU {gpu_id} ({gpu_name})"
    return f"GPU {gpu_id}"


def render_metric_card(metric_name: str, value: Any) -> None:
    display_value = format_metric_value(value)
    label = metric_label(metric_name)
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #eef6ff 0%, #f7fbff 100%);
            border: 1px solid #d7e6fb;
            border-radius: 14px;
            padding: 16px 18px;
            min-height: 108px;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.08);
        ">
            <div style="font-size: 0.9rem; color: #4a5a70; margin-bottom: 10px;">{label}</div>
            <div style="font-size: 1.6rem; font-weight: 700; color: #0f172a;">{display_value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_elapsed_series(series: list[dict[str, Any]]) -> tuple[list[float], list[float]]:
    if not series:
        return [], []
    start_timestamp = float(series[0]["timestamp"])
    x_values = [float(point["timestamp"]) - start_timestamp for point in series]
    y_values = [float(point["value"]) for point in series]
    return x_values, y_values


def build_scalar_timeline_series(
    timeline: list[dict[str, Any]], metric_key: str
) -> list[dict[str, float]]:
    return [
        {"timestamp": sample["timestamp"], "value": sample[metric_key]}
        for sample in timeline
        if sample.get(metric_key) is not None
    ]


def build_gpu_timeline_series(
    timeline: list[dict[str, Any]], metric_key: str
) -> dict[str, dict[str, Any]]:
    series_by_gpu: dict[str, dict[str, Any]] = {}
    for sample in timeline:
        timestamp = sample.get("timestamp")
        for gpu_sample in sample.get("gpu_metrics", []):
            if gpu_sample.get(metric_key) is None:
                continue
            gpu_key = str(gpu_sample.get("gpu_id"))
            gpu_payload = series_by_gpu.setdefault(
                gpu_key,
                {
                    "gpu_id": gpu_sample.get("gpu_id"),
                    "name": gpu_sample.get("name"),
                    "values": [],
                },
            )
            gpu_payload["values"].append({"timestamp": timestamp, "value": gpu_sample[metric_key]})
    return series_by_gpu


def render_line_chart(
    title: str,
    x_values: list[float],
    y_series: dict[str, list[float]],
    y_label: str,
    legend_below: bool = False,
    legend_columns: int = 1,
) -> None:
    st.markdown(f"#### {title}")
    if not x_values or not y_series:
        st.info(f"No data for {title}.")
        return

    figure, axis = plt.subplots(figsize=(6, 2.6))
    for series_name, y_values in y_series.items():
        axis.plot(x_values[: len(y_values)], y_values, label=series_name)
    axis.set_xlabel("Elapsed Time (s)")
    axis.set_ylabel(y_label)
    axis.grid(True, alpha=0.3)
    if len(y_series) > 1:
        if legend_below:
            axis.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.28),
                ncol=min(legend_columns, len(y_series)),
                frameon=False,
                fontsize=8,
            )
            figure.subplots_adjust(bottom=0.32)
        else:
            axis.legend()
    st.pyplot(figure, clear_figure=True)


def render_step_metric_chart(step_metrics: list[dict[str, Any]], metric_key: str) -> None:
    x_values = [
        int(step_metric["step"]) for step_metric in step_metrics if metric_key in step_metric
    ]
    y_values = [
        float(step_metric[metric_key])
        for step_metric in step_metrics
        if step_metric.get(metric_key) is not None
    ]

    st.markdown(f"#### {metric_label(metric_key)}")
    if not x_values or not y_values:
        st.info(f"No data for {metric_key}.")
        return

    figure, axis = plt.subplots(figsize=(6, 2.6))
    axis.plot(x_values[: len(y_values)], y_values, marker="o")
    axis.set_xlabel("Step")
    axis.set_ylabel(metric_label(metric_key))
    axis.grid(True, alpha=0.3)
    st.pyplot(figure, clear_figure=True)


def render_header(report: dict[str, Any], report_path: str) -> None:
    run_meta = report.get("run_meta", {})
    status = report.get("status", {})

    st.title("Trinity Performance Report")
    st.caption(f"Report: {report_path}")
    st.caption(f"Generated At: {format_timestamp(run_meta.get('generated_at'))}")

    if not status.get("success"):
        st.error("Run failed.")
        if status.get("error"):
            with st.expander("Error Traceback"):
                st.code(str(status["error"]))


def compute_global_token_throughput_metrics(report: dict[str, Any]) -> dict[str, float | None]:
    timing = report.get("timing", {})
    return {
        "prompt_tokens_per_second": timing.get("prompt_tokens_per_second"),
        "response_tokens_per_second": timing.get("response_tokens_per_second"),
        "api_call_prompt_tokens_per_second": timing.get("api_call_prompt_tokens_per_second"),
        "api_call_response_tokens_per_second": timing.get("api_call_response_tokens_per_second"),
    }


def render_global_metrics(report: dict[str, Any]) -> None:
    st.header("Global Metrics")
    timing = report.get("timing", {})

    metric_items: list[tuple[str, Any]] = []
    metric_items.extend(
        (
            metric_key,
            timing.get(metric_key),
        )
        for metric_key in ("startup_time_sec", "execution_time_sec")
    )
    metric_items.extend(compute_global_token_throughput_metrics(report).items())

    shown_items = [(key, value) for key, value in metric_items if value is not None]
    if not shown_items:
        st.info("No global metrics found in this report.")
        return

    for row_start in range(0, len(shown_items), 2):
        row_items = shown_items[row_start : row_start + 2]
        columns = st.columns(len(row_items))
        for column, (metric_key, value) in zip(columns, row_items):
            with column:
                render_metric_card(metric_key, value)


def render_step_metrics(report: dict[str, Any]) -> None:
    st.header("Step Metrics")
    step_metrics = report.get("step_metrics", [])
    if not step_metrics:
        st.info("No step metrics found in this report.")
        return

    metric_prefixes = get_step_metric_prefixes(report)
    metric_keys: list[str] = []
    for step_metric in step_metrics:
        for metric_key, metric_value in step_metric.items():
            if metric_key in {"step", "raw_metrics"} or metric_value is None:
                continue
            if any(metric_key.startswith(prefix) for prefix in metric_prefixes):
                if metric_key not in metric_keys:
                    metric_keys.append(metric_key)

    if not metric_keys:
        st.info("No configured step metrics matched the current report.")
        return

    for metric_index in range(0, len(metric_keys), 2):
        columns = st.columns(2)
        for column_index, metric_key in enumerate(metric_keys[metric_index : metric_index + 2]):
            with columns[column_index]:
                render_step_metric_chart(step_metrics, metric_key)

    with st.expander("Step Metrics Table"):
        compact_rows = []
        for step_metric in step_metrics:
            compact_row = {key: value for key, value in step_metric.items() if key != "raw_metrics"}
            compact_rows.append(compact_row)
        st.dataframe(compact_rows, use_container_width=True)


def render_resource_utilization(report: dict[str, Any]) -> None:
    st.header("Resource Utilization")
    resource_timeline = report.get("resource_timeline", [])

    cpu_series = build_scalar_timeline_series(resource_timeline, "cpu_percent")
    cpu_x, cpu_y = build_elapsed_series(cpu_series)

    memory_series = build_scalar_timeline_series(resource_timeline, MEMORY_SERIES_KEY)
    memory_x, memory_y = build_elapsed_series(memory_series)

    gpu_util_series = build_gpu_timeline_series(resource_timeline, "gpu_util_percent")
    gpu_util_x: list[float] = []
    gpu_util_y: dict[str, list[float]] = {}
    for gpu_payload in gpu_util_series.values():
        gpu_util_x, values = build_elapsed_series(gpu_payload.get("values", []))
        gpu_util_y[gpu_series_label(gpu_payload)] = values
    gpu_memory_series = build_gpu_timeline_series(resource_timeline, "gpu_memory_used_mb")
    gpu_memory_x: list[float] = []
    gpu_memory_y: dict[str, list[float]] = {}
    for gpu_payload in gpu_memory_series.values():
        gpu_memory_x, values = build_elapsed_series(gpu_payload.get("values", []))
        gpu_memory_y[gpu_series_label(gpu_payload)] = values
    first_row = st.columns(2)
    with first_row[0]:
        render_line_chart("CPU Utilization", cpu_x, {"CPU": cpu_y}, "CPU %")
    with first_row[1]:
        render_line_chart("Memory Usage", memory_x, {"Memory": memory_y}, "MB")

    second_row = st.columns(2)
    with second_row[0]:
        render_line_chart(
            "GPU Utilization",
            gpu_util_x,
            gpu_util_y,
            "GPU %",
            legend_below=True,
            legend_columns=2,
        )
    with second_row[1]:
        render_line_chart(
            "GPU Memory Usage",
            gpu_memory_x,
            gpu_memory_y,
            "MB",
            legend_below=True,
            legend_columns=2,
        )


def main(args: Optional[argparse.Namespace] = None) -> None:
    configure_streamlit_page()
    if args is None:
        args = parse_args()

    try:
        report = load_report(args.report)
    except (FileNotFoundError, json.JSONDecodeError, OSError) as error:
        st.title("Trinity Perf Report Viewer")
        st.error(str(error))
        return

    render_header(report, args.report)
    render_global_metrics(report)
    render_step_metrics(report)
    render_resource_utilization(report)


if __name__ == "__main__":
    parsed_args = parse_args()
    if has_streamlit_context():
        main(parsed_args)
    else:
        launch_report_viewer(parsed_args.report, parsed_args.port)

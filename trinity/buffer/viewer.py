import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st
from transformers import AutoTokenizer

from trinity.buffer.storage.sql import SQLExperienceStorage
from trinity.common.config import StorageConfig
from trinity.common.experience import Experience
from trinity.common.experience_visualizer import build_experience_token_view


class _SyncViewerStorage:
    """Thin sync wrapper around async SQLExperienceStorage for Streamlit."""

    def __init__(self, config: StorageConfig) -> None:
        self._loop = asyncio.new_event_loop()
        self._async = SQLExperienceStorage(config)
        self._loop.run_until_complete(self._async.prepare())

    def _run(self, coro):
        return self._loop.run_until_complete(coro)

    def close(self):
        if self._loop and not self._loop.is_closed():
            self._loop.run_until_complete(self._async.engine.dispose())
            self._loop.close()

    def __del__(self):
        self.close()

    def query(self, offset: int = 0, limit: int = 10, filters=None) -> List[Experience]:
        return self._run(self._async.query(offset, limit, filters))

    def count(self, filters=None) -> int:
        return self._run(self._async.count(filters))


class SQLExperienceViewer:
    def __init__(self, config: StorageConfig) -> None:
        self.storage = _SyncViewerStorage(config)

    def get_experiences(
        self, offset: int, limit: int = 10, filters: Optional[Dict] = None
    ) -> List[Experience]:
        return self.storage.query(offset=offset, limit=limit, filters=filters)

    def total_experiences(self, filters: Optional[Dict] = None) -> int:
        return self.storage.count(filters=filters)

    @staticmethod
    def run_viewer(
        model_path: str, db_url: str, table_name: str, schema_type: str, port: int
    ) -> None:
        """Start the Streamlit viewer.

        Args:
            model_path (str): Path to the tokenizer/model directory.
            db_url (str): Database URL for the experience database.
            table_name (str): Name of the experience table in the database.
            schema_type (str): Schema type of the experience table.
            port (int): Port number to run the Streamlit app on.
        """

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
            "--db-url",
            db_url,
            "--table",
            table_name,
            "--schema",
            schema_type,
            "--tokenizer",
            model_path,
        ]
        sys.exit(cli.main())


st.set_page_config(page_title="Trinity-RFT Experience Visualizer", layout="wide")


def get_color_for_action_mask(action_mask_value: int) -> str:
    if action_mask_value == 1:
        return "#c8e6c9"
    else:
        return "#ffcdd2"


def render_token_detail_html(html: str) -> None:
    st.html(html)


def render_experience(exp: Experience, tokenizer: Any) -> None:
    """Render a single experience in Streamlit."""
    token_view = build_experience_token_view(exp, tokenizer)

    def html_escape(text):
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;")
        )

    st.markdown("---")

    # Header with EID
    st.subheader(f"Experience [{exp.eid}]")

    # Reward and metadata first (before prompt/response)
    col_reward, col_metrics, col_info = st.columns(3)
    with col_reward:
        reward_val = exp.reward if exp.reward is not None else 0.0
        st.markdown("**Reward**")
        st.markdown(f"`{reward_val:.4f}`")
    with col_metrics:
        st.markdown("**Metrics**")
        st.json(exp.metrics or {}, expanded=True)
    with col_info:
        st.markdown("**Info**")
        st.json(exp.info or {}, expanded=False)

    # Prompt (collapsed by default)
    with st.expander("Prompt", expanded=False):
        st.code(token_view.prompt_text, language=None, wrap_lines=True, line_numbers=True)

    # Response (collapsed by default)
    with st.expander("Response", expanded=False):
        st.code(token_view.response_text, language=None, wrap_lines=True, line_numbers=True)

    # Response Tokens Detail (collapsed by default)
    with st.expander("Response Tokens Detail", expanded=False):
        html = """
        <style>
            .token-detail-root * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            .token-detail-root {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
                padding: 10px;
            }
            .token-detail-root .token-container {
                display: flex;
                flex-wrap: wrap;
                gap: 4px;
                padding: 12px;
                background-color: #fafafa;
                border-radius: 6px;
            }
            .token-detail-root .token-box {
                display: inline-flex;
                flex-direction: column;
                align-items: center;
                padding: 6px 10px;
                border-radius: 4px;
                border: 1px solid #e0e0e0;
                min-width: 50px;
                transition: transform 0.2s, box-shadow 0.2s;
            }
            .token-detail-root .token-box:hover {
                transform: scale(1.5);
                box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                z-index: 10;
            }
            .token-detail-root .token-text {
                font-family: 'Courier New', monospace;
                font-size: 13px;
                font-weight: 600;
                margin-bottom: 3px;
                text-align: center;
                word-break: break-all;
                max-width: 90px;
            }
            .token-detail-root .token-logprob {
                font-size: 10px;
                color: #666;
                font-family: 'Courier New', monospace;
                text-align: center;
            }
        </style>
        <div class="token-detail-root">
            <div class="token-container">
        """

        for token in token_view.response_tokens:
            bg_color = get_color_for_action_mask(int(token.is_action))
            token_display = token.token_text.replace(" ", "␣").replace("\n", "↵").replace("\t", "⇥")
            token_display = html_escape(token_display)
            logprob_text = f"{token.logprob:.4f}" if token.logprob is not None else "N/A"

            html += f"""
                    <div class="token-box" style="background-color: {bg_color};">
                        <div class="token-text">{token_display}</div>
                        <div class="token-logprob">{logprob_text}</div>
                    </div>
            """
        html += """
            </div>
        </div>
        """
        render_token_detail_html(html)


def parse_args():
    parser = argparse.ArgumentParser(description="Experience Visualizer")
    parser.add_argument("--db-url", type=str, help="Path to the experience database.")
    parser.add_argument("--table", type=str, help="Name of the experience table.")
    parser.add_argument(
        "--schema",
        type=str,
        default="experience",
        choices=("experience", "sft"),
        help="Schema type of the experience table.",
    )
    parser.add_argument("--tokenizer", type=str, required=True, help="Path to the tokenizer.")
    return parser.parse_args()


@st.cache_resource
def get_viewer(db_url: str, table_name: str, schema_type: str) -> SQLExperienceViewer:
    config = StorageConfig()
    config.name = table_name
    config.path = db_url
    config.schema_type = schema_type
    config.storage_type = "sql"
    config.wrap_in_ray = False
    return SQLExperienceViewer(config)


def main():  # noqa: [C901]
    args = parse_args()

    viewer = get_viewer(args.db_url, args.table, args.schema)

    st.title("Trinity-RFT Experience Visualizer")

    # Initialize session state
    if "page" not in st.session_state:
        st.session_state.page = 1

    # === Sidebar: Filters ===
    st.sidebar.header("Filters")

    # Reward range filter
    st.sidebar.markdown("**Reward Range**")
    col_rmin, col_rmax = st.sidebar.columns(2)
    with col_rmin:
        reward_min_str = st.text_input("Min", value="", key="reward_min")
    with col_rmax:
        reward_max_str = st.text_input("Max", value="", key="reward_max")
    reward_min = float(reward_min_str) if reward_min_str.strip() else None
    reward_max = float(reward_max_str) if reward_max_str.strip() else None

    # Model version range filter
    st.sidebar.markdown("**Model Version Range**")
    col_vmin, col_vmax = st.sidebar.columns(2)
    with col_vmin:
        mv_min_str = st.text_input("Min", value="", key="mv_min")
    with col_vmax:
        mv_max_str = st.text_input("Max", value="", key="mv_max")
    model_version_min = int(mv_min_str) if mv_min_str.strip() else None
    model_version_max = int(mv_max_str) if mv_max_str.strip() else None

    # Task ID exact match filter
    task_id_filter = st.sidebar.text_input("Task ID (exact match)", value="", key="task_id")

    # Apply filters button
    if st.sidebar.button("Apply Filters", use_container_width=True):
        new_filters: Dict = {}
        if reward_min is not None:
            new_filters["reward_min"] = reward_min
        if reward_max is not None:
            new_filters["reward_max"] = reward_max
        if model_version_min is not None:
            new_filters["model_version_min"] = int(model_version_min)
        if model_version_max is not None:
            new_filters["model_version_max"] = int(model_version_max)
        if task_id_filter:
            new_filters["task_id"] = task_id_filter
        st.session_state.active_filters = new_filters
        st.session_state.page = 1
        st.rerun()

    # Use committed filters from session state
    if "active_filters" not in st.session_state:
        st.session_state.active_filters = {}
    filters: Dict = st.session_state.active_filters

    # Sidebar bottom: per-page setting (low-profile)
    st.sidebar.markdown("---")
    experiences_per_page = st.sidebar.number_input(
        "Per page", min_value=1, max_value=50, value=10, step=1, key="per_page"
    )

    # Query total with filters
    total_seq_num = viewer.total_experiences(filters=filters or None)
    total_pages = max(1, (total_seq_num + experiences_per_page - 1) // experiences_per_page)

    # Clamp current page
    if st.session_state.page > total_pages:
        st.session_state.page = total_pages

    # Calculate offset and fetch
    offset = (st.session_state.page - 1) * experiences_per_page
    experiences = viewer.get_experiences(offset, experiences_per_page, filters=filters or None)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # Sidebar: table of contents
    if experiences:
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Contents**")
        for exp in experiences:
            eid_str = str(exp.eid)
            reward_str = f"{exp.reward:.2f}" if exp.reward is not None else "N/A"
            st.sidebar.markdown(
                f"- [{eid_str}](#experience-{eid_str}) (r={reward_str})",
                unsafe_allow_html=True,
            )

    # Render experiences
    if experiences:
        for exp in experiences:
            st.markdown(f'<a name="experience-{exp.eid}"></a>', unsafe_allow_html=True)
            render_experience(exp, tokenizer)
    else:
        st.info("No experiences found matching the current filters.")

    # === Bottom: Pagination ===
    st.markdown("---")

    # Row 1: Previous | [current_page] / total_pages | Next
    col_prev, col_page_input, col_slash, col_total, col_next = st.columns([1, 1, 0.3, 0.7, 1])
    with col_prev:
        if st.button("Previous", disabled=(st.session_state.page <= 1)):
            st.session_state.page -= 1
            st.rerun()
    with col_page_input:
        new_page = st.number_input(
            "page",
            min_value=1,
            max_value=total_pages,
            value=st.session_state.page,
            step=1,
            label_visibility="collapsed",
            key="page_input",
        )
        if new_page != st.session_state.page:
            st.session_state.page = new_page
            st.rerun()
    with col_slash:
        st.markdown(
            "<div style='text-align:center;line-height:38px;'>/</div>",
            unsafe_allow_html=True,
        )
    with col_total:
        st.markdown(
            f"<div style='line-height:38px;'>{total_pages}</div>",
            unsafe_allow_html=True,
        )
    with col_next:
        if st.button("Next", disabled=(st.session_state.page >= total_pages)):
            st.session_state.page += 1
            st.rerun()

    # Row 2: total count
    st.caption(f"{total_seq_num} experiences in total")


if __name__ == "__main__":
    main()

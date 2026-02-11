import argparse
import sys
from pathlib import Path
from typing import List

import streamlit as st
import streamlit.components.v1 as components
from sqlalchemy.orm import sessionmaker
from transformers import AutoTokenizer

from trinity.buffer.schema import init_engine
from trinity.common.config import StorageConfig
from trinity.common.experience import Experience
from trinity.utils.log import get_logger


class SQLExperienceViewer:
    def __init__(self, config: StorageConfig) -> None:
        self.logger = get_logger(f"sql_{config.name}", in_ray_actor=True)
        if not config.path:
            raise ValueError("`path` is required for SQL storage type.")
        self.engine, self.table_model_cls = init_engine(
            db_url=config.path,
            table_name=config.name,
            schema_type=config.schema_type,
        )
        self.session = sessionmaker(bind=self.engine)

    def get_experiences(self, offset: int, limit: int = 10) -> List[Experience]:
        self.logger.info(f"Viewing experiences from offset {offset} with limit {limit}.")
        with self.session() as session:
            query = session.query(self.table_model_cls).offset(offset).limit(limit)
            results = query.all()
            exps = [self.table_model_cls.to_experience(row) for row in results]
        return exps

    def total_experiences(self) -> int:
        with self.session() as session:
            count = session.query(self.table_model_cls).count()
        return count

    @staticmethod
    def run_viewer(model_path: str, db_url: str, table_name: str, port: int):
        """Start the Streamlit viewer.

        Args:
            model_path (str): Path to the tokenizer/model directory.
            db_url (str): Database URL for the experience database.
            table_name (str): Name of the experience table in the database.
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
            "--tokenizer",
            model_path,
        ]
        sys.exit(cli.main())


st.set_page_config(page_title="Trinity-RFT Experience Visualizer", layout="wide")


def get_color_for_action_mask(action_mask_value: int) -> str:
    """Return color based on action_mask value"""
    if action_mask_value == 1:
        return "#c8e6c9"
    else:
        return "#ffcdd2"


def render_experience(exp: Experience, tokenizer):
    """Render a single experience sequence in Streamlit."""
    token_ids = exp.tokens
    logprobs = exp.logprobs
    action_mask = exp.action_mask

    prompt_length = exp.prompt_length

    prompt_token_ids = token_ids[:prompt_length]  # type: ignore [index]
    response_token_ids = token_ids[prompt_length:]  # type: ignore [index]

    # Decode tokens
    prompt_text = (
        tokenizer.decode(prompt_token_ids)
        if hasattr(tokenizer, "decode")
        else "".join([str(tid) for tid in prompt_token_ids])
    )
    response_text = (
        tokenizer.decode(response_token_ids)
        if hasattr(tokenizer, "decode")
        else "".join([str(tid) for tid in response_token_ids])
    )

    # Get each response token text
    response_tokens = []
    for tid in response_token_ids:
        if hasattr(tokenizer, "decode"):
            token_text = tokenizer.decode([tid])
        else:
            token_text = f"[{tid}]"
        response_tokens.append(token_text)

    # HTML escape function
    def html_escape(text):
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;")
        )

    # === Use Streamlit Native Components for Prompt and Response ===
    st.subheader(f"Experience [{exp.eid}]")

    # Prompt section using st.text_area
    st.markdown("**📝 Prompt:**")
    st.code(prompt_text, language=None, wrap_lines=True, line_numbers=True)

    # Response section using st.text_area
    st.markdown("**💬 Response:**")
    st.code(response_text, language=None, wrap_lines=True, line_numbers=True)

    # Reward and other info
    st.markdown("**🏆 Reward and Other Info:**")
    reward, info, metrics = st.columns(3)
    reward.metric("**Reward:**", f"{exp.reward or 0.0:.4f}")
    metrics.markdown("**Metrics:**")
    metrics.json(exp.metrics or {}, expanded=False)
    info.markdown("**Info:**")
    info.json(exp.info or {}, expanded=False)

    # Response Tokens Detail section using components.html
    st.markdown("**🔍 Response Tokens Detail:**")

    # Build HTML only for Response Tokens Detail
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }

            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
                padding: 10px;
            }

            .token-container {
                display: flex;
                flex-wrap: wrap;
                gap: 5px;
                padding: 15px;
                background-color: white;
                border-radius: 5px;
            }

            .token-box {
                display: inline-flex;
                flex-direction: column;
                align-items: center;
                padding: 8px 12px;
                border-radius: 5px;
                border: 1px solid #ddd;
                min-width: 60px;
                transition: transform 0.2s, box-shadow 0.2s;
            }

            .token-box:hover {
                transform: scale(1.5);
                box-shadow: 0 4px 8px rgba(0,0,0,0.2);
                z-index: 10;
            }

            .token-text {
                font-family: 'Courier New', monospace;
                font-size: 14px;
                font-weight: bold;
                margin-bottom: 5px;
                text-align: center;
                word-break: break-all;
                max-width: 100px;
            }

            .token-logprob {
                font-size: 11px;
                color: #555;
                font-family: 'Courier New', monospace;
                text-align: center;
            }
        </style>
    </head>
    <body>
        <div class="token-container">
    """

    # Add each response token
    for i, (token_text, logprob, mask) in enumerate(zip(response_tokens, logprobs, action_mask)):  # type: ignore [arg-type]
        bg_color = get_color_for_action_mask(mask)

        # Handle special character display
        token_display = token_text.replace(" ", "␣").replace("\n", "↵").replace("\t", "⇥")
        token_display = html_escape(token_display)

        html += f"""
                <div class="token-box" style="background-color: {bg_color};">
                    <div class="token-text">{token_display}</div>
                    <div class="token-logprob">{logprob:.4f}</div>
                </div>
        """
    html += """
        </div>
    </body>
    </html>
    """

    # Use components.html for token details only
    components.html(html, height=200, scrolling=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Experience Visualizer")
    parser.add_argument(
        "--db-url",
        type=str,
        help="Path to the experience database.",
    )
    parser.add_argument(
        "--table",
        type=str,
        help="Name of the experience table.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        help="Path to the tokenizer.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Initialize SQLExperienceViewer
    config = StorageConfig(
        name=args.table,
        path=args.db_url,
        schema_type="experience",
        storage_type="sql",
    )
    viewer = SQLExperienceViewer(config)

    st.title("🎯 Trinity-RFT Experience Visualizer")
    if "page" not in st.session_state:
        st.session_state.page = 1

    # Add instructions
    with st.expander("ℹ️ Instructions"):
        st.markdown(
            """
        - **Green background**: action_mask = 1
        - **Red background**: action_mask = 0
        - **Top**: Token text (special characters: space=␣, newline=↵, tab=⇥)
        - **Bottom**: Logprob value of the token
        - Hover over token to zoom in
        """
        )

    # Get total sequence number
    total_seq_num = viewer.total_experiences()

    # Sidebar configuration
    st.sidebar.header("⚙️ Settings")

    # Pagination settings
    experiences_per_page = st.sidebar.slider(
        "Experiences per page", min_value=1, max_value=20, value=5
    )

    # Calculate total pages
    total_pages = (total_seq_num + experiences_per_page - 1) // experiences_per_page

    # Page selection (sidebar)
    current_page = st.sidebar.number_input(
        "Select page",
        min_value=1,
        max_value=max(1, total_pages),
        step=1,
        value=st.session_state.page,
    )
    if current_page != st.session_state.page:
        st.session_state.page = current_page
        st.rerun()

    # Show statistics
    st.sidebar.markdown("---")
    st.sidebar.metric("Total experiences", total_seq_num)
    st.sidebar.metric("Total pages", total_pages)
    st.sidebar.metric("Current page", f"{st.session_state.page}/{total_pages}")

    # Calculate offset
    offset = (st.session_state.page - 1) * experiences_per_page

    # Get experiences for current page
    experiences = viewer.get_experiences(offset, experiences_per_page)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # Generate catalog in sidebar
    exp_catalog = []  # [(eid, subheader_text)]
    if experiences:
        for exp in experiences:
            exp_catalog.append(exp.eid)

    if exp_catalog:
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Contents**")
        catalog_md = "\n".join([f"- [ {eid} ](#exp-{eid})" for eid in exp_catalog])
        st.sidebar.markdown(catalog_md, unsafe_allow_html=True)

    if experiences:
        for exp in experiences:
            st.markdown(f'<a name="exp-{exp.eid}"></a>', unsafe_allow_html=True)
            render_experience(exp, tokenizer)
    else:
        st.warning("No experience data found")

    # Pagination navigation
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        if st.session_state.page > 1:
            if st.button("⬅️ Previous Page"):
                st.session_state.page = st.session_state.page - 1
                st.rerun()

    with col2:
        st.markdown(
            f"<center>Page {st.session_state.page} / {total_pages}</center>", unsafe_allow_html=True
        )
    with col3:
        if st.session_state.page < total_pages:
            if st.button("Next Page ➡️"):
                st.session_state.page = st.session_state.page + 1
                st.rerun()


if __name__ == "__main__":
    main()

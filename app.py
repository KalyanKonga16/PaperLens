import json
import os
import shutil
import streamlit as st
import pandas as pd

from src.config import settings
from src.keyword_pipeline import process_pdf
from src.visuals import (
    create_weighted_word_cloud,
    create_category_donut,
    create_radial_evidence_chart,
    create_top_keywords_bar,
    create_keyword_metrics_table
)

st.set_page_config(
    page_title="PaperLens",
    page_icon="🔬",
    layout="wide"
)

# --- Sidebar (Clean, User-Focused) ---
with st.sidebar:
    st.markdown("## 🔬 PaperLens")
    st.markdown(
        "AI-powered keyword extraction from any PDF. "
        "Upload, analyze, and discover what matters most."
    )

    st.markdown("---")

    num_keywords = st.slider(
        "How many keywords do you need?",
        min_value=5,
        max_value=50,
        value=20,
        step=1,
        help="Choose how many keywords you want the system to extract from your document."
    )

    st.markdown("---")

    st.markdown("#### How It Works")
    st.markdown(
        """
        1. Upload any PDF document
        2. Choose the number of keywords
        3. Click **Extract Keywords**
        4. View results, charts, and download data
        """
    )

    st.markdown("---")
    st.caption("Powered by Page-Aware Retrieval + LLM Refinement")
    st.markdown("---")
    st.markdown("👨‍💻 **Built by:** [Kalyan Konga](https://github.com/KalyanKonga16)")

# --- Main Page ---
st.title("🔬 PaperLens")
st.markdown("### Look deeper into any document")
st.markdown(
    "Upload any PDF and instantly extract the most relevant keywords with AI-powered analysis."
)

uploaded_file = st.file_uploader("Upload your document (PDF)", type=["pdf"])

if uploaded_file is not None:
    st.markdown(f"**Selected:** {uploaded_file.name}")

    if st.button("🚀 Extract Keywords", type="primary", use_container_width=True):
        st.session_state['last_file'] = uploaded_file.name
        st.session_state['last_num_keywords'] = num_keywords
        pdf_bytes = uploaded_file.read()

        try:
            with st.spinner(f"Analyzing your document and extracting {num_keywords} keywords..."):
                st.session_state.result = process_pdf(
                    pdf_bytes=pdf_bytes,
                    filename=uploaded_file.name,
                    use_cache=True,
                    max_keywords=num_keywords
                )
            st.success("✅ Analysis complete!")
        except Exception as e:
            st.error(f"Could not process this PDF. Please try a different file. Error: {e}")
            st.session_state.result = None

# --- Display Results ---
if 'result' in st.session_state and st.session_state.result:
    result = st.session_state.result
    all_keywords = result.get("keywords", [])
    keyword_metrics = result.get("keyword_metrics", [])
    total_kw = len(all_keywords)

    if not all_keywords:
        st.warning("No keywords could be extracted. The document might be empty, scanned, or image-based.")
    else:
        st.markdown("---")

        tab1, tab2, tab3 = st.tabs([
            "📄 Keywords & Summary",
            "📊 Visual Insights",
            "ℹ️ About PaperLens"
        ])

        # --- Tab 1: Keywords & Summary ---
        with tab1:
            st.subheader("Document Summary")
            summary_text = result.get("summary") or "Summary not available for this document."
            st.info(summary_text)

            st.subheader(f"Extracted Keywords ({total_kw})")
            df = create_keyword_metrics_table(keyword_metrics)
            st.dataframe(df, use_container_width=True, height=420)

            st.markdown("---")
            st.subheader("📥 Download Your Results")

            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.download_button(
                    label="⬇️ Download Keywords (CSV)",
                    data=df.to_csv(index=True),
                    file_name=f"{result['file_name'].rsplit('.', 1)[0]}_paperlens_keywords.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            with col_d2:
                st.download_button(
                    label="⬇️ Download Full Report (JSON)",
                    data=json.dumps(result, indent=2, ensure_ascii=False),
                    file_name=f"{result['file_name'].rsplit('.', 1)[0]}_paperlens_report.json",
                    mime="application/json",
                    use_container_width=True
                )

        # --- Tab 2: Visual Insights ---
        with tab2:
            st.subheader("Visual Keyword Analysis")
            st.caption(f"Analyzing {total_kw} keywords based on real document evidence.")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Word Cloud")
                st.caption("Bigger words appear more prominently in the document.")
                fig_wc = create_weighted_word_cloud(keyword_metrics)
                if fig_wc:
                    st.pyplot(fig_wc, use_container_width=True)

                st.markdown("#### Evidence Radar")
                st.caption("Circular view of keyword importance based on document evidence.")
                fig_radial = create_radial_evidence_chart(keyword_metrics)
                if fig_radial:
                    st.plotly_chart(fig_radial, use_container_width=True)

            with col2:
                st.markdown("#### Topic Distribution")
                st.caption("What technical areas does this document focus on?")
                fig_donut = create_category_donut(keyword_metrics)
                if fig_donut:
                    st.plotly_chart(fig_donut, use_container_width=True)

                st.markdown("#### Strongest Keywords")
                st.caption("Top 10 keywords ranked by how strongly the document supports them.")
                fig_top = create_top_keywords_bar(keyword_metrics, top_n=10)
                if fig_top:
                    st.plotly_chart(fig_top, use_container_width=True)

        # --- Tab 3: About PaperLens ---
        with tab3:
            st.subheader("About PaperLens")

            st.markdown(
                """
                **PaperLens** is an AI-powered tool that helps researchers, students,
                and professionals look deeper into any PDF document and instantly identify
                the most important keywords.

                #### What Makes PaperLens Different?

                Unlike simple keyword counters, PaperLens uses a **3-stage intelligent pipeline**:

                **Stage 1 — Page-Aware Indexing**
                The PDF is split into page-level chunks, preserving the document's
                natural structure. This prevents mixing unrelated sections together,
                which is a common problem in traditional text analysis.

                **Stage 2 — AI-Powered Refinement**
                A large language model analyzes the most relevant sections and selects
                keywords that are meaningful — not just frequently repeated.

                **Stage 3 — Evidence Scoring**
                Every keyword receives an evidence score based on:
                - How often it appears in the document
                - How many pages mention it
                - Whether it appears in the most relevant sections

                #### Who Is PaperLens For?

                - **Researchers** — Quickly understand the main themes of a paper
                - **Students** — Identify key concepts for literature reviews
                - **Professionals** — Analyze technical reports and whitepapers
                - **Business Analysts** — Extract insights from contracts and reports
                - **Recruiters / Managers** — Understand technical documents at a glance

                #### Technical Foundation

                - **Retrieval:** Page-aware semantic indexing
                - **AI Model:** Cloud-hosted LLM with automatic fallback
                - **Visualizations:** Evidence-based (not synthetic rankings)
                - **Privacy:** Your PDF is processed locally and never stored permanently
                """
            )

            st.markdown("---")
            st.markdown("#### 👨‍💻 Built By")
            st.markdown(
                """
                **Kalyan Konga**  
                GitHub: [@KalyanKonga16](https://github.com/KalyanKonga16)  
                Live Demo: [paperlensforu.streamlit.app](https://paperlensforu.streamlit.app/)
                """
            )

            st.markdown("---")
            st.caption("PaperLens • Built with Streamlit • Python • Hugging Face • Page-Aware Retrieval")

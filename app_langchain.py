import streamlit as st
from html import escape

from src.answer_langchain import generate_answer


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AgriLens | Agriculture RAG",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =========================================================
# CUSTOM THEME
# =========================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

    :root {
        --bg: #101719;
        --panel: #192426;
        --panel2: #202d30;
        --border: #344448;
        --text: #edf3ef;
        --muted: #b0bfba;
        --green: #a4d481;
        --amber: #e4b16c;
    }

    .stApp {
        background:
            radial-gradient(
                circle at 85% 0%,
                rgba(77, 119, 83, 0.13),
                transparent 35%
            ),
            var(--bg);
        color: var(--text);
        font-family: 'DM Sans', sans-serif;
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }

    #MainMenu, footer {
        visibility: hidden;
    }

    .block-container {
        max-width: 1120px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Brand */

    .brand-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
        border-bottom: 1px solid var(--border);
        padding-bottom: 18px;
        margin-bottom: 30px;
    }

    .brand-left {
        display: flex;
        align-items: center;
        gap: 13px;
    }

    .brand-icon {
        width: 52px;
        height: 52px;
        flex-shrink: 0;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 14px;
        background: linear-gradient(145deg, #b7e69a, #527d59);
        font-size: 26px;
    }

    .brand-title {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 27px !important;
        font-weight: 700 !important;
        line-height: 1.25;
        letter-spacing: .4px;
        opacity: 1 !important;
        margin: 0;
    }

    .brand-subtitle {
        color: #b8c9c2 !important;
        -webkit-text-fill-color: #b8c9c2 !important;
        font-size: 10px;
        font-weight: 600;
        letter-spacing: 1.5px;
        margin-top: 5px;
    }

    .status-pill {
        color: var(--green);
        background: #1b2921;
        border: 1px solid #3c5b43;
        border-radius: 20px;
        padding: 8px 12px;
        font-size: 10px;
        font-weight: 700;
        white-space: nowrap;
    }

    /* Hero */

    .eyebrow {
        color: var(--amber);
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 2px;
        margin-bottom: 10px;
    }

    .hero-title {
        color: var(--text) !important;
        font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(25px, 3vw, 36px);
        font-weight: 700;
        line-height: 1.25;
        letter-spacing: -.7px;
        margin-bottom: 12px;
    }

    .hero-description {
        color: var(--muted);
        font-size: 14px;
        line-height: 1.8;
        max-width: 800px;
        margin-bottom: 27px;
    }

    /* Question heading */

    .question-heading {
        background: linear-gradient(120deg, #1d2d2d, #192426);
        border: 1px solid var(--border);
        border-radius: 15px;
        padding: 19px 21px;
        margin: 0 0 18px;
    }

    .question-kicker {
        color: var(--green);
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 1.7px;
        margin-bottom: 8px;
    }

    .question-title {
        color: var(--text);
        font-family: 'Space Grotesk', sans-serif;
        font-size: 20px;
        font-weight: 600;
        margin-bottom: 6px;
    }

    .question-description {
        color: var(--muted);
        font-size: 12px;
        line-height: 1.6;
    }

    /* Inputs */

    div[data-testid="stTextArea"] textarea {
        background: #121a1c !important;
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        border: 1px solid #3b4c50 !important;
        border-radius: 11px !important;
        font-size: 14px;
        line-height: 1.6;
    }

    div[data-testid="stTextArea"] textarea::placeholder {
        color: #b0bfba !important;
        -webkit-text-fill-color: #b0bfba !important;
        opacity: 1 !important;
    }

    div[data-testid="stSelectbox"] > div > div {
        background: #121a1c !important;
        border: 1px solid #3b4c50 !important;
        border-radius: 10px !important;
        color: #ffffff !important;
    }

    div[data-testid="stSelectbox"] span {
        color: #ffffff !important;
    }

    div.stButton > button[kind="primary"] {
        background: var(--green) !important;
        color: #152019 !important;
        border: 1px solid var(--green) !important;
        border-radius: 10px !important;
        min-height: 44px;
        font-weight: 700;
    }

    div.stButton > button[kind="primary"]:hover {
        background: #bce5a0 !important;
        border-color: #bce5a0 !important;
    }

    /* Results */

    .section-kicker {
        color: var(--amber);
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 1.7px;
        margin-bottom: 6px;
    }

    .section-title {
        color: var(--text);
        font-family: 'Space Grotesk', sans-serif;
        font-size: 24px;
        font-weight: 600;
        margin-bottom: 5px;
    }

    .section-description {
        color: var(--muted);
        font-size: 12px;
        line-height: 1.7;
        margin-bottom: 16px;
    }

    .answer-card {
        background: var(--panel);
        border: 1px solid var(--border);
        border-left: 4px solid var(--green);
        border-radius: 13px;
        padding: 18px 21px;
        margin: 10px 0 25px;
    }

    .answer-label {
        display: inline-block;
        color: var(--green);
        background: #233428;
        border: 1px solid #3c5b43;
        border-radius: 5px;
        padding: 5px 8px;
        font-size: 9px;
        font-weight: 700;
        letter-spacing: 1px;
        margin-bottom: 10px;
    }

    .source-card {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 13px 15px;
        margin: 8px 0;
        overflow-wrap: anywhere;
    }

    .source-title {
        color: var(--text);
        font-size: 13px;
        font-weight: 600;
    }

    .source-meta {
        color: var(--muted);
        font-size: 10px;
        margin-top: 5px;
    }

    .stat-card {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 11px;
        padding: 14px 16px;
        margin: 8px 0 17px;
    }

    .stat-label {
        color: var(--muted);
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
    }

    .stat-number {
        color: var(--green);
        font-family: 'Space Grotesk', sans-serif;
        font-size: 25px;
        font-weight: 700;
        margin-top: 5px;
    }

    div[data-testid="stExpander"] {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 10px;
        margin-bottom: 9px;
    }

    div[data-testid="stExpander"] summary {
        color: var(--text) !important;
        font-weight: 600;
    }

    .footer-note {
        border-top: 1px solid var(--border);
        padding-top: 14px;
        margin-top: 35px;
        color: var(--muted);
        font-size: 10px;
        line-height: 1.8;
    }

    @media (max-width: 650px) {
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .brand-title {
            font-size: 23px !important;
        }

        .status-pill {
            display: none;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# HEADER
# Keep each HTML block self-contained.
# =========================================================

st.markdown(
    """
    <div class="brand-row">
        <div class="brand-left">
            <div class="brand-icon">🌿</div>
            <div>
                <div class="brand-title">AgriLens</div>
                <div class="brand-subtitle">
                    AGRICULTURAL KNOWLEDGE SYSTEM
                </div>
            </div>
        </div>
        <div class="status-pill">
            ● KNOWLEDGE-BASED ASSISTANT
        </div>
    </div>

    <div class="eyebrow">ASK · RETRIEVE · UNDERSTAND</div>

    <div class="hero-title">
        Find the evidence behind better farming decisions.
    </div>

    <div class="hero-description">
        Ask an agriculture question and explore guidance drawn from your
        indexed knowledge base. Read the response separately from the
        retrieved passages, so you can inspect the supporting evidence.
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# QUESTION FORM
# =========================================================

st.markdown(
    """
    <div class="question-heading">
        <div class="question-kicker">NEW ENQUIRY</div>
        <div class="question-title">What would you like to know?</div>
        <div class="question-description">
            Choose a suggested question or describe your agricultural concern.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

examples = [
    "Choose an example...",
    "How can I improve soil health before sowing?",
    "What information is needed to identify a crop disease?",
    "How should I plan irrigation efficiently?",
    "What are the main steps of integrated pest management?",
    "How can I choose a crop for my field?",
]

if "agrilens_question" not in st.session_state:
    st.session_state.agrilens_question = ""

selected = st.selectbox(
    "Suggested questions",
    examples,
    key="agrilens_example",
    label_visibility="collapsed",
)

# Only replace the text when the user chooses a real example.
if selected != "Choose an example...":
    st.session_state.agrilens_question = selected

question = st.text_area(
    "Your agriculture question",
    key="agrilens_question",
    placeholder=(
        "Describe the crop, symptoms, soil or farming practice "
        "you need help with..."
    ),
    height=110,
    label_visibility="collapsed",
)

button_col, help_col = st.columns([1.25, 3.75])

with button_col:
    ask = st.button(
        "Search knowledge  ↗",
        type="primary",
        use_container_width=True,
    )

with help_col:
    st.markdown(
        """
        <div style="
            color:#b0bfba;
            font-size:11px;
            padding-top:13px;
        ">
            Answers are generated using your indexed agricultural documents.
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# RUN THE EXISTING RAG PIPELINE
# =========================================================

if ask:
    if not question.strip():
        st.warning("Please enter an agriculture question.")

    else:
        try:
            with st.spinner(
                "Searching agricultural documents and preparing guidance..."
            ):
                answer, results = generate_answer(question.strip())

            st.session_state["agrilens_last_question"] = question.strip()
            st.session_state["agrilens_last_answer"] = answer
            st.session_state["agrilens_last_results"] = results

        except Exception as exc:
            st.error(f"Could not generate guidance: {exc}")
            st.info(
                "Check that Ollama is running, the required model is available, "
                "and your retrieval index can be accessed."
            )


# =========================================================
# DISPLAY THE GENERATED ANSWER
# =========================================================

if "agrilens_last_answer" in st.session_state:

    answer = st.session_state["agrilens_last_answer"]
    results = st.session_state["agrilens_last_results"]
    last_question = st.session_state["agrilens_last_question"]

    st.markdown("---")

    st.markdown(
        """
        <div class="section-kicker">01 / GENERATED RESPONSE</div>
        <div class="section-title">Agricultural guidance</div>
        <div class="section-description">
            Guidance generated from the question and the retrieved evidence.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div style="
            color:#b0bfba;
            font-size:12px;
            margin-bottom:12px;
        ">
            <strong>QUESTION:</strong> {escape(last_question)}
        </div>

        <div class="answer-card">
            <div class="answer-label">EVIDENCE-INFORMED RESPONSE</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Display Markdown from the answer generator.
    # Do not open an HTML container around this Streamlit element.
    st.markdown(answer)

    # -----------------------------------------------------
    # SUPPORTING SOURCES
    # -----------------------------------------------------

    st.markdown("---")

    st.markdown(
        """
        <div class="section-kicker">02 / KNOWLEDGE TRAIL</div>
        <div class="section-title">Supporting sources</div>
        <div class="section-description">
            Explore the source documents and inspect the passages returned
            by the retrieval pipeline.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not results:
        st.info("No relevant documents were retrieved.")

    else:
        unique_sources = []
        seen_sources = set()

        for item in results:
            source = str(
                item.get("path")
                or item.get("source")
                or "Unknown source"
            )

            if source not in seen_sources:
                seen_sources.add(source)
                unique_sources.append((source, item.get("score")))

        stat1, stat2 = st.columns(2)

        with stat1:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-label">Retrieved passages</div>
                    <div class="stat-number">{len(results)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with stat2:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-label">Unique sources</div>
                    <div class="stat-number">{len(unique_sources)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Show each unique source as a separate card.
        for index, (source, score) in enumerate(unique_sources, start=1):

            score_text = (
                f" · Score: {escape(str(score))}"
                if score is not None
                else ""
            )

            st.markdown(
                f"""
                <div class="source-card">
                    <div class="source-title">
                        ▤ &nbsp; {escape(source)}
                    </div>
                    <div class="source-meta">
                        SOURCE {index}{score_text}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Expandable evidence area.
        with st.expander(
            "Inspect retrieved evidence passages",
            expanded=False,
        ):
            for index, item in enumerate(results, start=1):

                source = str(
                    item.get("path")
                    or item.get("source")
                    or "Unknown source"
                )

                score = item.get("score")

                heading = f"Passage {index} — {source}"

                if score is not None:
                    heading += f" — Score: {score}"

                document = (
                    item.get("document")
                    or item.get("text")
                    or item.get("content")
                    or "No passage text is available."
                )

                with st.expander(heading, expanded=False):
                    st.text(str(document).strip())


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer-note">
        <strong style="color:#a4d481;">AGRILENS</strong>
        · AGRICULTURAL KNOWLEDGE RETRIEVAL
        <br>
        Use the response as decision support. Verify crop diagnosis,
        pesticide use and location-specific recommendations with current
        local agricultural guidance or a qualified expert.
    </div>
    """,
    unsafe_allow_html=True,
)
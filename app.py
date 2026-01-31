import os
import io
import textwrap
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import requests
from dotenv import load_dotenv

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

load_dotenv()

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(
    page_title="Data Analytics Platform",
    page_icon="■",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------
# Helpers
# ---------------------------
def truncate_filename(name: str, max_len: int = 28) -> str:
    if not name:
        return ""
    return name if len(name) <= max_len else name[: max_len - 1] + "…"

def load_data_file(uploaded_file):
    """Load CSV or Excel file with basic validation."""
    try:
        ext = uploaded_file.name.split(".")[-1].lower()

        if ext == "csv":
            # Handle common encodings a bit more gracefully
            content = uploaded_file.getvalue()
            try:
                df = pd.read_csv(io.BytesIO(content))
            except UnicodeDecodeError:
                df = pd.read_csv(io.BytesIO(content), encoding="latin-1")
        elif ext in ["xlsx", "xls"]:
            df = pd.read_excel(uploaded_file)
        else:
            st.error(f"Unsupported file type: .{ext}")
            return None

        if df is None or df.empty or len(df.columns) == 0:
            st.error("The uploaded file is empty or has no columns.")
            return None

        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

def df_profile_summary(df: pd.DataFrame) -> str:
    """Compact schema summary for the LLM prompt."""
    cols = []
    for c in df.columns:
        dtype = str(df[c].dtype)
        nulls = int(df[c].isna().sum())
        cols.append(f"- {c} ({dtype}), nulls={nulls}")
    return "\n".join(cols)

def df_quick_context(df: pd.DataFrame) -> str:
    """Give the LLM enough context without dumping huge text."""
    head_md = df.head(8).to_markdown(index=False)
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols]

    stats_md = ""
    if len(num_cols) > 0:
        stats_md = df[num_cols].describe().round(3).to_markdown()

    cat_md = ""
    # only a couple categorical columns, top values
    for c in cat_cols[:2]:
        vc = df[c].astype(str).value_counts().head(8)
        cat_md += f"\n\nTop values for `{c}`:\n" + vc.to_frame("count").to_markdown()

    parts = [
        f"Rows: {len(df)}, Columns: {len(df.columns)}",
        "Columns:\n" + df_profile_summary(df),
        "\nPreview (first 8 rows):\n" + head_md,
    ]
    if stats_md:
        parts.append("\nNumeric summary (describe):\n" + stats_md)
    if cat_md:
        parts.append(cat_md)

    return "\n\n".join(parts)

def call_openai_chat_completion(query: str, df: pd.DataFrame, temperature: float = 0.0) -> str:
    """
    Calls OpenAI Chat Completions using raw HTTP (requests),
    avoiding the 'proxies' / httpx client mismatch you hit.
    """
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return "Missing OPENAI_API_KEY. Add it to your .env file (OPENAI_API_KEY=...)."

    # Keep prompt tight but useful
    context = df_quick_context(df)

    system = (
        "You are a senior data analyst. "
        "Answer clearly and practically. "
        "If the user asks for a calculation, explain the method and provide the result. "
        "If a chart is requested, describe what to plot and which columns to use. "
        "Do not hallucinate columns that don't exist."
    )

    user = f"""Dataset context:
{context}

User question:
{query}

Return:
- A direct answer
- If helpful: bullet insights
- If needed: pandas code snippet (ONLY if it uses the columns above)
"""

    # Use a modern small model by default (you can change this)
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    try:
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": float(temperature),
            },
            timeout=60,
        )
        if resp.status_code != 200:
            return f"LLM error ({resp.status_code}): {resp.text}"

        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"LLM request failed: {e}"

# ---------------------------
# Session state
# ---------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "df" not in st.session_state:
    st.session_state.df = None
if "queued_query" not in st.session_state:
    st.session_state.queued_query = None

# ---------------------------
# CSS (professional + stable)
# ---------------------------
SIDEBAR_W = 340  # single source of truth for alignment

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

* {{ font-family: 'Inter', sans-serif; }}

html, body, [class*="css"] {{
    background: #141414 !important;
}}

.main {{
    background: #141414 !important;
}}

.block-container {{
    max-width: 1200px;
    padding-top: 28px;
    padding-bottom: 120px;
}}

h1 {{
    color: #ffffff;
    font-weight: 600;
    font-size: 34px;
    letter-spacing: -0.02em;
    margin-bottom: 6px;
    text-align: center;
}}

hr {{
    border: none;
    border-top: 1px solid #303030;
    margin: 26px 0;
}}

[data-testid="stSidebar"] {{
    width: {SIDEBAR_W}px !important;
    min-width: {SIDEBAR_W}px !important;
    max-width: {SIDEBAR_W}px !important;
    background: #1c1c1c !important;
    border-right: 1px solid #2a2a2a;
    padding: 22px 18px;
}}

[data-testid="stSidebar"] h3 {{
    font-size: 12px !important;
    font-weight: 700 !important;
    letter-spacing: 0.10em !important;
    color: #9a9a9a !important;
    text-transform: uppercase;
    text-align: center;
    margin: 10px 0 16px 0;
}}

.sidebar-card {{
    background: #171717;
    border: 1px solid #2b2b2b;
    border-radius: 14px;
    padding: 16px;
}}

.sidebar-subtext {{
    color: #9a9a9a;
    font-size: 12.5px;
    line-height: 1.5;
    text-align: center;
    margin-top: 10px;
}}

.sidebar-status {{
    background: rgba(34, 197, 94, 0.18);
    border: 1px solid rgba(34, 197, 94, 0.28);
    color: #e8ffe8;
    border-radius: 999px;
    padding: 10px 12px;
    font-size: 13px;
    font-weight: 600;
    display: flex;
    gap: 8px;
    align-items: center;
    justify-content: center;
    margin-top: 14px;
    overflow: hidden;
    white-space: nowrap;
    text-overflow: ellipsis;
}}

.sidebar-actions {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    margin-top: 14px;
}}

div.stButton > button {{
    width: 100%;
    border-radius: 12px !important;
    padding: 12px 12px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    border: 1px solid #2c2c2c !important;
}}

.sidebar-actions div.stButton > button {{
    height: 44px !important;
}}

.sidebar-actions div.stButton > button:first-child {{
    background: #ffffff !important;
    color: #121212 !important;
    border: none !important;
}}

.sidebar-actions div.stButton > button:last-child {{
    background: #262626 !important;
    color: #ffffff !important;
}}

div.stButton > button:hover {{
    filter: brightness(0.95);
}}

[data-testid="stFileUploader"] {{
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}}

[data-testid="stFileUploader"] section {{
    border: none !important;
}}

[data-testid="stFileUploaderDropzone"] {{
    background: #101318 !important;
    border: 1px solid #2b2b2b !important;
    border-radius: 14px !important;
    padding: 18px 14px !important;
}}

[data-testid="stFileUploaderDropzone"] * {{
    text-align: center !important;
}}

[data-testid="stFileUploaderDropzone"] button {{
    margin: 10px auto 0 auto !important;
    display: block !important;
    border-radius: 12px !important;
    padding: 12px 16px !important;
    border: 1px solid #3a3a3a !important;
    background: #232323 !important;
    color: #ffffff !important;
}}

[data-testid="stFileUploaderDropzone"] small {{
    display: none !important; /* hides Streamlit's default "limit 200MB..." line */
}}

[data-testid="stChatInput"] {{
    position: fixed !important;
    bottom: 0 !important;
    left: {SIDEBAR_W}px !important;
    right: 0 !important;
    background: #141414 !important;
    border-top: 1px solid #2a2a2a !important;
    padding: 18px 0 !important;
    z-index: 1000 !important;
}}

[data-testid="stChatInput"] > div {{
    max-width: 1200px !important;
    margin: 0 auto !important;
    padding: 0 18px !important;
}}

[data-testid="stChatInput"] textarea {{
    background: #1f1f1f !important;
    border: 1px solid #2d2d2d !important;
    border-radius: 14px !important;
    color: #ffffff !important;
    padding: 14px 52px 14px 16px !important;
}}

[data-testid="stChatInput"] button {{
    background: transparent !important;
    border: none !important;
    position: absolute !important;
    right: 14px !important;
    top: 50% !important;
    transform: translateY(-50%) !important;
}}

.stChatMessage {{
    background: transparent !important;
    border: 1px solid #252525;
    border-radius: 16px;
    padding: 14px 16px;
    margin: 10px 0;
}}

.stChatMessage p {{
    color: #f2f2f2 !important;
    font-size: 14px;
    line-height: 1.6;
}}

.stTabs [data-baseweb="tab-list"] {{
    gap: 4px;
    border-bottom: 1px solid #2a2a2a;
}}

.stTabs [data-baseweb="tab"] {{
    background: transparent;
    border-radius: 10px;
    padding: 10px 14px;
    font-weight: 600;
    font-size: 13px;
    color: #cfcfcf;
}}

.stTabs [aria-selected="true"] {{
    color: #ffffff !important;
    background: rgba(255,255,255,0.06) !important;
}}

.quick-title {{
    margin-top: 18px;
    margin-bottom: 10px;
    text-align: center;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 0.12em;
    color: #9a9a9a;
    text-transform: uppercase;
}}

.quick-btn div.stButton > button {{
    background: #222;
    color: #fff;
    border: 1px solid #2c2c2c !important;
}}

</style>
""",
    unsafe_allow_html=True
)

# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    st.markdown("<h3>Data Source</h3>", unsafe_allow_html=True)

    st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload CSV or Excel file",
        type=["csv", "xlsx", "xls"],
        label_visibility="collapsed",
    )

    st.markdown(
        '<div class="sidebar-subtext">File size limit: 200MB<br/>Acceptable formats: CSV, XLSX, XLS</div>',
        unsafe_allow_html=True
    )

    # load file
    loaded_name = None
    if uploaded_file is not None:
        loaded_df = load_data_file(uploaded_file)
        if loaded_df is not None:
            st.session_state.df = loaded_df
            loaded_name = uploaded_file.name

    # status pill (clean, truncated)
    if st.session_state.df is not None:
        if loaded_name is None and hasattr(uploaded_file, "name") and uploaded_file is not None:
            loaded_name = uploaded_file.name
        pill_name = truncate_filename(loaded_name or "dataset")
        st.markdown(f'<div class="sidebar-status">Loaded {pill_name}</div>', unsafe_allow_html=True)

    # actions row (no overlap)
    st.markdown('<div class="sidebar-actions">', unsafe_allow_html=True)
    colA, colB = st.columns(2)
    with colA:
        if st.button("Load Sample Data"):
            try:
                st.session_state.df = pd.read_csv("sample_data/sales_data.csv")
                st.session_state.messages = []
                st.session_state.queued_query = None
                st.rerun()
            except Exception:
                st.error("Sample data not found at sample_data/sales_data.csv")
    with colB:
        if st.button("Clear Data"):
            st.session_state.df = None
            st.session_state.messages = []
            st.session_state.queued_query = None
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)  # sidebar-actions

    st.markdown("</div>", unsafe_allow_html=True)  # sidebar-card

    st.markdown("<hr/>", unsafe_allow_html=True)

    with st.expander("Advanced Settings"):
        temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)
        llm_mode = st.toggle("LLM mode", value=True, help="Uses an LLM to answer questions about your dataframe.")

    st.markdown('<div class="quick-title">Quick Queries</div>', unsafe_allow_html=True)

    quick_queries = [
        "Summary statistics",
        "Show top 10 rows",
        "Show column data types",
        "Nulls per column",
        "Correlations (numeric columns)",
        "Group by category and sum sales (if columns exist)",
        "Trend over time (if a date column exists)",
    ]

    for q in quick_queries:
        st.markdown('<div class="quick-btn">', unsafe_allow_html=True)
        if st.button(q, key=f"qq_{q}"):
            st.session_state.queued_query = q
        st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# Main
# ---------------------------
st.title("Data Analytics Platform")
st.markdown(
    "<p style='text-align:center;color:#8a8a8a;margin-top:-6px;'>GPT-powered data analysis</p>",
    unsafe_allow_html=True
)

df = st.session_state.df

if df is None:
    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="text-align:center;padding:70px 10px;color:#9a9a9a;">
            <div style="font-size:16px;font-weight:600;color:#bdbdbd;margin-bottom:6px;">No dataset loaded</div>
            <div style="font-size:13px;">Upload a CSV/XLSX/XLS from the sidebar to begin.</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.stop()

st.markdown("<hr/>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Data", "Analysis", "Charts"])

# ---------------------------
# Data tab (keep it useful)
# ---------------------------
with tab1:
    st.markdown("### Dataset Preview")
    st.dataframe(df.head(50), use_container_width=True, height=520)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Column Information")
        col_info = pd.DataFrame({
            "Type": df.dtypes.astype(str),
            "Non-Null": df.count(),
            "Null %": (df.isna().sum() / len(df) * 100).round(1),
        })
        st.dataframe(col_info, use_container_width=True, height=360)

    with c2:
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if len(num_cols) > 0:
            st.markdown("### Summary Statistics (Numeric)")
            st.dataframe(df[num_cols].describe().round(3), use_container_width=True, height=360)
        else:
            st.info("No numeric columns found to summarize.")

# ---------------------------
# Analysis tab (LLM chat)
# ---------------------------
with tab2:
    # Render history
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # Determine query (from quick query or chat input)
    user_query = st.chat_input("Ask your question here...")

    if st.session_state.queued_query:
        user_query = st.session_state.queued_query
        st.session_state.queued_query = None

    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})

        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                if llm_mode:
                    answer = call_openai_chat_completion(user_query, df, temperature=temperature)
                else:
                    answer = "Turn on **LLM mode** in Advanced Settings to get model answers."

                st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()

# ---------------------------
# Charts tab (safe + generic)
# ---------------------------
with tab3:
    st.markdown("### Data Visualizations")

    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()

    # Histogram for first numeric column
    if len(num_cols) > 0:
        st.markdown(f"#### Distribution: {num_cols[0]}")
        fig = px.histogram(df, x=num_cols[0], nbins=30)
        fig.update_layout(
            plot_bgcolor="#141414",
            paper_bgcolor="#141414",
            font=dict(color="#ffffff"),
            xaxis=dict(gridcolor="#2a2a2a"),
            yaxis=dict(gridcolor="#2a2a2a"),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No numeric columns found for charts.")

    # Bar chart for first categorical vs first numeric (if possible)
    if len(cat_cols) > 0 and len(num_cols) > 0:
        st.markdown(f"#### {num_cols[0]} by {cat_cols[0]} (Top 12)")
        tmp = (
            df.groupby(cat_cols[0])[num_cols[0]]
            .sum()
            .sort_values(ascending=False)
            .head(12)
            .reset_index()
        )
        fig2 = px.bar(tmp, x=cat_cols[0], y=num_cols[0])
        fig2.update_layout(
            plot_bgcolor="#141414",
            paper_bgcolor="#141414",
            font=dict(color="#ffffff"),
            xaxis=dict(gridcolor="#2a2a2a"),
            yaxis=dict(gridcolor="#2a2a2a"),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Line chart if there is a date-like column + numeric
    date_candidate = None
    for c in df.columns:
        if "date" in c.lower() or "time" in c.lower():
            date_candidate = c
            break

    if date_candidate and len(num_cols) > 0:
        try:
            dtmp = df.copy()
            dtmp[date_candidate] = pd.to_datetime(dtmp[date_candidate], errors="coerce")
            dtmp = dtmp.dropna(subset=[date_candidate]).sort_values(date_candidate)
            if not dtmp.empty:
                st.markdown(f"#### Trend: {num_cols[0]} over {date_candidate}")
                fig3 = go.Figure()
                fig3.add_trace(go.Scatter(
                    x=dtmp[date_candidate],
                    y=dtmp[num_cols[0]],
                    mode="lines",
                ))
                fig3.update_layout(
                    plot_bgcolor="#141414",
                    paper_bgcolor="#141414",
                    font=dict(color="#ffffff"),
                    xaxis=dict(gridcolor="#2a2a2a"),
                    yaxis=dict(gridcolor="#2a2a2a"),
                )
                st.plotly_chart(fig3, use_container_width=True)
        except Exception:
            pass

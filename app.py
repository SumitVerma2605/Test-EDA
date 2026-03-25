"""
Statistical Data Analysis Tool - Main Streamlit App
Production-grade modular application for advanced statistical analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64
from utils import (
    load_from_github,
    handle_missing_values,
    detect_outliers_iqr,
    detect_outliers_zscore,
    convert_dtypes,
    summary_statistics,
    compute_correlation,
    run_ttest_independent,
    run_ttest_paired,
    run_anova,
    run_chi_square,
    run_shapiro_wilk,
    run_pearson,
    run_spearman,
    compute_confidence_interval,
    run_linear_regression,
    run_logistic_regression,
    generate_report_csv,
    generate_report_pdf,
)

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="StatLab Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg: #0d0f14;
    --surface: #161922;
    --surface2: #1e2230;
    --accent: #00e5ff;
    --accent2: #7c3aed;
    --text: #e2e8f0;
    --muted: #64748b;
    --success: #10b981;
    --warn: #f59e0b;
    --danger: #ef4444;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg);
    color: var(--text);
}

.stApp { background-color: var(--bg); }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--surface);
    border-right: 1px solid #2d3748;
}

section[data-testid="stSidebar"] .stRadio label {
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    color: var(--muted);
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

section[data-testid="stSidebar"] .stRadio div[data-testid="stMarkdownContainer"] p {
    color: var(--accent) !important;
}

/* Headers */
h1, h2, h3 { font-family: 'Space Mono', monospace; }

/* Metric cards */
[data-testid="stMetric"] {
    background: var(--surface2);
    border: 1px solid #2d3748;
    border-radius: 8px;
    padding: 16px;
}

[data-testid="stMetricValue"] { color: var(--accent) !important; font-family: 'Space Mono', monospace; }

/* Dataframe */
.stDataFrame { border: 1px solid #2d3748; border-radius: 8px; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, var(--accent2), var(--accent));
    color: white;
    border: none;
    border-radius: 6px;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.05em;
    font-weight: 700;
    padding: 10px 20px;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }

/* Expander */
details { background: var(--surface2) !important; border-radius: 8px !important; border: 1px solid #2d3748 !important; }

/* Select/Input */
.stSelectbox div[data-baseweb="select"] > div,
.stMultiSelect div[data-baseweb="select"] > div {
    background: var(--surface2) !important;
    border: 1px solid #2d3748 !important;
}

/* Dividers */
hr { border-color: #2d3748; }

/* Tags */
.tag {
    display: inline-block;
    background: var(--surface2);
    border: 1px solid var(--accent);
    color: var(--accent);
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 2px 8px;
    border-radius: 4px;
    margin: 2px;
}

.stat-card {
    background: var(--surface2);
    border: 1px solid #2d3748;
    border-left: 3px solid var(--accent);
    border-radius: 8px;
    padding: 16px 20px;
    margin: 8px 0;
}

.result-box {
    background: var(--surface2);
    border: 1px solid #2d3748;
    border-radius: 8px;
    padding: 20px;
    margin: 12px 0;
}

.result-box h4 { font-family: 'Space Mono', monospace; color: var(--accent); margin: 0 0 12px 0; font-size: 0.85rem; letter-spacing: 0.05em; }

.sig { color: var(--success); font-weight: 600; }
.not-sig { color: var(--warn); font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ─── Session State ────────────────────────────────────────────────────────────
if "df" not in st.session_state:
    st.session_state.df = None
if "df_processed" not in st.session_state:
    st.session_state.df_processed = None

# ─── Sidebar Navigation ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 📊 StatLab Pro")
    st.markdown("<p style='color:#64748b;font-size:0.75rem;font-family:Space Mono,monospace;'>v1.0 · Statistical Analysis</p>", unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["📁  Upload Data", "🔧  Preprocessing", "🔍  EDA", "🧪  Statistical Tests", "📈  Visualization", "💾  Export"],
        label_visibility="collapsed",
    )
    st.markdown("---")

    # Quick dataset status
    if st.session_state.df is not None:
        df_ref = st.session_state.df_processed if st.session_state.df_processed is not None else st.session_state.df
        st.markdown(f"<div class='stat-card'><b style='color:#00e5ff;font-family:Space Mono,monospace;font-size:0.75rem;'>ACTIVE DATASET</b><br><span style='font-size:0.85rem;'>{df_ref.shape[0]:,} rows · {df_ref.shape[1]} cols</span></div>", unsafe_allow_html=True)

# ─── Helper ───────────────────────────────────────────────────────────────────
def get_df():
    if st.session_state.df_processed is not None:
        return st.session_state.df_processed
    return st.session_state.df

def plotly_dark():
    return "plotly_dark"

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — UPLOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════
if page == "📁  Upload Data":
    st.markdown("# Upload Data")
    st.markdown("Load your dataset from a local file or GitHub raw URL.")
    st.markdown("---")

    tab1, tab2 = st.tabs(["📂 Local File", "🐙 GitHub URL"])

    with tab1:
        uploaded = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"])
        if uploaded:
            try:
                if uploaded.name.endswith(".csv"):
                    df = pd.read_csv(uploaded)
                else:
                    df = pd.read_excel(uploaded)
                st.session_state.df = df
                st.session_state.df_processed = None
                st.success(f"✅ Loaded **{uploaded.name}** — {df.shape[0]:,} rows × {df.shape[1]} columns")
            except Exception as e:
                st.error(f"Error reading file: {e}")

    with tab2:
        url = st.text_input(
            "GitHub Raw URL",
            placeholder="https://raw.githubusercontent.com/user/repo/main/data.csv",
        )
        if st.button("Load from GitHub"):
            if url:
                with st.spinner("Fetching dataset…"):
                    df, err = load_from_github(url)
                if err:
                    st.error(err)
                else:
                    st.session_state.df = df
                    st.session_state.df_processed = None
                    st.success(f"✅ Loaded from GitHub — {df.shape[0]:,} rows × {df.shape[1]} columns")
            else:
                st.warning("Please enter a URL.")

    # Preview
    if st.session_state.df is not None:
        df = st.session_state.df
        st.markdown("---")
        st.markdown("### Dataset Preview")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{df.shape[0]:,}")
        c2.metric("Columns", f"{df.shape[1]}")
        c3.metric("Missing Values", f"{df.isnull().sum().sum():,}")
        c4.metric("Duplicates", f"{df.duplicated().sum():,}")

        with st.expander("📋 Column Info", expanded=True):
            info = pd.DataFrame({
                "Column": df.columns,
                "Dtype": df.dtypes.values,
                "Non-Null": df.notnull().sum().values,
                "Null": df.isnull().sum().values,
                "Null %": (df.isnull().sum().values / len(df) * 100).round(2),
                "Unique": df.nunique().values,
            })
            st.dataframe(info, use_container_width=True)

        n_rows = st.slider("Preview rows", 5, min(100, len(df)), 10)
        st.dataframe(df.head(n_rows), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔧  Preprocessing":
    st.markdown("# Preprocessing")
    if st.session_state.df is None:
        st.warning("⚠️ Please upload a dataset first.")
        st.stop()

    df = get_df().copy()
    st.markdown("---")

    # ── Missing Values ──────────────────────────────────────────────
    with st.expander("🕳️ Missing Value Handling", expanded=True):
        missing_cols = df.columns[df.isnull().any()].tolist()
        if not missing_cols:
            st.success("No missing values detected.")
        else:
            st.markdown(f"**{len(missing_cols)} columns** have missing values.")
            strategy = st.selectbox("Strategy", ["Drop rows", "Mean imputation", "Median imputation", "Mode imputation", "Forward fill", "Backward fill"])
            cols_to_fix = st.multiselect("Apply to columns", missing_cols, default=missing_cols)
            if st.button("Apply Missing Value Strategy"):
                df = handle_missing_values(df, strategy, cols_to_fix)
                st.session_state.df_processed = df
                st.success("✅ Applied.")

    # ── Data Type Conversion ────────────────────────────────────────
    with st.expander("🔁 Data Type Conversion"):
        col_to_convert = st.selectbox("Column", df.columns)
        target_type = st.selectbox("Convert to", ["int64", "float64", "str", "datetime64", "category"])
        if st.button("Convert"):
            df, err = convert_dtypes(df, col_to_convert, target_type)
            if err:
                st.error(err)
            else:
                st.session_state.df_processed = df
                st.success(f"✅ Converted `{col_to_convert}` to `{target_type}`.")

    # ── Outlier Detection ───────────────────────────────────────────
    with st.expander("📍 Outlier Detection"):
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        if not num_cols:
            st.info("No numeric columns found.")
        else:
            out_col = st.selectbox("Column", num_cols, key="out_col")
            out_method = st.radio("Method", ["IQR", "Z-Score"], horizontal=True)
            if out_method == "IQR":
                multiplier = st.slider("IQR multiplier", 1.0, 3.0, 1.5, 0.1)
                outliers, bounds = detect_outliers_iqr(df, out_col, multiplier)
                st.markdown(f"Bounds: **{bounds[0]:.3f}** — **{bounds[1]:.3f}** · Outliers: **{len(outliers)}**")
            else:
                threshold = st.slider("Z-score threshold", 1.0, 4.0, 3.0, 0.1)
                outliers, _ = detect_outliers_zscore(df, out_col, threshold)
                st.markdown(f"Outliers detected: **{len(outliers)}**")

            action = st.radio("Action", ["Highlight only", "Remove outliers"], horizontal=True)
            if st.button("Apply Outlier Action"):
                if action == "Remove outliers":
                    df = df.drop(outliers.index)
                    st.session_state.df_processed = df
                    st.success(f"✅ Removed {len(outliers)} outliers.")
                else:
                    st.dataframe(outliers, use_container_width=True)

    # ── Feature Selection ───────────────────────────────────────────
    with st.expander("🎯 Feature Selection"):
        all_cols = df.columns.tolist()
        selected = st.multiselect("Keep columns", all_cols, default=all_cols)
        if st.button("Apply Feature Selection"):
            df = df[selected]
            st.session_state.df_processed = df
            st.success(f"✅ Kept {len(selected)} columns.")

    st.markdown("---")
    st.markdown("### Processed Dataset Preview")
    df_show = get_df()
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{df_show.shape[0]:,}")
    c2.metric("Columns", f"{df_show.shape[1]}")
    c3.metric("Missing", f"{df_show.isnull().sum().sum():,}")
    st.dataframe(df_show.head(20), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — EDA
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔍  EDA":
    st.markdown("# Exploratory Data Analysis")
    if st.session_state.df is None:
        st.warning("⚠️ Please upload a dataset first.")
        st.stop()

    df = get_df()
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    st.markdown("---")

    # Summary statistics
    with st.expander("📊 Summary Statistics", expanded=True):
        if num_cols:
            stats = summary_statistics(df[num_cols])
            st.dataframe(stats.style.format("{:.4f}"), use_container_width=True)
        else:
            st.info("No numeric columns.")

    # Correlation Heatmap
    with st.expander("🌡️ Correlation Heatmap"):
        if len(num_cols) >= 2:
            corr_method = st.radio("Method", ["pearson", "spearman", "kendall"], horizontal=True)
            corr = df[num_cols].corr(method=corr_method)
            fig = px.imshow(
                corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                zmin=-1, zmax=1, template=plotly_dark(),
                title=f"{corr_method.capitalize()} Correlation Matrix",
            )
            fig.update_layout(paper_bgcolor="#161922", plot_bgcolor="#161922")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Need at least 2 numeric columns.")

    # Distribution Plots
    with st.expander("📉 Distribution Plots"):
        if num_cols:
            dist_col = st.selectbox("Column", num_cols)
            plot_type = st.radio("Plot type", ["Histogram", "KDE", "Box", "Violin"], horizontal=True)

            if plot_type == "Histogram":
                bins = st.slider("Bins", 5, 100, 30)
                fig = px.histogram(df, x=dist_col, nbins=bins, template=plotly_dark(), marginal="rug")
            elif plot_type == "KDE":
                fig = px.histogram(df, x=dist_col, histnorm="density", template=plotly_dark())
            elif plot_type == "Box":
                fig = px.box(df, y=dist_col, template=plotly_dark(), points="outliers")
            else:
                fig = px.violin(df, y=dist_col, template=plotly_dark(), box=True, points="all")

            fig.update_layout(paper_bgcolor="#161922", plot_bgcolor="#161922")
            st.plotly_chart(fig, use_container_width=True)

    # Pair Plot
    with st.expander("🔗 Pair Plot"):
        if len(num_cols) >= 2:
            pair_cols = st.multiselect("Select columns (2–6)", num_cols, default=num_cols[:min(4, len(num_cols))])
            color_col = st.selectbox("Color by (optional)", ["None"] + cat_cols)
            if len(pair_cols) >= 2:
                if st.button("Generate Pair Plot"):
                    with st.spinner("Rendering…"):
                        fig = px.scatter_matrix(
                            df,
                            dimensions=pair_cols,
                            color=None if color_col == "None" else color_col,
                            template=plotly_dark(),
                        )
                        fig.update_layout(paper_bgcolor="#161922", height=700)
                        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — STATISTICAL TESTS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🧪  Statistical Tests":
    st.markdown("# Statistical Tests")
    if st.session_state.df is None:
        st.warning("⚠️ Please upload a dataset first.")
        st.stop()

    df = get_df()
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    alpha = st.sidebar.slider("Significance Level (α)", 0.01, 0.10, 0.05, 0.01)
    st.markdown("---")

    test_cat = st.selectbox("Test Category", [
        "Normality — Shapiro-Wilk",
        "t-test — Independent Samples",
        "t-test — Paired Samples",
        "ANOVA — One-Way",
        "Chi-Square Test",
        "Correlation — Pearson",
        "Correlation — Spearman",
        "Confidence Interval",
    ])

    def show_result(result: dict, alpha: float):
        stat_fmt = f"{result.get('statistic', 0):.4f}"
        pval = result.get("p_value", 1)
        sig = pval < alpha
        sig_label = f"<span class='sig'>✅ SIGNIFICANT</span>" if sig else f"<span class='not-sig'>❌ NOT SIGNIFICANT</span>"
        st.markdown(f"""
        <div class='result-box'>
            <h4>{result.get('test_name','Result')}</h4>
            <table style='width:100%;font-size:0.9rem;'>
                <tr><td style='color:#64748b;'>Test Statistic</td><td><b>{stat_fmt}</b></td></tr>
                <tr><td style='color:#64748b;'>p-value</td><td><b>{pval:.6f}</b></td></tr>
                <tr><td style='color:#64748b;'>α level</td><td>{alpha}</td></tr>
                <tr><td style='color:#64748b;'>Decision</td><td>{sig_label}</td></tr>
                <tr><td style='color:#64748b;'>Interpretation</td><td>{result.get('interpretation','')}</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

        if "extras" in result:
            for k, v in result["extras"].items():
                st.markdown(f"**{k}**: {v}")

    # ── Tests ───────────────────────────────────────────────────────
    if test_cat == "Normality — Shapiro-Wilk":
        col = st.selectbox("Column", num_cols)
        if st.button("Run Shapiro-Wilk"):
            result = run_shapiro_wilk(df, col, alpha)
            show_result(result, alpha)

    elif test_cat == "t-test — Independent Samples":
        col = st.selectbox("Numeric column", num_cols)
        group_col = st.selectbox("Grouping column", cat_cols if cat_cols else num_cols)
        groups = df[group_col].unique()
        g1 = st.selectbox("Group 1", groups)
        g2 = st.selectbox("Group 2", [g for g in groups if g != g1])
        equal_var = st.checkbox("Assume equal variances", value=True)
        if st.button("Run t-test"):
            result = run_ttest_independent(df, col, group_col, g1, g2, equal_var, alpha)
            show_result(result, alpha)

    elif test_cat == "t-test — Paired Samples":
        c1 = st.selectbox("Variable 1", num_cols)
        c2 = st.selectbox("Variable 2", [c for c in num_cols if c != c1])
        if st.button("Run Paired t-test"):
            result = run_ttest_paired(df, c1, c2, alpha)
            show_result(result, alpha)

    elif test_cat == "ANOVA — One-Way":
        col = st.selectbox("Numeric column", num_cols)
        group_col = st.selectbox("Grouping column", cat_cols if cat_cols else num_cols)
        if st.button("Run ANOVA"):
            result = run_anova(df, col, group_col, alpha)
            show_result(result, alpha)

    elif test_cat == "Chi-Square Test":
        c1 = st.selectbox("Variable 1", cat_cols if cat_cols else df.columns.tolist())
        c2 = st.selectbox("Variable 2", [c for c in (cat_cols if cat_cols else df.columns.tolist()) if c != c1])
        if st.button("Run Chi-Square"):
            result = run_chi_square(df, c1, c2, alpha)
            show_result(result, alpha)
            with st.expander("Contingency Table"):
                st.dataframe(pd.crosstab(df[c1], df[c2]), use_container_width=True)

    elif test_cat == "Correlation — Pearson":
        c1 = st.selectbox("Variable 1", num_cols)
        c2 = st.selectbox("Variable 2", [c for c in num_cols if c != c1])
        if st.button("Run Pearson"):
            result = run_pearson(df, c1, c2, alpha)
            show_result(result, alpha)
            fig = px.scatter(df, x=c1, y=c2, trendline="ols", template=plotly_dark())
            fig.update_layout(paper_bgcolor="#161922")
            st.plotly_chart(fig, use_container_width=True)

    elif test_cat == "Correlation — Spearman":
        c1 = st.selectbox("Variable 1", num_cols)
        c2 = st.selectbox("Variable 2", [c for c in num_cols if c != c1])
        if st.button("Run Spearman"):
            result = run_spearman(df, c1, c2, alpha)
            show_result(result, alpha)

    elif test_cat == "Confidence Interval":
        col = st.selectbox("Column", num_cols)
        ci_level = st.slider("Confidence Level", 0.80, 0.99, 0.95, 0.01)
        if st.button("Compute CI"):
            result = compute_confidence_interval(df, col, ci_level)
            st.markdown(f"""
            <div class='result-box'>
                <h4>Confidence Interval — {col}</h4>
                <table style='width:100%;font-size:0.9rem;'>
                    <tr><td style='color:#64748b;'>Mean</td><td><b>{result['mean']:.4f}</b></td></tr>
                    <tr><td style='color:#64748b;'>Standard Error</td><td><b>{result['se']:.4f}</b></td></tr>
                    <tr><td style='color:#64748b;'>Lower Bound</td><td><b>{result['lower']:.4f}</b></td></tr>
                    <tr><td style='color:#64748b;'>Upper Bound</td><td><b>{result['upper']:.4f}</b></td></tr>
                    <tr><td style='color:#64748b;'>Confidence</td><td>{int(ci_level*100)}%</td></tr>
                </table>
            </div>
            """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📈  Visualization":
    st.markdown("# Visualization Dashboard")
    if st.session_state.df is None:
        st.warning("⚠️ Please upload a dataset first.")
        st.stop()

    df = get_df()
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    all_cols = df.columns.tolist()
    st.markdown("---")

    # Dynamic filter
    with st.expander("🔎 Dynamic Filters"):
        filter_col = st.selectbox("Filter by column", ["None"] + cat_cols)
        if filter_col != "None":
            filter_vals = st.multiselect("Values", df[filter_col].unique(), default=df[filter_col].unique())
            df = df[df[filter_col].isin(filter_vals)]
        if num_cols:
            range_col = st.selectbox("Numeric range filter", ["None"] + num_cols)
            if range_col != "None":
                mn, mx = float(df[range_col].min()), float(df[range_col].max())
                rng = st.slider(f"{range_col} range", mn, mx, (mn, mx))
                df = df[(df[range_col] >= rng[0]) & (df[range_col] <= rng[1])]
        st.markdown(f"Filtered: **{len(df):,}** rows")

    chart_type = st.selectbox("Chart Type", [
        "Scatter Plot", "Line Chart", "Bar Chart", "Box Plot", "Violin Plot",
        "Histogram", "Bubble Chart", "Heatmap", "Treemap", "Sunburst",
    ])

    fig = None

    if chart_type == "Scatter Plot":
        x = st.selectbox("X axis", num_cols)
        y = st.selectbox("Y axis", [c for c in num_cols if c != x])
        color = st.selectbox("Color", ["None"] + cat_cols + num_cols)
        size = st.selectbox("Size", ["None"] + num_cols)
        fig = px.scatter(df, x=x, y=y,
                         color=None if color == "None" else color,
                         size=None if size == "None" else size,
                         template=plotly_dark(), trendline="ols" if st.checkbox("Trendline") else None)

    elif chart_type == "Line Chart":
        x = st.selectbox("X axis", all_cols)
        y = st.multiselect("Y axis", num_cols, default=[num_cols[0]] if num_cols else [])
        if y:
            fig = px.line(df, x=x, y=y, template=plotly_dark())

    elif chart_type == "Bar Chart":
        x = st.selectbox("X axis", cat_cols + num_cols)
        y = st.selectbox("Y axis", num_cols)
        barmode = st.radio("Mode", ["group", "stack", "overlay"], horizontal=True)
        color = st.selectbox("Color", ["None"] + cat_cols)
        fig = px.bar(df, x=x, y=y, barmode=barmode,
                     color=None if color == "None" else color, template=plotly_dark())

    elif chart_type == "Box Plot":
        y = st.selectbox("Numeric", num_cols)
        x = st.selectbox("Group by", ["None"] + cat_cols)
        fig = px.box(df, x=None if x == "None" else x, y=y, template=plotly_dark(), points="outliers")

    elif chart_type == "Violin Plot":
        y = st.selectbox("Numeric", num_cols)
        x = st.selectbox("Group by", ["None"] + cat_cols)
        fig = px.violin(df, x=None if x == "None" else x, y=y, template=plotly_dark(), box=True)

    elif chart_type == "Histogram":
        x = st.selectbox("Column", num_cols)
        bins = st.slider("Bins", 5, 200, 30)
        color = st.selectbox("Color", ["None"] + cat_cols)
        fig = px.histogram(df, x=x, nbins=bins, color=None if color == "None" else color,
                           template=plotly_dark(), marginal="box")

    elif chart_type == "Bubble Chart":
        x = st.selectbox("X", num_cols)
        y = st.selectbox("Y", [c for c in num_cols if c != x])
        size = st.selectbox("Bubble size", [c for c in num_cols if c not in [x, y]])
        color = st.selectbox("Color", ["None"] + cat_cols)
        fig = px.scatter(df, x=x, y=y, size=size, color=None if color == "None" else color,
                         template=plotly_dark())

    elif chart_type == "Heatmap":
        if len(num_cols) >= 2:
            fig = px.imshow(df[num_cols].corr(), text_auto=".2f",
                            color_continuous_scale="RdBu_r", template=plotly_dark())

    elif chart_type == "Treemap":
        path = st.multiselect("Hierarchy (path)", cat_cols, default=cat_cols[:min(2, len(cat_cols))])
        val = st.selectbox("Value", num_cols) if num_cols else None
        if path and val:
            fig = px.treemap(df, path=path, values=val, template=plotly_dark())

    elif chart_type == "Sunburst":
        path = st.multiselect("Hierarchy (path)", cat_cols, default=cat_cols[:min(2, len(cat_cols))])
        val = st.selectbox("Value", num_cols) if num_cols else None
        if path and val:
            fig = px.sunburst(df, path=path, values=val, template=plotly_dark())

    if fig:
        fig.update_layout(paper_bgcolor="#161922", plot_bgcolor="#161922", font_color="#e2e8f0")
        st.plotly_chart(fig, use_container_width=True)

        # Download plot
        buf = io.StringIO()
        fig.write_html(buf)
        html_bytes = buf.getvalue().encode()
        st.download_button(
            "⬇️ Download Plot (HTML)",
            data=html_bytes,
            file_name="plot.html",
            mime="text/html",
        )

    # Model Insights
    st.markdown("---")
    st.markdown("### 🤖 Model Insights")
    model_tab1, model_tab2 = st.tabs(["Linear Regression", "Logistic Regression"])

    with model_tab1:
        if len(num_cols) >= 2:
            target_lr = st.selectbox("Target (Y)", num_cols, key="lr_target")
            features_lr = st.multiselect("Features (X)", [c for c in num_cols if c != target_lr], key="lr_feats")
            if features_lr and st.button("Run Linear Regression"):
                results = run_linear_regression(df, features_lr, target_lr)
                if "error" in results:
                    st.error(results["error"])
                else:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("R²", f"{results['r2']:.4f}")
                    c2.metric("Adj R²", f"{results['adj_r2']:.4f}")
                    c3.metric("RMSE", f"{results['rmse']:.4f}")

                    st.markdown("**Coefficients**")
                    coef_df = pd.DataFrame({"Feature": ["Intercept"] + features_lr,
                                            "Coefficient": [results["intercept"]] + list(results["coefs"])})
                    st.dataframe(coef_df, use_container_width=True)

                    # Residual plot
                    fig_res = px.scatter(x=results["predictions"], y=results["residuals"],
                                         labels={"x": "Predicted", "y": "Residuals"},
                                         title="Residual Plot", template=plotly_dark())
                    fig_res.add_hline(y=0, line_dash="dash", line_color="red")
                    fig_res.update_layout(paper_bgcolor="#161922")
                    st.plotly_chart(fig_res, use_container_width=True)

    with model_tab2:
        cat_targets = [c for c in cat_cols if df[c].nunique() == 2]
        if cat_targets and num_cols:
            target_log = st.selectbox("Binary Target", cat_targets, key="log_target")
            features_log = st.multiselect("Features (X)", num_cols, key="log_feats")
            if features_log and st.button("Run Logistic Regression"):
                results = run_logistic_regression(df, features_log, target_log)
                if "error" in results:
                    st.error(results["error"])
                else:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Accuracy", f"{results['accuracy']:.4f}")
                    c2.metric("Precision", f"{results['precision']:.4f}")
                    c3.metric("Recall", f"{results['recall']:.4f}")

                    st.markdown("**Feature Importance**")
                    imp_df = pd.DataFrame({"Feature": features_log, "Coefficient": results["coefs"]})
                    imp_df = imp_df.sort_values("Coefficient", key=abs, ascending=False)
                    fig_imp = px.bar(imp_df, x="Coefficient", y="Feature", orientation="h",
                                     template=plotly_dark(), title="Feature Coefficients")
                    fig_imp.update_layout(paper_bgcolor="#161922")
                    st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.info("Need numeric features and a binary categorical target column.")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — EXPORT
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "💾  Export":
    st.markdown("# Export")
    if st.session_state.df is None:
        st.warning("⚠️ Please upload a dataset first.")
        st.stop()

    df = get_df()
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📄 Dataset")
        fmt = st.radio("Format", ["CSV", "Excel"], horizontal=True)
        if fmt == "CSV":
            data = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download CSV", data=data, file_name="processed_dataset.csv", mime="text/csv")
        else:
            buf = io.BytesIO()
            df.to_excel(buf, index=False, engine="openpyxl")
            st.download_button("⬇️ Download Excel", data=buf.getvalue(),
                               file_name="processed_dataset.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    with col2:
        st.markdown("### 📊 Statistical Summary")
        report_fmt = st.radio("Report Format", ["CSV", "PDF"], horizontal=True)
        if report_fmt == "CSV":
            csv_report = generate_report_csv(df)
            st.download_button("⬇️ Download Report CSV", data=csv_report,
                               file_name="statistical_report.csv", mime="text/csv")
        else:
            pdf_bytes = generate_report_pdf(df)
            if pdf_bytes:
                st.download_button("⬇️ Download Report PDF", data=pdf_bytes,
                                   file_name="statistical_report.pdf", mime="application/pdf")
            else:
                st.info("PDF export requires `reportlab`. Install it via `pip install reportlab`.")

    # Info summary
    st.markdown("---")
    st.markdown("### Dataset Summary")
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if num_cols:
        st.dataframe(df[num_cols].describe().T.round(4), use_container_width=True)

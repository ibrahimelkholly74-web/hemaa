import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DataLens - Intelligent Analysis",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, [class*="css"], .stApp { font-family: 'Syne', sans-serif !important; background-color: #08080f !important; color: #e8e8f8 !important; }
.stApp { background-color: #08080f !important; }
section[data-testid="stSidebar"] { background-color: #0f0f1c !important; border-right: 1px solid #1e1e35 !important; }
.block-container { padding-top: 1.5rem !important; max-width: 1400px; }
[data-testid="metric-container"] { background: #0f0f1c; border: 1px solid #1e1e35; border-radius: 12px; padding: 16px !important; }
[data-testid="stMetricValue"] { color: #e8e8f8 !important; font-weight: 800 !important; }
[data-testid="stMetricLabel"] { color: #5a5a80 !important; font-family: 'JetBrains Mono', monospace !important; font-size: 11px !important; }
.stTabs [data-baseweb="tab-list"] { background: #0f0f1c; border-radius: 12px; padding: 4px; border: 1px solid #1e1e35; gap: 4px; }
.stTabs [data-baseweb="tab"] { background: transparent; border-radius: 9px; color: #5a5a80; font-weight: 600; font-family: 'Syne', sans-serif; }
.stTabs [aria-selected="true"] { background: #7c6af7 !important; color: white !important; }
hr { border-color: #1e1e35 !important; }
.note-ok   { background:#0a2e1f; border-left:3px solid #34d399; border-radius:8px; padding:10px 14px; margin-bottom:8px; color:#a7f3d0; font-size:13px; }
.note-warn { background:#2d1f08; border-left:3px solid #fbbf24; border-radius:8px; padding:10px 14px; margin-bottom:8px; color:#fde68a; font-size:13px; }
.note-info { background:#16142e; border-left:3px solid #7c6af7; border-radius:8px; padding:10px 14px; margin-bottom:8px; color:#a78bfa; font-size:13px; }
</style>
""", unsafe_allow_html=True)

PALETTE = ['#7c6af7','#22d3ee','#34d399','#f87171','#fbbf24','#f472b6','#818cf8','#6ee7b7','#fca5a5','#fde68a']
PLOT_BG = "#0f0f1c"

def plot(fig):
    fig.update_layout(
        plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG,
        font=dict(family="JetBrains Mono", color="#94a3b8", size=11),
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor="#1e1e35", zerolinecolor="#1e1e35"),
        yaxis=dict(gridcolor="#1e1e35", zerolinecolor="#1e1e35"),
    )
    return fig

def classify(df):
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    cat = df.select_dtypes(exclude=[np.number]).columns.tolist()
    return num, cat

def clean(df_raw):
    df = df_raw.copy()
    notes = []
    df.columns = df.columns.str.strip()

    before = len(df)
    df.dropna(how="all", inplace=True)
    if (r := before - len(df)): notes.append(("info", f"🗑️ Removed **{r}** empty row(s)"))

    before = len(df)
    df.drop_duplicates(inplace=True)
    if (d := before - len(df)): notes.append(("info", f"🔁 Removed **{d}** duplicate row(s)"))

    str_cols = df.select_dtypes(include="object").columns.tolist()
    for col in str_cols:
        df[col] = df[col].astype(str).str.strip().replace("nan", np.nan)

    for col in list(str_cols):
        converted = pd.to_numeric(df[col], errors="coerce")
        orig = df[col].notna().sum()
        if orig > 0 and converted.notna().sum() / orig > 0.8:
            coerced = int(orig - converted.notna().sum())
            df[col] = converted
            if coerced: notes.append(("warn", f'🔢 **"{col}"** coerced to numeric — {coerced} value(s) → NaN'))

    for col in df.columns:
        if (m := int(df[col].isna().sum())):
            notes.append(("warn", f'⚠️ **"{col}"** has {m} missing ({m/len(df)*100:.1f}%)'))

    if not notes:
        notes.append(("ok", "✅ Data looks clean — no issues found!"))
    return df, notes

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ DataLens")
    st.caption("Intelligent Analysis System")
    st.divider()
    uploaded = st.file_uploader("Upload dataset", type=["csv","xlsx","xls"])
    if uploaded:
        st.success(f"📁 {uploaded.name}")
        st.caption(f"{uploaded.size/1024:.1f} KB")

# ── Load ───────────────────────────────────────────────────────────────────────
if not uploaded:
    st.markdown("# ⚡ DataLens")
    st.markdown("### Intelligent Data Analysis System")
    st.divider()
    c1, c2, c3 = st.columns(3)
    with c1: st.markdown("**📂 Upload**\n\nCSV or Excel. Auto-detects types and cleans instantly.")
    with c2: st.markdown("**📊 Visualize**\n\nBar, Line, Pie, Histogram, Scatter, Heatmap — placed smartly.")
    with c3: st.markdown("**🧮 Analyze**\n\nPivot tables, correlation matrix, column stats, quality report.")
    st.info("👈 Upload a file from the sidebar to get started")
    st.stop()

try:
    df_raw = pd.read_csv(uploaded) if uploaded.name.lower().endswith(".csv") else pd.read_excel(uploaded)
except Exception as e:
    st.error(f"❌ Could not read file: {e}"); st.stop()

df, notes = clean(df_raw)
num_cols, cat_cols = classify(df)

# ── Stats ──────────────────────────────────────────────────────────────────────
st.markdown("### Dataset Overview")
sc = st.columns(4 + min(len(num_cols), 3))
sc[0].metric("Rows", f"{len(df):,}", delta=f"-{len(df_raw)-len(df)} cleaned" if len(df_raw)!=len(df) else "All kept")
sc[1].metric("Columns", len(df.columns))
sc[2].metric("Numeric", len(num_cols))
sc[3].metric("Categorical", len(cat_cols))
for i, col in enumerate(num_cols[:3]):
    sc[4+i].metric(col[:14], f"{df[col].mean():.2f}", delta=f"max {df[col].max():.1f}")

st.divider()

# ── Tabs ───────────────────────────────────────────────────────────────────────
t1, t2, t3, t4 = st.tabs(["📊 Dashboard", "🧮 Pivot Tables", "🧹 Data Quality", "📋 Preview"])

# ── DASHBOARD ─────────────────────────────────────────────────────────────────
with t1:
    cc1, cc2, _ = st.columns([1,1,2])
    all_cols = df.columns.tolist()
    x_col = cc1.selectbox("X Axis", all_cols, index=all_cols.index(cat_cols[0]) if cat_cols else 0)
    y_col = cc2.selectbox("Y Axis (numeric)", all_cols, index=all_cols.index(num_cols[0]) if num_cols else 0)
    st.markdown("---")

    a, b = st.columns(2)

    # Bar chart
    with a:
        st.markdown("##### 📊 Bar Chart")
        if x_col in cat_cols and y_col in num_cols:
            bd = df.groupby(x_col)[y_col].sum().reset_index().sort_values(y_col, ascending=False).head(15)
            fig = px.bar(bd, x=x_col, y=y_col, color=x_col, color_discrete_sequence=PALETTE, template="plotly_dark")
            fig.update_layout(showlegend=False)
            st.plotly_chart(plot(fig), use_container_width=True)
        else:
            st.info("Select a categorical X and numeric Y.")

    # Line chart
    with b:
        st.markdown("##### 📈 Line Chart")
        if y_col in num_cols:
            ld = df[[x_col, y_col]].dropna().head(300)
            fig = px.line(ld, x=x_col, y=y_col, color_discrete_sequence=["#34d399"], template="plotly_dark")
            fig.update_traces(line=dict(width=2))
            st.plotly_chart(plot(fig), use_container_width=True)

    c, d = st.columns(2)

    # Pie chart
    with c:
        st.markdown("##### 🥧 Pie Chart")
        if cat_cols:
            pd_data = df[cat_cols[0]].value_counts().head(10).reset_index()
            pd_data.columns = [cat_cols[0], "count"]
            fig = px.pie(pd_data, names=cat_cols[0], values="count", color_discrete_sequence=PALETTE,
                         template="plotly_dark", hole=0.4)
            st.plotly_chart(plot(fig), use_container_width=True)
        else:
            st.info("No categorical columns found.")

    # Histogram
    with d:
        st.markdown("##### 📉 Histogram")
        if y_col in num_cols:
            fig = px.histogram(df, x=y_col, nbins=20, color_discrete_sequence=["#818cf8"], template="plotly_dark")
            fig.update_layout(bargap=0.05)
            st.plotly_chart(plot(fig), use_container_width=True)

    e, f = st.columns(2)

    # Scatter
    with e:
        st.markdown("##### 🔵 Scatter Plot")
        if len(num_cols) >= 2:
            sx, sy = num_cols[0], num_cols[1]
            sdf = df[[sx, sy]].dropna().sample(min(500, len(df)), random_state=42)
            ckw = dict(color=cat_cols[0], color_discrete_sequence=PALETTE) if cat_cols else {}
            fig = px.scatter(sdf, x=sx, y=sy, opacity=0.7, template="plotly_dark", **ckw)
            st.plotly_chart(plot(fig), use_container_width=True)
        else:
            st.info("Need at least 2 numeric columns.")

    # Heatmap
    with f:
        st.markdown("##### 🌡️ Heatmap")
        if len(num_cols) >= 2:
            corr = df[num_cols[:10]].corr().round(3)
            fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                            zmin=-1, zmax=1, template="plotly_dark")
            fig.update_traces(textfont=dict(size=9))
            st.plotly_chart(plot(fig), use_container_width=True)
        elif len(cat_cols) >= 2:
            hm = df.groupby([cat_cols[0], cat_cols[1]]).size().reset_index(name="n")
            hm = hm.pivot_table(index=cat_cols[0], columns=cat_cols[1], values="n", fill_value=0).iloc[:10,:10]
            fig = px.imshow(hm, text_auto=True, color_continuous_scale="Viridis", template="plotly_dark")
            st.plotly_chart(plot(fig), use_container_width=True)
        else:
            st.info("Need at least 2 numeric columns.")

# ── PIVOT TABLES ──────────────────────────────────────────────────────────────
with t2:
    if not cat_cols or not num_cols:
        st.warning("Need at least one categorical and one numeric column.")
    else:
        p1, p2 = st.columns(2)
        grp = p1.selectbox("Group By", cat_cols, key="grp")
        val = p2.selectbox("Values", num_cols, key="val")
        pv = (df.groupby(grp)[val]
              .agg(Count="count", Sum="sum", Mean="mean", Min="min", Max="max", Std="std")
              .round(2).reset_index().sort_values("Sum", ascending=False))
        st.dataframe(pv, use_container_width=True, hide_index=True)
        st.divider()
        st.markdown("##### All Combinations")
        for cat in cat_cols[:2]:
            for num in num_cols[:3]:
                with st.expander(f"📌 {cat} × {num}"):
                    p = (df.groupby(cat)[num]
                         .agg(Count="count", Sum="sum", Mean="mean", Min="min", Max="max")
                         .round(2).reset_index().sort_values("Sum", ascending=False).head(20))
                    st.dataframe(p, use_container_width=True, hide_index=True)

# ── DATA QUALITY ──────────────────────────────────────────────────────────────
with t3:
    removed = len(df_raw) - len(df)
    st.markdown(f"**{len(df_raw):,}** raw rows → **{len(df):,}** cleaned rows · "
                + (f"`{removed} removed`" if removed else "`no rows removed`"))
    st.markdown("")
    for level, text in notes:
        st.markdown(f'<div class="note-{level}">{text}</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown("##### Column Summary")
    summary = [{"Column": c, "Type": "numeric" if c in num_cols else "categorical",
                "Missing": int(df[c].isna().sum()),
                "% Missing": f"{df[c].isna().mean()*100:.1f}%",
                "Unique": int(df[c].nunique()),
                "Sample": str(df[c].dropna().iloc[0])[:40] if df[c].notna().any() else "N/A"}
               for c in df.columns]
    st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)

    if num_cols:
        st.divider()
        st.markdown("##### Numeric Statistics")
        st.dataframe(df[num_cols].describe().round(3), use_container_width=True)

# ── PREVIEW ───────────────────────────────────────────────────────────────────
with t4:
    st.markdown(f"##### First 100 rows · {len(df):,} total")
    search = st.text_input("🔍 Search", placeholder="Filter rows...", label_visibility="collapsed")
    disp = df.head(100)
    if search:
        mask = disp.apply(lambda r: r.astype(str).str.contains(search, case=False, na=False).any(), axis=1)
        disp = disp[mask]
        st.caption(f"{len(disp)} matching rows")
    st.dataframe(disp, use_container_width=True, hide_index=True)

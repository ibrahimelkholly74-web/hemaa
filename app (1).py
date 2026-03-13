from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import numpy as np
import json
import os
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__, static_folder="static")

# ── Helpers ────────────────────────────────────────────────────────────────────

def classify_columns(df):
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical = df.select_dtypes(exclude=[np.number]).columns.tolist()
    return numeric, categorical


def clean_dataframe(df_raw):
    df = df_raw.copy()
    notes = []

    df.columns = df.columns.str.strip()

    before = len(df)
    df.dropna(how="all", inplace=True)
    removed = before - len(df)
    if removed:
        notes.append({"level": "info", "text": f"Removed {removed} completely empty row(s)"})

    before = len(df)
    df.drop_duplicates(inplace=True)
    dups = before - len(df)
    if dups:
        notes.append({"level": "info", "text": f"Removed {dups} duplicate row(s)"})

    str_cols = df.select_dtypes(include="object").columns.tolist()
    for col in str_cols:
        df[col] = df[col].astype(str).str.strip().replace("nan", np.nan)

    for col in str_cols:
        converted = pd.to_numeric(df[col], errors="coerce")
        notna_orig = df[col].notna().sum()
        if notna_orig > 0 and converted.notna().sum() / notna_orig > 0.8:
            coerced = int(notna_orig - converted.notna().sum())
            df[col] = converted
            if coerced > 0:
                notes.append({"level": "warn", "text": f'Column "{col}" coerced to numeric; {coerced} value(s) set to null'})

    for col in df.columns:
        missing = int(df[col].isna().sum())
        if missing > 0:
            pct = missing / len(df) * 100
            notes.append({"level": "warn", "text": f'Column "{col}": {missing} missing value(s) ({pct:.1f}%)'})

    if not notes:
        notes.append({"level": "ok", "text": "Data looks clean — no issues found!"})

    return df, notes


def safe_val(v):
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


def df_to_safe(df):
    rows = []
    for row in df.to_dict(orient="records"):
        rows.append({k: safe_val(v) for k, v in row.items()})
    return rows


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    f = request.files["file"]
    name = f.filename.lower()

    try:
        if name.endswith(".csv"):
            df_raw = pd.read_csv(f)
        elif name.endswith((".xlsx", ".xls")):
            df_raw = pd.read_excel(f)
        else:
            return jsonify({"error": "Unsupported file type. Use CSV or XLSX."}), 400
    except Exception as e:
        return jsonify({"error": f"Could not parse file: {str(e)}"}), 400

    df_clean, notes = clean_dataframe(df_raw)
    numeric_cols, cat_cols = classify_columns(df_clean)

    # ── Stats ──────────────────────────────────────────────────────────────────
    stats = {
        "rows_raw": len(df_raw),
        "rows_clean": len(df_clean),
        "cols": len(df_clean.columns),
        "numeric_cols": numeric_cols,
        "cat_cols": cat_cols,
        "columns": df_clean.columns.tolist(),
    }

    col_summary = []
    for col in df_clean.columns:
        col_summary.append({
            "name": col,
            "type": "numeric" if col in numeric_cols else "categorical",
            "missing": int(df_clean[col].isna().sum()),
            "unique": int(df_clean[col].nunique()),
            "sample": str(df_clean[col].dropna().iloc[0]) if df_clean[col].notna().any() else "N/A",
        })

    # ── Chart data ─────────────────────────────────────────────────────────────
    charts = {}

    # Bar chart: first cat x first numeric
    if cat_cols and numeric_cols:
        cat, num = cat_cols[0], numeric_cols[0]
        bar = df_clean.groupby(cat)[num].sum().sort_values(ascending=False).head(15)
        charts["bar"] = {"labels": bar.index.tolist(), "values": [safe_val(v) for v in bar.values], "x": cat, "y": num}

    # Line chart: index vs numeric
    if numeric_cols:
        num = numeric_cols[0]
        ldf = df_clean[[num]].dropna().head(200)
        x_col = cat_cols[0] if cat_cols else None
        if x_col:
            ldf2 = df_clean[[x_col, num]].dropna().head(200)
            charts["line"] = {"labels": ldf2[x_col].tolist(), "values": [safe_val(v) for v in ldf2[num].tolist()], "y": num}
        else:
            charts["line"] = {"labels": list(range(len(ldf))), "values": [safe_val(v) for v in ldf[num].tolist()], "y": num}

    # Pie chart: first categorical
    if cat_cols:
        pie_col = cat_cols[0]
        pie = df_clean[pie_col].value_counts().head(10)
        charts["pie"] = {"labels": pie.index.tolist(), "values": pie.values.tolist(), "col": pie_col}

    # Histogram: first numeric
    if numeric_cols:
        num = numeric_cols[0]
        vals = df_clean[num].dropna()
        counts, edges = np.histogram(vals, bins=15)
        charts["histogram"] = {
            "labels": [f"{e:.1f}" for e in edges[:-1]],
            "values": counts.tolist(),
            "col": num
        }

    # Scatter: first two numerics
    if len(numeric_cols) >= 2:
        sc_x, sc_y = numeric_cols[0], numeric_cols[1]
        sdf = df_clean[[sc_x, sc_y]].dropna().sample(min(300, len(df_clean)))
        charts["scatter"] = {
            "x": sc_x, "y": sc_y,
            "points": [{"x": safe_val(r[sc_x]), "y": safe_val(r[sc_y])} for _, r in sdf.iterrows()]
        }

    # Heatmap: correlation matrix of numeric cols
    if len(numeric_cols) >= 2:
        corr = df_clean[numeric_cols[:8]].corr().round(3)
        charts["heatmap"] = {
            "cols": corr.columns.tolist(),
            "matrix": [[safe_val(v) for v in row] for row in corr.values.tolist()]
        }

    # ── Pivot tables ───────────────────────────────────────────────────────────
    pivots = []
    for cat in cat_cols[:2]:
        for num in numeric_cols[:3]:
            pv = df_clean.groupby(cat)[num].agg(
                count="count", sum="sum", mean="mean", min="min", max="max"
            ).round(2).reset_index().sort_values("sum", ascending=False).head(15)
            pivots.append({
                "title": f"{cat} x {num}",
                "cat": cat, "num": num,
                "rows": df_to_safe(pv)
            })

    # ── Preview ────────────────────────────────────────────────────────────────
    preview = df_to_safe(df_clean.head(50))

    # ── Numeric describe ───────────────────────────────────────────────────────
    describe = {}
    if numeric_cols:
        d = df_clean[numeric_cols].describe().round(3)
        for col in d.columns:
            describe[col] = {k: safe_val(v) for k, v in d[col].to_dict().items()}

    return jsonify({
        "stats": stats,
        "notes": notes,
        "col_summary": col_summary,
        "charts": charts,
        "pivots": pivots,
        "preview": preview,
        "describe": describe,
    })


if __name__ == "__main__":
    os.makedirs("static", exist_ok=True)
    app.run(debug=True, port=5000)

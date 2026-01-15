"""
Streamlit application for Excel Insights AI.

This app allows users to upload Excel or CSV files and generates
interactive dashboards, data quality diagnostics, trend analysis,
correlation insights and narrative summaries using pandas, plotly
and statsmodels. It replicates the core features of the FastAPI
version but runs entirely inside Streamlit, which simplifies
deployment and user interaction.
"""

import io
import datetime
from typing import Optional, Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# NOTE: We avoid importing heavy forecasting libraries such as statsmodels
# because they require compiled dependencies that are not available in all
# deployment environments (e.g. Streamlit Community Cloud).  Instead we
# implement a simple linear regression based forecasting routine below.


def detect_datetime_column(df: pd.DataFrame) -> Optional[str]:
    """Attempt to find a column containing datetime information.

    Prioritises columns whose names contain typical datetime
    keywords and falls back to converting each column to datetime,
    selecting the first with at least half valid parses.

    Args:
        df: Input DataFrame.

    Returns:
        Name of the detected datetime column or ``None``.
    """
    candidates = [c for c in df.columns if any(k in str(c).lower() for k in ["date", "time", "timestamp"])]
    for col in candidates + list(df.columns):
        try:
            conv = pd.to_datetime(df[col], errors="coerce")
        except Exception:
            continue
        if conv.notna().sum() >= len(df) / 2:
            return col
    return None


def describe_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return summary statistics for all columns as a DataFrame."""
    descr = df.describe(include="all").T
    descr = descr.reset_index().rename(columns={"index": "column"})
    return descr


def data_quality_checks(df: pd.DataFrame) -> Dict[str, Any]:
    """Perform expanded data quality checks on the DataFrame.

    Returns a dictionary describing missing values, duplicate rows,
    negative values, outliers (±3 std dev), constant columns and
    high-cardinality categorical columns.
    """
    quality: Dict[str, Any] = {}
    missing = df.isna().sum()
    missing_filtered = missing[missing > 0].to_dict()
    if missing_filtered:
        quality['missing'] = missing_filtered
    dup_count = int(df.duplicated().sum())
    if dup_count > 0:
        quality['duplicates'] = dup_count
    negatives: Dict[str, int] = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        neg_count = int((df[col] < 0).sum())
        if neg_count > 0:
            negatives[col] = neg_count
    if negatives:
        quality['negatives'] = negatives
    outliers: Dict[str, int] = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        series = df[col].dropna().astype(float)
        if len(series) < 10:
            continue
        std = series.std(ddof=0)
        if std == 0:
            continue
        zscores = (series - series.mean()) / std
        out_count = int((abs(zscores) > 3).sum())
        if out_count > 0:
            outliers[col] = out_count
    if outliers:
        quality['outliers'] = outliers
    constants = {col: df[col].iloc[0] for col in df.columns if df[col].nunique(dropna=False) == 1}
    if constants:
        quality['constants'] = constants
    high_card: Dict[str, int] = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        uniq = df[col].nunique(dropna=True)
        if len(df) > 0 and uniq / len(df) > 0.7:
            high_card[col] = uniq
    if high_card:
        quality['high_cardinality'] = high_card
    return quality


def quality_messages(quality: Dict[str, Any]) -> List[str]:
    """Convert data quality check results into human-readable messages."""
    msgs: List[str] = []
    if 'missing' in quality:
        for col, count in quality['missing'].items():
            msgs.append(f"Column '{col}' contains {count} missing value{'s' if count != 1 else ''}.")
    if 'duplicates' in quality:
        count = quality['duplicates']
        msgs.append(f"The dataset contains {count} duplicate row{'s' if count != 1 else ''}.")
    if 'negatives' in quality:
        for col, count in quality['negatives'].items():
            msgs.append(f"Column '{col}' has {count} negative value{'s' if count != 1 else ''}.")
    if 'outliers' in quality:
        for col, count in quality['outliers'].items():
            msgs.append(f"Column '{col}' contains {count} outlier{'s' if count != 1 else ''} beyond ±3 std dev.")
    if 'constants' in quality:
        for col, value in quality['constants'].items():
            msgs.append(f"Column '{col}' has a single unique value ('{value}').")
    if 'high_cardinality' in quality:
        for col, uniq in quality['high_cardinality'].items():
            msgs.append(f"Column '{col}' has a very high number of unique values ({uniq}).")
    if not msgs:
        msgs.append("No obvious data quality issues were detected.")
    return msgs


def compute_trends(df: pd.DataFrame, time_col: Optional[str]) -> List[Dict[str, Any]]:
    """Compute linear trends for numeric columns against time or index."""
    results = []
    if time_col:
        try:
            x = pd.to_datetime(df[time_col], errors="coerce").map(datetime.datetime.toordinal).astype(float)
        except Exception:
            x = pd.to_numeric(df[time_col], errors="coerce")
    else:
        x = np.arange(len(df))
    x = (x - np.nanmean(x)) / (np.nanstd(x) if np.nanstd(x) != 0 else 1)
    for col in df.select_dtypes(include=[np.number]).columns:
        y = df[col].astype(float)
        mask = ~np.isnan(x) & ~np.isnan(y)
        if mask.sum() < 2:
            continue
        xx = x[mask]
        yy = y[mask]
        slope = float(np.cov(xx, yy, bias=True)[0, 1] / np.var(xx)) if np.var(xx) != 0 else 0.0
        if slope > 0.1:
            direction = "increasing"
        elif slope < -0.1:
            direction = "decreasing"
        else:
            direction = "stable"
        results.append({"column": col, "slope": slope, "direction": direction, "comment": f"Column '{col}' shows a {direction} trend."})
    return results


def compute_correlations(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Compute pairwise correlations among numeric columns."""
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        return None
    corr_matrix = numeric_df.corr().fillna(0.0)
    strong_pairs: List[Tuple[str, str, float]] = []
    cols = corr_matrix.columns
    for i, c1 in enumerate(cols):
        for c2 in cols[i+1:]:
            val = float(corr_matrix.loc[c1, c2])
            if abs(val) >= 0.8:
                strong_pairs.append((c1, c2, val))
    return {"matrix": corr_matrix, "strong_pairs": strong_pairs}


def forecast_next(df: pd.DataFrame, time_col: str, target_col: str, periods: int = 3) -> Optional[List[float]]:
    """Forecast future values for ``target_col`` based on a simple linear trend."""
    try:
        # Drop rows with missing values in the relevant columns and sort by time
        series = df[[time_col, target_col]].dropna().sort_values(time_col)
        y = series[target_col].astype(float).values
        n = len(y)
        # Require at least 3 points to build a reasonable trend line
        if n < 3:
            return None
        x = np.arange(n)
        # Fit linear regression y = m*x + b
        m, b = np.polyfit(x, y, 1)
        # Forecast future values
        x_future = np.arange(n, n + periods)
        forecast = m * x_future + b
        return [float(v) for v in forecast]
    except Exception:
        return None


def generate_category_charts(df: pd.DataFrame) -> list:
    """Generate bar charts for categorical columns with low cardinality."""
    charts: list = []
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    chart_limit = 6
    for cat_col in cat_cols:
        uniq_count = df[cat_col].nunique(dropna=True)
        if uniq_count < 2 or uniq_count > 10:
            continue
        for num_col in numeric_cols:
            try:
                agg = df.groupby(cat_col)[num_col].sum().sort_values(ascending=False)
                if agg.empty:
                    continue
            except Exception:
                continue
            fig = px.bar(x=agg.index.astype(str), y=agg.values, labels={'x': cat_col, 'y': f"Sum of {num_col}"}, title=f"Sum of {num_col} by {cat_col}")
            fig.update_layout(height=400, margin=dict(l=30, r=30, t=50, b=30))
            charts.append(fig)
            break
        if len(charts) >= chart_limit:
            break
    return charts


def build_narrative(q_msgs: List[str], trends: List[Dict[str, Any]], forecasts: Dict[str, List[float]], corr_msgs: List[str]) -> str:
    """Assemble a narrative summary from quality issues, trends, forecasts and correlations."""
    parts: List[str] = []
    if q_msgs:
        parts.append("Data quality summary:")
        parts.extend(q_msgs)
    if trends:
        sorted_trends = sorted(trends, key=lambda t: abs(t['slope']), reverse=True)[:3]
        parts.append("Trend highlights:")
        parts.extend([t['comment'] for t in sorted_trends])
    if forecasts:
        parts.append("Forecasts:")
        for col, fc in forecasts.items():
            parts.append(f"Forecast for '{col}' over the next {len(fc)} periods: {', '.join(f'{v:.2f}' for v in fc)}")
    if corr_msgs:
        parts.append("Correlation insights:")
        parts.extend(corr_msgs)
    if not parts:
        return "No notable insights found."
    return "\n\n".join(parts)


def parse_uploaded_file(uploaded_file) -> Optional[pd.DataFrame]:
    """Read uploaded file into a DataFrame using Excel or CSV parsers."""
    if uploaded_file is None:
        return None
    contents = uploaded_file.read()
    name = uploaded_file.name or ""
    ext = name.split('.')[-1].lower()
    df = None
    # Try Excel if extension suggests it
    if ext in ["xlsx", "xlsm", "xlsb", "xls", "ods"]:
        try:
            xls = pd.ExcelFile(io.BytesIO(contents), engine="openpyxl")
        except Exception:
            try:
                xls = pd.ExcelFile(io.BytesIO(contents))
            except Exception:
                xls = None
        if xls is not None:
            best_sheet = None
            best_score = -1
            for sheet_name in xls.sheet_names:
                try:
                    tmp = xls.parse(sheet_name)
                except Exception:
                    continue
                num_numeric = tmp.select_dtypes(include=[np.number]).shape[1]
                n_rows = tmp.shape[0]
                score = num_numeric * n_rows
                if num_numeric > 0 and score > best_score:
                    best_score = score
                    best_sheet = sheet_name
            if best_sheet is None and xls.sheet_names:
                best_sheet = xls.sheet_names[0]
            if best_sheet is not None:
                try:
                    df = xls.parse(best_sheet)
                except Exception:
                    df = None
    if df is None and ext in ["xlsx", "xlsm", "xlsb", "xls", "ods"]:
        try:
            df = pd.read_excel(io.BytesIO(contents), engine="openpyxl")
        except Exception:
            df = None
    if df is None:
        # try CSV with utf-8 then latin-1
        try:
            df = pd.read_csv(io.BytesIO(contents), encoding="utf-8")
        except Exception:
            try:
                df = pd.read_csv(io.BytesIO(contents), encoding="latin1")
            except Exception:
                df = None
    if df is not None:
        df = df.reset_index(drop=True)
    return df


def main() -> None:
    st.set_page_config(page_title="Excel Insights AI", layout="wide")
    st.title("Excel Insights AI")
    st.write(
        "Upload an Excel or CSV file to receive detailed analysis, data quality diagnostics, trend and correlation insights, and forecasts."
    )
    uploaded_file = st.file_uploader("Upload Excel (.xlsx) or CSV file", type=["xlsx", "xls", "xlsm", "xlsb", "ods", "csv"])
    if uploaded_file is not None:
        df = parse_uploaded_file(uploaded_file)
        if df is None or df.empty:
            st.error("Unable to read the uploaded file or the file is empty.")
            return
        st.subheader("Raw Data Preview")
        st.dataframe(df.head())
        # Data summary
        descr = describe_dataframe(df)
        st.subheader("Summary Statistics")
        st.dataframe(descr)
        # Quality checks
        quality = data_quality_checks(df)
        q_msgs = quality_messages(quality)
        st.subheader("Data Quality Diagnostics")
        for m in q_msgs:
            st.write("- " + m)
        # Trend analysis
        time_col = detect_datetime_column(df)
        trends = compute_trends(df, time_col)
        if trends:
            st.subheader("Trend Analysis")
            for t in trends:
                st.write(f"{t['column']}: {t['direction']} (slope={t['slope']:.3f})")
        # Forecasts
        fc_results: Dict[str, List[float]] = {}
        if time_col:
            for col in df.select_dtypes(include=[np.number]).columns:
                fc = forecast_next(df, time_col, col)
                if fc:
                    fc_results[col] = fc
        if fc_results:
            st.subheader("Forecasts (Next 3 Periods)")
            for col, fc in fc_results.items():
                st.write(f"{col}: {', '.join(f'{v:.2f}' for v in fc)}")
        # Correlations
        corr_info = compute_correlations(df)
        corr_msgs: List[str] = []
        if corr_info:
            corr_mat = corr_info['matrix']
            strong_pairs = corr_info['strong_pairs']
            if not corr_mat.empty:
                st.subheader("Correlation Heatmap")
                fig = px.imshow(
                    corr_mat,
                    text_auto=".2f",
                    color_continuous_scale="RdBu",
                    zmin=-1.0,
                    zmax=1.0,
                    aspect="auto",
                )
                fig.update_layout(height=400, margin=dict(l=30, r=30, t=50, b=30))
                st.plotly_chart(fig, use_container_width=True)
            if strong_pairs:
                for a, b, val in strong_pairs:
                    relation = "positively" if val > 0 else "negatively"
                    corr_msgs.append(f"{a} and {b} are highly {relation} correlated (corr = {val:.2f}).")
        # Category charts
        cat_charts = generate_category_charts(df)
        if cat_charts:
            st.subheader("Category Summary Charts")
            for chart in cat_charts:
                st.plotly_chart(chart, use_container_width=True)
        # Narrative summary
        narrative = build_narrative(q_msgs, trends, fc_results, corr_msgs)
        st.subheader("Narrative Summary")
        st.write(narrative)


if __name__ == "__main__":
    main()
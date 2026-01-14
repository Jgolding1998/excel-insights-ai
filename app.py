"""
Main application for the Excel Insights AI website.

This FastAPI application provides a user interface that allows users to
upload Excel or CSV files and receive detailed analysis, data quality
diagnostics, trend detection, forecasting and narrative summaries
about their datasets. It is designed to run on a cloud hosting
platform such as Render and can be deployed via the accompanying
render.yaml file.

Key features:

* **Data ingestion** – reads Excel or CSV uploads into a pandas
  DataFrame, automatically attempting to detect a datetime column.
* **Descriptive statistics** – computes summary statistics for all
  numeric and object columns using ``DataFrame.describe()``.
* **Data quality checks** – identifies missing values, duplicate rows,
  negative numeric values and outliers (values beyond three standard
  deviations) to help users diagnose issues in their data. Generates
  human‑readable explanations for each problem.
* **Trend detection** – fits simple linear models to each numeric
  column against either time or row index to classify trends as
  increasing, decreasing or stable.
* **Forecasting** – when a time column and sufficient data are
  available, forecasts the next three periods for each numeric
  column using an ARIMA model from ``statsmodels``.
* **Interactive dashboards** – uses Plotly to create interactive
  charts (time series or histograms) for visual exploration of the
  dataset, which are embedded directly into the result page.
* **Narrative generation** – produces a concise narrative summary
  combining quality issues, trends and forecasts to give users an
  immediate overview of what the data shows and any problems
  encountered.

Dependencies are declared in the accompanying ``requirements.txt`` file
to ensure reproducibility.
"""

import io
import datetime
from typing import Optional, Dict, List, Any, Tuple

import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA

# Initialise FastAPI app and template directory
app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


def detect_datetime_column(df: pd.DataFrame) -> Optional[str]:
    """Attempt to find a column containing datetime information.

    In addition to checking for common substrings in the column name
    (e.g. "date", "time", "timestamp"), this function will try to
    convert **every** column to datetimes and select the first column
    where more than half of the values convert successfully. This
    allows detection of date columns even if they do not contain a
    descriptive name.

    Args:
        df: DataFrame to search.

    Returns:
        Name of the detected datetime column or ``None`` if none is
        identified.
    """
    # Prioritise columns that contain typical datetime keywords
    candidates = [c for c in df.columns if any(k in str(c).lower() for k in ["date", "time", "timestamp"])]
    for col in candidates + list(df.columns):
        # Skip columns with non-string convertible values entirely
        try:
            conv = pd.to_datetime(df[col], errors="coerce")
        except Exception:
            continue
        # At least half of the values must be parsed successfully
        if conv.notna().sum() >= len(df) / 2:
            return col
    return None


def describe_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame of summary statistics for all columns.

    Includes both numeric and object columns. ``DataFrame.describe``
    returns a DataFrame where the index contains statistic names and
    columns are features; transposing yields a more readable format.

    Args:
        df: Input DataFrame.

    Returns:
        A DataFrame with one row per column containing statistics.
    """
    descr = df.describe(include="all").T
    descr = descr.reset_index().rename(columns={"index": "column"})
    return descr


def data_quality_checks(df: pd.DataFrame) -> Dict[str, Any]:
    """Perform expanded data quality checks on the DataFrame.

    In addition to the basic checks (missing values, duplicate rows,
    negative values in numeric columns and z-score outliers), this
    function also identifies columns with a single unique value (i.e.
    constants) and categorical columns with extremely high cardinality
    (where nearly every value is unique). These additional checks help
    to flag uninformative features and potential join keys that may
    need special handling.

    Args:
        df: DataFrame to assess.

    Returns:
        Dictionary mapping check names to counts or per-column
        dictionaries. Keys may include 'missing', 'duplicates',
        'negatives', 'outliers', 'constants' and 'high_cardinality'.
    """
    quality: Dict[str, Any] = {}
    # Missing values per column
    missing = df.isna().sum()
    missing_filtered = missing[missing > 0].to_dict()
    if missing_filtered:
        quality['missing'] = missing_filtered
    # Duplicate rows across the entire DataFrame
    dup_count = int(df.duplicated().sum())
    if dup_count > 0:
        quality['duplicates'] = dup_count
    # Negative values in numeric columns
    negatives: Dict[str, int] = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        neg_count = int((df[col] < 0).sum())
        if neg_count > 0:
            negatives[col] = neg_count
    if negatives:
        quality['negatives'] = negatives
    # Outliers using z-score method (only if more than 10 values)
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
    # Constant columns (all values identical)
    constants = {col: df[col].iloc[0] for col in df.columns if df[col].nunique(dropna=False) == 1}
    if constants:
        quality['constants'] = constants
    # High cardinality categorical columns (unique values / rows > 0.7)
    high_card = {}
    for col in df.columns:
        # Only consider non-numeric columns for cardinality check
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        # Unique values as proportion of total rows
        uniq = df[col].nunique(dropna=True)
        if len(df) > 0 and uniq / len(df) > 0.7:
            high_card[col] = uniq
    if high_card:
        quality['high_cardinality'] = high_card
    return quality


def quality_messages(quality: Dict[str, Any]) -> List[str]:
    """Convert data quality check results into human‑readable messages.

    This helper function traverses the dictionary produced by
    ``data_quality_checks`` and composes explanatory sentences for
    each issue identified. Only checks present in the dictionary will
    generate messages.

    Args:
        quality: Dictionary of quality issues and their counts.

    Returns:
        A list of human‑readable strings describing each issue.
    """
    messages: List[str] = []
    # Missing values
    if 'missing' in quality:
        for col, count in quality['missing'].items():
            msg = f"Column '{col}' contains {count} missing value{'s' if count != 1 else ''}. Consider imputing or removing them."
            messages.append(msg)
    # Duplicates
    if 'duplicates' in quality:
        count = quality['duplicates']
        messages.append(f"The dataset contains {count} duplicate row{'s' if count != 1 else ''}. Consider removing duplicates to ensure accurate analyses.")
    # Negative values
    if 'negatives' in quality:
        for col, count in quality['negatives'].items():
            msg = f"Column '{col}' has {count} negative value{'s' if count != 1 else ''}. If negative values are unexpected for this field, verify the data."
            messages.append(msg)
    # Outliers
    if 'outliers' in quality:
        for col, count in quality['outliers'].items():
            msg = f"Column '{col}' contains {count} outlier{'s' if count != 1 else ''} beyond ±3 standard deviations. Investigate these anomalies."
            messages.append(msg)
    # Constant columns
    if 'constants' in quality:
        for col, value in quality['constants'].items():
            msg = f"Column '{col}' has a single unique value ('{value}'), which may not be informative. Consider dropping this column."
            messages.append(msg)
    # High cardinality
    if 'high_cardinality' in quality:
        for col, uniq in quality['high_cardinality'].items():
            msg = f"Column '{col}' has a very high number of unique values ({uniq}). It may represent an identifier or require encoding for modelling."
            messages.append(msg)
    if not messages:
        messages.append("No obvious data quality issues were detected.")
    return messages


def compute_trends(df: pd.DataFrame, time_col: Optional[str]) -> List[Dict[str, Any]]:
    """Compute linear trends for numeric columns.

    Fits a simple linear relationship between each numeric column and a
    normalised representation of time (if available) or row index. A
    slope above 0.1 indicates increasing trend; below –0.1 indicates
    decreasing; otherwise stable.

    Args:
        df: DataFrame containing the data.
        time_col: Name of the datetime column if detected, else None.

    Returns:
        List of dicts with 'column', 'slope', 'direction' and
        descriptive 'comment'.
    """
    results = []
    # create x values
    if time_col:
        try:
            x = pd.to_datetime(df[time_col], errors="coerce").map(datetime.datetime.toordinal).astype(float)
        except Exception:
            x = pd.to_numeric(df[time_col], errors="coerce")
    else:
        x = np.arange(len(df))
    # normalise
    x = (x - np.nanmean(x)) / (np.nanstd(x) if np.nanstd(x) != 0 else 1)
    for col in df.select_dtypes(include=[np.number]).columns:
        y = df[col].astype(float)
        mask = ~np.isnan(x) & ~np.isnan(y)
        if mask.sum() < 2:
            continue
        xx = x[mask]
        yy = y[mask]
        # compute slope using simple covariance/variance formula
        slope = float(np.cov(xx, yy, bias=True)[0, 1] / np.var(xx)) if np.var(xx) != 0 else 0.0
        if slope > 0.1:
            direction = "increasing"
        elif slope < -0.1:
            direction = "decreasing"
        else:
            direction = "stable"
        results.append({
            "column": col,
            "slope": slope,
            "direction": direction,
            "comment": f"Column '{col}' shows a {direction} trend over the observed period."
        })
    return results


def compute_correlations(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Compute pairwise correlations among numeric columns and identify strong relationships.

    For datasets with at least two numeric columns, a correlation matrix is
    calculated. Pairs with an absolute correlation coefficient above
    0.8 are considered "strong" and returned separately.

    Args:
        df: Input DataFrame.

    Returns:
        A dictionary with keys 'matrix' (DataFrame) and 'strong_pairs'
        (list of tuples) or ``None`` if fewer than two numeric columns
        are present.
    """
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


def generate_correlation_heatmap(corr_matrix: pd.DataFrame) -> str:
    """Generate an interactive Plotly heatmap for a correlation matrix."""
    import plotly.express as px
    # Use a diverging colour scale and fix the z-range to [-1, 1]
    fig = px.imshow(
        corr_matrix,
        text_auto=".2f",
        color_continuous_scale="RdBu",
        zmin=-1.0,
        zmax=1.0,
        aspect="auto",
    )
    fig.update_layout(title="Correlation heatmap", height=400, margin=dict(l=30, r=30, t=50, b=30))
    return fig.to_html(full_html=False, include_plotlyjs='cdn')


def generate_category_charts(df: pd.DataFrame) -> List[str]:
    """Generate bar charts for categorical columns with low cardinality.

    For each non-numeric column with between 2 and 10 unique values,
    this function groups the data by that column and computes the sum
    of each numeric column. A bar chart is created for the first
    numeric column encountered per category. Only a limited number of
    charts (max 6) are produced to avoid overwhelming the user.

    Args:
        df: Input DataFrame.

    Returns:
        A list of Plotly chart HTML strings.
    """
    charts: List[str] = []
    import plotly.express as px
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    # Limit total charts to six to maintain readability
    chart_limit = 6
    for cat_col in cat_cols:
        uniq_count = df[cat_col].nunique(dropna=True)
        if uniq_count < 2 or uniq_count > 10:
            continue
        # Choose the first numeric column to summarise
        for num_col in numeric_cols:
            # Compute group sum and sort
            try:
                agg = df.groupby(cat_col)[num_col].sum().sort_values(ascending=False)
                # Skip if there are too many categories to meaningfully display
                if agg.empty:
                    continue
            except Exception:
                continue
            fig = px.bar(
                x=agg.index.astype(str),
                y=agg.values,
                labels={'x': cat_col, 'y': f"Sum of {num_col}"},
                title=f"Sum of {num_col} by {cat_col}"
            )
            fig.update_layout(height=400, margin=dict(l=30, r=30, t=50, b=30))
            charts.append(fig.to_html(full_html=False, include_plotlyjs='cdn'))
            # Break after first numeric col for this cat col
            break
        if len(charts) >= chart_limit:
            break
    return charts


def forecast_next(df: pd.DataFrame, time_col: str, target_col: str, periods: int = 3) -> Optional[List[float]]:
    """Forecast future values using a simple ARIMA(1,1,0) model.

    If there are fewer than 12 observations or the model fails, returns
    ``None``.
    """
    try:
        series = df[[time_col, target_col]].dropna()
        series = series.sort_values(time_col)
        y = series[target_col].astype(float).values
        if len(y) < 12:
            return None
        model = ARIMA(y, order=(1, 1, 0))
        fit = model.fit()
        fc = fit.forecast(steps=periods)
        return [float(v) for v in fc]
    except Exception:
        return None


def generate_plots(df: pd.DataFrame, time_col: Optional[str]) -> List[str]:
    """Create Plotly chart HTML strings for each numeric column."""
    charts = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if time_col:
        try:
            times = pd.to_datetime(df[time_col], errors="coerce")
        except Exception:
            times = df[time_col]
        for col in numeric_cols:
            fig = px.line(x=times, y=df[col], title=f"{col} over time")
            fig.update_layout(height=400, margin=dict(l=30, r=30, t=50, b=30))
            charts.append(fig.to_html(full_html=False, include_plotlyjs='cdn'))
    else:
        for col in numeric_cols:
            fig = px.histogram(df, x=col, nbins=30, title=f"Distribution of {col}")
            fig.update_layout(height=400, margin=dict(l=30, r=30, t=50, b=30))
            charts.append(fig.to_html(full_html=False, include_plotlyjs='cdn'))
    return charts


def build_narrative(quality_msgs: List[str], trends: List[Dict[str, Any]], forecasts: Dict[str, List[float]]) -> str:
    """Assemble a narrative summary from quality issues, trends and forecasts.

    The narrative provides a structured description of the key
    observations in the dataset. It combines data quality warnings,
    notable trends (based on the most extreme slopes) and concise
    forecasts for each numeric column. Additional narrative elements
    such as correlations or category insights can be appended by the
    caller if desired.

    Args:
        quality_msgs: List of messages returned from quality checks.
        trends: List of trend dictionaries produced by ``compute_trends``.
        forecasts: Mapping of column names to forecast arrays.

    Returns:
        A multi‑line string summarising the findings.
    """
    parts: List[str] = []
    # Data quality section
    if quality_msgs:
        parts.append("Data quality summary:")
        parts.extend(quality_msgs)
    # Trends section (select up to 3 most extreme trends by absolute slope)
    if trends:
        sorted_trends = sorted(trends, key=lambda t: abs(t['slope']), reverse=True)[:3]
        parts.append("Trend highlights:")
        parts.extend([t['comment'] for t in sorted_trends])
    # Forecasts section
    if forecasts:
        parts.append("Forecasts:")
        for col, fc in forecasts.items():
            forecast_str = ", ".join(f"{v:.2f}" for v in fc)
            parts.append(f"Forecast for '{col}' over the next {len(fc)} periods: {forecast_str}")
    if not parts:
        return "No notable insights found."
    return "\n".join(parts)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Render the upload page."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload", response_class=HTMLResponse)
async def upload(request: Request, file: UploadFile):
    """Handle file uploads and return analysis results."""
    contents = await file.read()
    # Read file into DataFrame; try Excel then CSV
    try:
        df = pd.read_excel(io.BytesIO(contents))
    except Exception:
        try:
            df = pd.read_csv(io.BytesIO(contents))
        except Exception as e:
            return templates.TemplateResponse("error.html", {"request": request, "message": f"Could not read the uploaded file: {e}"})
    # Reset index to ensure continuous integer index
    df = df.reset_index(drop=True)
    # Detect datetime column
    time_col = detect_datetime_column(df)
    # Summary statistics
    descr = describe_dataframe(df)
    descr_html = descr.to_html(classes="table table-striped table-sm", index=False)
    # Data quality diagnostics
    quality = data_quality_checks(df)
    quality_msgs = quality_messages(quality)
    # Trend analysis
    trends = compute_trends(df, time_col)
    # Forecasts for numeric columns if a datetime column was detected
    fc_results: Dict[str, List[float]] = {}
    if time_col:
        for col in df.select_dtypes(include=[np.number]).columns:
            fc = forecast_next(df, time_col, col)
            if fc:
                fc_results[col] = fc
    # Correlation analysis (matrix and strong pairs)
    corr_info = compute_correlations(df)
    corr_heatmap = None
    correlation_messages: List[str] = []
    if corr_info:
        corr_heatmap = generate_correlation_heatmap(corr_info["matrix"])
        # Build messages for strongly correlated pairs
        for a, b, val in corr_info["strong_pairs"]:
            relation = "positively" if val > 0 else "negatively"
            correlation_messages.append(f"Columns '{a}' and '{b}' are highly {relation} correlated (corr = {val:.2f}).")
    # Category summary charts
    category_plots = generate_category_charts(df)
    # Narrative summary
    narrative = build_narrative(quality_msgs, trends, fc_results)
    # Append correlation messages to the narrative (optional)
    if correlation_messages:
        narrative += "\n\nCorrelation insights:\n" + "\n".join(correlation_messages)
    # Build interactive charts: general numeric plots, correlation heatmap, category plots
    charts = []
    charts.extend(generate_plots(df, time_col))
    if corr_heatmap:
        charts.append(corr_heatmap)
    charts.extend(category_plots)
    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "descr_table": descr_html,
            "quality_messages": quality_msgs,
            "trends": trends,
            "forecasts": fc_results,
            "narrative": narrative,
            "correlation_messages": correlation_messages,
            "plots": charts,
        },
    )
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
from typing import Optional, Dict, List, Any

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

    This function looks for columns whose names include keywords such
    as 'date', 'time' or 'timestamp'. It then tries to convert the
    column to datetimes and selects the first column where at least
    half of the values convert successfully.

    Args:
        df: DataFrame to search.

    Returns:
        Name of the detected datetime column or ``None``.
    """
    candidates = [c for c in df.columns if any(k in c.lower() for k in ["date", "time", "timestamp"])]
    for col in candidates + list(df.columns):
        try:
            conv = pd.to_datetime(df[col], errors="coerce")
            if conv.notna().sum() >= len(df) / 2:
                return col
        except Exception:
            continue
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
    """Perform basic data quality checks on the DataFrame.

    Checks performed include:
    - Missing values per column
    - Number of duplicate rows
    - Negative values in numeric columns
    - Outliers beyond 3 standard deviations in numeric columns

    Args:
        df: DataFrame to assess.

    Returns:
        Dictionary with keys 'missing', 'duplicates', 'negatives',
        'outliers' containing counts per column or overall counts.
    """
    quality = {}
    # Missing values
    missing = df.isna().sum()
    quality['missing'] = missing[missing > 0].to_dict()
    # Duplicate rows
    dup_count = int(df.duplicated().sum())
    if dup_count > 0:
        quality['duplicates'] = dup_count
    # Negative values in numeric columns
    negatives = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        neg_count = int((df[col] < 0).sum())
        if neg_count > 0:
            negatives[col] = neg_count
    if negatives:
        quality['negatives'] = negatives
    # Outliers using z-score method (only if more than 10 values)
    outliers = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        series = df[col].dropna().astype(float)
        if len(series) < 10:
            continue
        zscores = (series - series.mean()) / series.std(ddof=0)
        out_count = int((abs(zscores) > 3).sum())
        if out_count > 0:
            outliers[col] = out_count
    if outliers:
        quality['outliers'] = outliers
    return quality


def quality_messages(quality: Dict[str, Any]) -> List[str]:
    """Convert quality check results into human‑readable messages."""
    messages = []
    if 'missing' in quality:
        for col, count in quality['missing'].items():
            messages.append(f"Column '{col}' contains {count} missing value{'s' if count > 1 else ''}. Consider imputing or removing them.")
    if 'duplicates' in quality:
        messages.append(f"The dataset contains {quality['duplicates']} duplicate row{'s' if quality['duplicates'] != 1 else ''}. Consider dropping duplicates.")
    if 'negatives' in quality:
        for col, count in quality['negatives'].items():
            messages.append(f"Column '{col}' has {count} negative value{'s' if count > 1 else ''}. If negative values are unexpected for this field, verify the data.")
    if 'outliers' in quality:
        for col, count in quality['outliers'].items():
            messages.append(f"Column '{col}' contains {count} outlier{'s' if count > 1 else ''} beyond ±3 standard deviations. Investigate these anomalies.")
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
    """Assemble a narrative summary from quality issues, trends and forecasts."""
    parts = []
    # Quality issues
    if quality_msgs:
        parts.append("Data quality summary:")
        parts.extend(quality_msgs)
    # Trends: pick up to 3 most extreme slopes to mention
    if trends:
        # sort by absolute slope magnitude
        sorted_trends = sorted(trends, key=lambda t: abs(t['slope']), reverse=True)[:3]
        trend_sentences = [f"{t['comment']}" for t in sorted_trends]
        parts.append("Trend highlights:")
        parts.extend(trend_sentences)
    # Forecasts summary
    if forecasts:
        forecast_msgs = []
        for col, fc in forecasts.items():
            forecast_str = ", ".join(f"{v:.2f}" for v in fc)
            forecast_msgs.append(f"Forecast for '{col}' over the next {len(fc)} periods: {forecast_str}")
        parts.append("Forecasts:")
        parts.extend(forecast_msgs)
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
    # Forecasts
    fc_results: Dict[str, List[float]] = {}
    if time_col:
        for col in df.select_dtypes(include=[np.number]).columns:
            fc = forecast_next(df, time_col, col)
            if fc:
                fc_results[col] = fc
    # Narrative summary
    narrative = build_narrative(quality_msgs, trends, fc_results)
    # Charts
    plots = generate_plots(df, time_col)
    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "descr_table": descr_html,
            "quality_messages": quality_msgs,
            "trends": trends,
            "forecasts": fc_results,
            "narrative": narrative,
            "plots": plots,
        },
    )
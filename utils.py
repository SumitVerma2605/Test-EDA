"""
StatLab Pro — Utility Functions
All statistical, preprocessing, and ML helper functions
"""

import io
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import (
    ttest_ind, ttest_rel, f_oneway, chi2_contingency,
    shapiro, pearsonr, spearmanr, t as t_dist,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, precision_score, recall_score
import requests

# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_from_github(url: str):
    """
    Load a CSV or Excel file from a GitHub raw URL.
    Returns (DataFrame, error_message). On success, error_message is None.
    """
    try:
        # Ensure we use raw content URL
        if "github.com" in url and "raw.githubusercontent.com" not in url:
            url = url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")

        response = requests.get(url, timeout=30)
        response.raise_for_status()

        if url.lower().endswith(".csv"):
            df = pd.read_csv(io.StringIO(response.text))
        elif url.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(response.content))
        else:
            # Try CSV as default
            df = pd.read_csv(io.StringIO(response.text))

        return df, None
    except requests.exceptions.ConnectionError:
        return None, "Connection error. Please check the URL and your internet connection."
    except requests.exceptions.HTTPError as e:
        return None, f"HTTP error: {e}"
    except Exception as e:
        return None, f"Failed to load dataset: {e}"


# ─── Preprocessing ────────────────────────────────────────────────────────────

def handle_missing_values(df: pd.DataFrame, strategy: str, columns: list) -> pd.DataFrame:
    """Handle missing values according to the chosen strategy."""
    df = df.copy()
    if strategy == "Drop rows":
        df = df.dropna(subset=columns)
    elif strategy == "Mean imputation":
        for col in columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].mean())
    elif strategy == "Median imputation":
        for col in columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
    elif strategy == "Mode imputation":
        for col in columns:
            df[col] = df[col].fillna(df[col].mode()[0])
    elif strategy == "Forward fill":
        df[columns] = df[columns].ffill()
    elif strategy == "Backward fill":
        df[columns] = df[columns].bfill()
    return df


def convert_dtypes(df: pd.DataFrame, column: str, target_type: str):
    """
    Convert a column to the target dtype.
    Returns (df, error) tuple.
    """
    df = df.copy()
    try:
        if target_type == "datetime64":
            df[column] = pd.to_datetime(df[column], infer_datetime_format=True)
        elif target_type == "category":
            df[column] = df[column].astype("category")
        elif target_type == "str":
            df[column] = df[column].astype(str)
        else:
            df[column] = df[column].astype(target_type)
        return df, None
    except Exception as e:
        return df, str(e)


def detect_outliers_iqr(df: pd.DataFrame, column: str, multiplier: float = 1.5):
    """
    Detect outliers using the IQR method.
    Returns (outlier_df, (lower_bound, upper_bound)).
    """
    q1, q3 = df[column].quantile(0.25), df[column].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    mask = (df[column] < lower) | (df[column] > upper)
    return df[mask], (lower, upper)


def detect_outliers_zscore(df: pd.DataFrame, column: str, threshold: float = 3.0):
    """
    Detect outliers using Z-score method.
    Returns (outlier_df, z_scores_series).
    """
    z_scores = np.abs(stats.zscore(df[column].dropna()))
    z_series = pd.Series(z_scores, index=df[column].dropna().index)
    mask = z_series > threshold
    return df.loc[mask.index[mask]], z_series


# ─── EDA ──────────────────────────────────────────────────────────────────────

def summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute extended summary statistics including skewness and kurtosis.
    """
    desc = df.describe().T
    desc["skewness"] = df.skew()
    desc["kurtosis"] = df.kurtosis()
    desc["missing"] = df.isnull().sum()
    desc["missing_%"] = (df.isnull().sum() / len(df) * 100).round(2)
    return desc


def compute_correlation(df: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    """Compute correlation matrix for numeric columns."""
    num = df.select_dtypes(include=np.number)
    return num.corr(method=method)


# ─── Statistical Tests ────────────────────────────────────────────────────────

def run_shapiro_wilk(df: pd.DataFrame, column: str, alpha: float = 0.05) -> dict:
    """Shapiro-Wilk normality test."""
    data = df[column].dropna()
    stat, p = shapiro(data)
    normal = p >= alpha
    return {
        "test_name": "Shapiro-Wilk Test",
        "statistic": stat,
        "p_value": p,
        "interpretation": (
            f"Data in '{column}' appears normally distributed (fail to reject H₀)."
            if normal else
            f"Data in '{column}' is NOT normally distributed (reject H₀)."
        ),
    }


def run_ttest_independent(df, column, group_col, g1, g2, equal_var, alpha):
    """Independent samples t-test."""
    s1 = df[df[group_col] == g1][column].dropna()
    s2 = df[df[group_col] == g2][column].dropna()
    stat, p = ttest_ind(s1, s2, equal_var=equal_var)
    reject = p < alpha
    return {
        "test_name": "Independent Samples t-test",
        "statistic": stat,
        "p_value": p,
        "interpretation": (
            f"Significant difference between '{g1}' and '{g2}' on '{column}' (reject H₀)."
            if reject else
            f"No significant difference between '{g1}' and '{g2}' on '{column}' (fail to reject H₀)."
        ),
        "extras": {
            f"Mean ({g1})": f"{s1.mean():.4f}",
            f"Mean ({g2})": f"{s2.mean():.4f}",
            f"n ({g1})": len(s1),
            f"n ({g2})": len(s2),
        },
    }


def run_ttest_paired(df, col1, col2, alpha):
    """Paired samples t-test."""
    s1 = df[col1].dropna()
    s2 = df[col2].dropna()
    n = min(len(s1), len(s2))
    stat, p = ttest_rel(s1[:n], s2[:n])
    reject = p < alpha
    return {
        "test_name": "Paired Samples t-test",
        "statistic": stat,
        "p_value": p,
        "interpretation": (
            f"Significant difference between '{col1}' and '{col2}' (reject H₀)."
            if reject else
            f"No significant difference between '{col1}' and '{col2}' (fail to reject H₀)."
        ),
        "extras": {
            f"Mean ({col1})": f"{s1.mean():.4f}",
            f"Mean ({col2})": f"{s2.mean():.4f}",
            "Mean difference": f"{(s1[:n] - s2[:n]).mean():.4f}",
        },
    }


def run_anova(df, column, group_col, alpha):
    """One-way ANOVA."""
    groups = [df[df[group_col] == g][column].dropna().values for g in df[group_col].unique()]
    stat, p = f_oneway(*groups)
    reject = p < alpha
    return {
        "test_name": "One-Way ANOVA",
        "statistic": stat,
        "p_value": p,
        "interpretation": (
            f"Significant difference exists among groups in '{group_col}' for '{column}' (reject H₀)."
            if reject else
            f"No significant difference among groups in '{group_col}' for '{column}' (fail to reject H₀)."
        ),
        "extras": {"Groups": str(df[group_col].nunique()), "Total N": str(len(df))},
    }


def run_chi_square(df, col1, col2, alpha):
    """Chi-square test of independence."""
    ct = pd.crosstab(df[col1], df[col2])
    stat, p, dof, expected = chi2_contingency(ct)
    reject = p < alpha
    return {
        "test_name": "Chi-Square Test of Independence",
        "statistic": stat,
        "p_value": p,
        "interpretation": (
            f"Significant association between '{col1}' and '{col2}' (reject H₀)."
            if reject else
            f"No significant association between '{col1}' and '{col2}' (fail to reject H₀)."
        ),
        "extras": {"Degrees of freedom": str(dof)},
    }


def run_pearson(df, col1, col2, alpha):
    """Pearson correlation test."""
    data = df[[col1, col2]].dropna()
    r, p = pearsonr(data[col1], data[col2])
    reject = p < alpha
    return {
        "test_name": "Pearson Correlation",
        "statistic": r,
        "p_value": p,
        "interpretation": (
            f"Significant linear correlation (r = {r:.4f}) between '{col1}' and '{col2}' (reject H₀)."
            if reject else
            f"No significant linear correlation between '{col1}' and '{col2}' (fail to reject H₀)."
        ),
        "extras": {"r²": f"{r**2:.4f}", "n": str(len(data))},
    }


def run_spearman(df, col1, col2, alpha):
    """Spearman rank correlation test."""
    data = df[[col1, col2]].dropna()
    r, p = spearmanr(data[col1], data[col2])
    reject = p < alpha
    return {
        "test_name": "Spearman Correlation",
        "statistic": r,
        "p_value": p,
        "interpretation": (
            f"Significant monotonic correlation (ρ = {r:.4f}) between '{col1}' and '{col2}' (reject H₀)."
            if reject else
            f"No significant monotonic correlation between '{col1}' and '{col2}' (fail to reject H₀)."
        ),
        "extras": {"n": str(len(data))},
    }


def compute_confidence_interval(df, column, confidence=0.95):
    """Compute confidence interval for a column mean."""
    data = df[column].dropna()
    n = len(data)
    mean = data.mean()
    se = stats.sem(data)
    h = se * t_dist.ppf((1 + confidence) / 2, n - 1)
    return {"mean": mean, "se": se, "lower": mean - h, "upper": mean + h, "n": n}


# ─── ML Models ────────────────────────────────────────────────────────────────

def run_linear_regression(df, features, target):
    """Fit a linear regression model and return metrics."""
    try:
        data = df[features + [target]].dropna()
        X = data[features].values
        y = data[target].values
        model = LinearRegression()
        model.fit(X, y)
        preds = model.predict(X)
        residuals = y - preds
        r2 = r2_score(y, preds)
        n, p = len(y), len(features)
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        rmse = np.sqrt(mean_squared_error(y, preds))
        return {
            "r2": r2,
            "adj_r2": adj_r2,
            "rmse": rmse,
            "intercept": model.intercept_,
            "coefs": model.coef_,
            "predictions": preds,
            "residuals": residuals,
        }
    except Exception as e:
        return {"error": str(e)}


def run_logistic_regression(df, features, target):
    """Fit a logistic regression model and return metrics."""
    try:
        data = df[features + [target]].dropna()
        X = data[features].values
        le = LabelEncoder()
        y = le.fit_transform(data[target])
        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)
        preds = model.predict(X)
        return {
            "accuracy": accuracy_score(y, preds),
            "precision": precision_score(y, preds, zero_division=0),
            "recall": recall_score(y, preds, zero_division=0),
            "coefs": model.coef_[0],
        }
    except Exception as e:
        return {"error": str(e)}


# ─── Export ───────────────────────────────────────────────────────────────────

def generate_report_csv(df: pd.DataFrame) -> bytes:
    """Generate a statistical summary report as CSV bytes."""
    num = df.select_dtypes(include=np.number)
    report = summary_statistics(num)
    return report.to_csv().encode("utf-8")


def generate_report_pdf(df: pd.DataFrame):
    """
    Generate a PDF statistical report using reportlab.
    Returns bytes or None if reportlab is not installed.
    """
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib import colors

        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=60, bottomMargin=40)
        styles = getSampleStyleSheet()
        elements = []

        # Title
        elements.append(Paragraph("Statistical Analysis Report — StatLab Pro", styles["Title"]))
        elements.append(Spacer(1, 12))
        elements.append(Paragraph(f"Dataset: {df.shape[0]} rows × {df.shape[1]} columns", styles["Normal"]))
        elements.append(Spacer(1, 12))

        # Summary statistics table
        num = df.select_dtypes(include=np.number)
        if not num.empty:
            elements.append(Paragraph("Summary Statistics", styles["Heading2"]))
            elements.append(Spacer(1, 6))
            stats_df = num.describe().T.round(4).reset_index()
            stats_df.columns = ["Column"] + list(stats_df.columns[1:])
            table_data = [stats_df.columns.tolist()] + stats_df.values.tolist()
            t = Table(table_data, repeatRows=1)
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e2230")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                ("PADDING", (0, 0), (-1, -1), 4),
            ]))
            elements.append(t)

        doc.build(elements)
        return buf.getvalue()
    except ImportError:
        return None

#!/usr/bin/env python3
"""Retail analytics: ingestion, EDA, time-aware sales prediction, and reporting."""
from __future__ import annotations

import argparse
import re
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore", category=FutureWarning)

ALIASES = {
    "order_date": ["order date", "date", "transaction date", "sales date"],
    "sales": ["sales", "revenue", "amount", "sales amount", "net sales"],
    "quantity": ["quantity", "qty", "units", "units sold"],
    "discount": ["discount", "discount percent", "discount %"],
    "profit": ["profit", "profit amount", "margin amount"],
    "order_id": ["order id", "order_id", "transaction id", "invoice id"],
    "store": ["store", "store name", "store id", "branch"],
    "region": ["region", "sales region"],
    "state": ["state", "province", "store state"],
    "category": ["category", "product category", "department"],
    "sub_category": ["sub category", "sub-category", "sub_category", "subcategory"],
    "product": ["product", "product name", "sku", "product id"],
}

def norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(name).lower()).strip()

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    normalized = {norm(c): c for c in df.columns}
    for target, aliases in ALIASES.items():
        found = next((normalized[a] for a in aliases if a in normalized), None)
        if found and found != target:
            rename[found] = target
    return df.rename(columns=rename)

def load_data(input_dir: Path) -> pd.DataFrame:
    files = sorted(input_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")
    frames = [standardize_columns(pd.read_csv(f)) for f in files]
    sales_frame = next((x for x in frames if {"order_date", "sales"}.issubset(x.columns)), None)
    if sales_frame is None:
        raise ValueError("A CSV must contain an order date and sales/revenue column.")
    for dim in frames:
        if dim is sales_frame:
            continue
        shared = [c for c in dim.columns if c.endswith("_id") and c in sales_frame.columns]
        if shared:
            sales_frame = sales_frame.merge(dim, on=shared, how="left", suffixes=("", "_dim"))
    return sales_frame

def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df = df.dropna(subset=["order_date", "sales"])
    for col in ["sales", "quantity", "discount", "profit"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["sales"] = df["sales"].fillna(df["sales"].median()).clip(lower=0)
    if "quantity" in df:
        df["quantity"] = df["quantity"].fillna(df["quantity"].median()).clip(lower=0)
    if "discount" in df:
        df["discount"] = df["discount"].fillna(0)
        # Per-value scale detection: values > 1 treated as percentage (0-100), others as decimal (0-1)
        mask = df["discount"] > 1
        if mask.any():
            df.loc[mask, "discount"] = df.loc[mask, "discount"] / 100
    df["year"] = df.order_date.dt.year
    df["month"] = df.order_date.dt.month
    df["quarter"] = df.order_date.dt.quarter
    df["day_of_week"] = df.order_date.dt.dayofweek
    df["is_weekend"] = (df.day_of_week >= 5).astype(int)
    df["order_year_month"] = df.order_date.dt.to_period("M").astype(str)
    # Public Superstore data has no store master. State is a transparent location proxy.
    if "store" not in df and "state" in df:
        df["store"] = df["state"]
    return df.sort_values("order_date").reset_index(drop=True)

def iqr_outliers(series: pd.Series) -> int:
    q1, q3 = series.quantile([.25, .75]); iqr = q3 - q1
    return int(((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum())

def write_svg_chart(data: pd.DataFrame, label: str, value: str, title: str, path: Path, kind: str = "bar"):
    """Write dependency-free SVG visuals for portability and Power BI documentation."""
    w, h, pad = 1000, 500, 65
    values = data[value].astype(float).to_numpy()
    maximum = max(values.max(), 1)
    labels = data[label].astype(str).to_list()
    pieces = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}"><rect width="100%" height="100%" fill="white"/><text x="{pad}" y="32" font-family="Arial" font-size="22" font-weight="bold">{title}</text><line x1="{pad}" y1="{h-pad}" x2="{w-pad}" y2="{h-pad}" stroke="#555"/>']
    if kind == "line":
        n = len(values)
        if n == 1:
            # Single point: center it horizontally
            x = w / 2
        else:
            x = None
        coords = [(pad + (i * (w - 2 * pad) / (n - 1) if x is None else x), h-pad - v / maximum * (h-2*pad)) for i, v in enumerate(values)]
        pieces.append('<polyline fill="none" stroke="#2563eb" stroke-width="3" points="' + ' '.join(f'{x:.1f},{y:.1f}' for x,y in coords) + '"/>')
        pieces.extend(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="#2563eb"/><text x="{x:.1f}" y="{h-pad+18}" text-anchor="end" transform="rotate(-45 {x:.1f} {h-pad+18})" font-family="Arial" font-size="10">{lab}</text>' for (x,y),lab in zip(coords, labels))
    else:
        bw = (w - 2*pad) / max(len(values), 1) * .65
        for i, (v, lab) in enumerate(zip(values, labels)):
            x = pad + i * (w - 2*pad) / len(values) + bw*.15
            bh = v / maximum * (h-2*pad)
            y = h-pad-bh
            pieces.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{bh:.1f}" fill="#0f766e"/><text x="{x+bw/2:.1f}" y="{h-pad+18}" text-anchor="end" transform="rotate(-45 {x+bw/2:.1f} {h-pad+18})" font-family="Arial" font-size="11">{lab}</text>')
    path.write_text(''.join(pieces) + '</svg>')

def save_eda(df: pd.DataFrame, figure_dir: Path) -> dict:
    figure_dir.mkdir(parents=True, exist_ok=True)
    monthly = df.groupby("order_year_month", as_index=False).sales.sum()
    # Sort chronologically by parsing year-month
    monthly["sort_key"] = pd.to_datetime(monthly["order_year_month"], format="%Y-%m", errors="coerce")
    monthly = monthly.sort_values("sort_key").drop(columns="sort_key")
    write_svg_chart(monthly, "order_year_month", "sales", "Monthly Revenue Trend", figure_dir / "monthly_revenue.svg", "line")
    if "category" in df:
        top = df.groupby("category", as_index=False).sales.sum().nlargest(12, "sales")
        write_svg_chart(top, "category", "sales", "Revenue by Category", figure_dir / "category_revenue.svg")
    if "store" in df:
        top = df.groupby("store", as_index=False).sales.sum().nlargest(12, "sales")
        write_svg_chart(top, "store", "sales", "Revenue by State (Store Proxy)", figure_dir / "store_revenue.svg")
    return {"mean_sales": df.sales.mean(), "median_sales": df.sales.median(), "std_sales": df.sales.std(), "sales_iqr_outliers": iqr_outliers(df.sales)}

def build_model(df: pd.DataFrame, output: Path) -> pd.DataFrame:
    dates = np.sort(df.order_date.dropna().unique())
    if len(dates) < 2:
        raise ValueError(f"Need at least 2 unique dates for chronological split, got {len(dates)}.")
    cutoff = dates[max(1, int(len(dates) * .8)) - 1]
    train, test = df[df.order_date <= cutoff].copy(), df[df.order_date > cutoff].copy()
    if test.empty or len(train) < 10:
        raise ValueError(f"Need at least 10 historical rows and a later test period. Got train={len(train)}, test={len(test)}.")
    # Use only information available at order time; exclude post-sale profit and arbitrary identifiers.
    candidate_features = ["quantity", "discount", "year", "month", "quarter", "day_of_week", "is_weekend", "category", "sub_category", "segment", "region", "store", "ship_mode"]
    features = [c for c in candidate_features if c in df and df[c].nunique(dropna=True) > 1]
    numeric = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in features if c not in numeric]
    pre = ColumnTransformer([
        ("number", Pipeline([("impute", SimpleImputer(strategy="median"))]), numeric),
        ("category", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=True))]), categorical),
    ])
    # Fit preprocessor once on training data
    X_train = pre.fit_transform(train[features])
    X_test = pre.transform(test[features])

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=12, min_samples_leaf=5, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=42, n_jobs=-1),
        "Neural Network": MLPRegressor(hidden_layer_sizes=(64, 32), early_stopping=True, max_iter=300, random_state=42),
    }
    results, predictions = [], {}
    for name, estimator in models.items():
        if name == "Neural Network":
            # Dense scaling for neural network: center and scale features
            scaler = StandardScaler(with_mean=True)
            X_train_scaled = scaler.fit_transform(X_train.toarray() if hasattr(X_train, "toarray") else X_train)
            X_test_scaled = scaler.transform(X_test.toarray() if hasattr(X_test, "toarray") else X_test)
            estimator.fit(X_train_scaled, train.sales)
            pred = np.maximum(0, estimator.predict(X_test_scaled))
        else:
            estimator.fit(X_train, train.sales)
            pred = np.maximum(0, estimator.predict(X_test))
        predictions[name] = pred
        # MAPE: handle zero sales by excluding them from mean, fallback to inf if all zero
        y_true = test.sales.values
        y_pred = pred
        nonzero_mask = y_true != 0
        if nonzero_mask.any():
            mape = np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask]).mean() * 100
        else:
            mape = np.inf
        wmape = np.abs(y_true - y_pred).sum() / max(y_true.sum(), 1) * 100
        results.append({"model": name, "MAE": mean_absolute_error(y_true, y_pred), "RMSE": mean_squared_error(y_true, y_pred) ** .5, "MAPE_pct": mape, "WMAPE_pct": wmape, "R2": r2_score(y_true, y_pred)})
    metrics = pd.DataFrame(results).sort_values("RMSE")
    metrics.to_csv(output / "model_comparison.csv", index=False)
    best = metrics.iloc[0].model
    test["predicted_sales"] = predictions[best]
    test["forecast_error"] = test.sales - test.predicted_sales
    test["absolute_percentage_error"] = np.where(test.sales != 0, np.abs(test.forecast_error / test.sales) * 100, np.nan)
    test.to_csv(output / "test_forecasts.csv", index=False)
    for dimension in ["store", "category"]:
        if dimension in test:
            grp = test.groupby(dimension, dropna=False).agg(
                actual_sales=("sales", "sum"),
                predicted_sales=("predicted_sales", "sum"),
                MAE=("forecast_error", lambda x: x.abs().mean()),
                MAPE_pct=("absolute_percentage_error", lambda x: x.dropna().mean() if x.dropna().any() else np.nan),
            ).reset_index()
            grp.to_csv(output / f"forecast_error_by_{dimension}.csv", index=False)
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/processed"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    df = clean_and_engineer(load_data(args.input))
    df.to_csv(args.output / "clean_sales.csv", index=False)
    figures_dir = args.output.parent / "figures"
    stats = save_eda(df, figures_dir)
    metrics = build_model(df, args.output)
    report_path = args.output.parent / "business_action_report.md"
    report_path.write_text(f"# Executive business action report\n\n- Clean transactions: **{len(df):,}**\n- Revenue: **{df.sales.sum():,.2f}**\n- Mean / median sale: **{stats['mean_sales']:,.2f} / {stats['median_sales']:,.2f}**\n- Sales IQR outliers: **{stats['sales_iqr_outliers']:,}**\n- Best held-out model: **{metrics.iloc[0].model}** (RMSE {metrics.iloc[0].RMSE:,.2f}; weighted MAPE {metrics.iloc[0].WMAPE_pct:.1f}%)\n\n## Actions\n\n1. Use category/store error files to flag locations needing local demand rules or better data.\n2. Set replenishment buffers higher for volatile, high-revenue categories.\n3. Review promotions where discounts are associated with weak profit contribution.\n4. Refresh forecasts on a fixed cadence and monitor weighted MAPE by category and store.\n")
    print(f"Complete. Outputs written to {args.output}")
if __name__ == "__main__": main()

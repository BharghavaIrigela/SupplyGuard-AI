import numpy as np
import pandas as pd
from xgboost import XGBRegressor


def prepare_features(df):
    df = df.copy()

    df["WEEK_DATE"] = pd.to_datetime(df["WEEK_DATE"])
    df = df.sort_values(["PART_ID", "WEEK_DATE"])

    df["WEEK_NUMBER"] = df["WEEK_DATE"].dt.isocalendar().week.astype(int)
    df["MONTH"] = df["WEEK_DATE"].dt.month
    df["QUARTER"] = df["WEEK_DATE"].dt.quarter
    df["YEAR"] = df["WEEK_DATE"].dt.year

    df["LAG_1"] = df.groupby("PART_ID")["DEMAND"].shift(1)
    df["LAG_2"] = df.groupby("PART_ID")["DEMAND"].shift(2)
    df["LAG_4"] = df.groupby("PART_ID")["DEMAND"].shift(4)

    df["ROLLING_MEAN_4"] = (
        df.groupby("PART_ID")["DEMAND"]
        .transform(lambda x: x.shift(1).rolling(4).mean())
    )

    df["ROLLING_STD_4"] = (
        df.groupby("PART_ID")["DEMAND"]
        .transform(lambda x: x.shift(1).rolling(4).std())
    )

    return df.dropna().reset_index(drop=True)


def train_forecasting_model(df):
    feature_df = prepare_features(df)

    feature_columns = [
        "WEEK_NUMBER",
        "MONTH",
        "QUARTER",
        "YEAR",
        "UNIT_PRICE",
        "PROMOTION",
        "LEAD_TIME_DAYS",
        "SUPPLIER_RELIABILITY",
        "INVENTORY_LEVEL",
        "LAG_1",
        "LAG_2",
        "LAG_4",
        "ROLLING_MEAN_4",
        "ROLLING_STD_4"
    ]

    X = feature_df[feature_columns]
    y = feature_df["DEMAND"]

    model = XGBRegressor(
        n_estimators=180,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X, y)

    return model, feature_columns, feature_df


def generate_forecasts(model, feature_columns, history, weeks=12):
    history = history.copy()
    history["WEEK_DATE"] = pd.to_datetime(history["WEEK_DATE"])

    forecasts = []

    for part_id in history["PART_ID"].unique():
        part_history = (
            history[history["PART_ID"] == part_id]
            .sort_values("WEEK_DATE")
            .copy()
        )

        last_date = part_history["WEEK_DATE"].max()

        last_row = part_history.iloc[-1]

        demand_values = part_history["DEMAND"].astype(float).tolist()

        for step in range(1, weeks + 1):
            forecast_date = last_date + pd.Timedelta(weeks=step)

            lag_1 = demand_values[-1]
            lag_2 = demand_values[-2]
            lag_4 = demand_values[-4]

            rolling_mean = float(np.mean(demand_values[-4:]))
            rolling_std = float(np.std(demand_values[-4:], ddof=1))

            features = {
                "WEEK_NUMBER": int(forecast_date.isocalendar().week),
                "MONTH": forecast_date.month,
                "QUARTER": forecast_date.quarter,
                "YEAR": forecast_date.year,
                "UNIT_PRICE": float(last_row["UNIT_PRICE"]),
                "PROMOTION": 0,
                "LEAD_TIME_DAYS": int(last_row["LEAD_TIME_DAYS"]),
                "SUPPLIER_RELIABILITY": float(
                    last_row["SUPPLIER_RELIABILITY"]
                ),
                "INVENTORY_LEVEL": int(last_row["INVENTORY_LEVEL"]),
                "LAG_1": lag_1,
                "LAG_2": lag_2,
                "LAG_4": lag_4,
                "ROLLING_MEAN_4": rolling_mean,
                "ROLLING_STD_4": rolling_std
            }

            X_future = pd.DataFrame([features])[feature_columns]

            predicted = max(
                0,
                float(model.predict(X_future)[0])
            )

            demand_values.append(predicted)

            forecasts.append({
                "PART_ID": part_id,
                "FORECAST_WEEK": step,
                "FORECAST_DATE": forecast_date,
                "PREDICTED_DEMAND": round(predicted, 2)
            })

    return pd.DataFrame(forecasts)


def calculate_inventory_risk(history, forecasts):
    latest = (
        history.sort_values("WEEK_DATE")
        .groupby("PART_ID")
        .tail(1)
        .copy()
    )

    next_four_weeks = (
        forecasts[forecasts["FORECAST_WEEK"] <= 4]
        .groupby("PART_ID")["PREDICTED_DEMAND"]
        .sum()
        .reset_index(name="EXPECTED_DEMAND_4W")
    )

    result = latest.merge(
        next_four_weeks,
        on="PART_ID",
        how="left"
    )

    result["EXPECTED_WEEKLY_DEMAND"] = (
        result["EXPECTED_DEMAND_4W"] / 4
    )

    result["COVERAGE_DAYS"] = (
        result["INVENTORY_LEVEL"]
        / result["EXPECTED_WEEKLY_DEMAND"].clip(lower=1)
        * 7
    )

    result["STOCKOUT_RISK"] = np.where(
        result["COVERAGE_DAYS"] < result["LEAD_TIME_DAYS"],
        "HIGH",
        np.where(
            result["COVERAGE_DAYS"] < result["LEAD_TIME_DAYS"] + 7,
            "MEDIUM",
            "LOW"
        )
    )

    result["RECOMMENDED_STOCK"] = np.ceil(
        result["EXPECTED_WEEKLY_DEMAND"]
        * (result["LEAD_TIME_DAYS"] / 7 + 2)
    ).astype(int)

    result["ADDITIONAL_STOCK_REQUIRED"] = (
        result["RECOMMENDED_STOCK"] - result["INVENTORY_LEVEL"]
    ).clip(lower=0)

    return result


def calculate_supplier_risk(history):
    suppliers = (
        history.groupby("SUPPLIER_ID")
        .agg(
            SUPPLIER_RELIABILITY=("SUPPLIER_RELIABILITY", "mean"),
            AVERAGE_LEAD_TIME=("LEAD_TIME_DAYS", "mean")
        )
        .reset_index()
    )

    suppliers["RISK_SCORE"] = (
        (1 - suppliers["SUPPLIER_RELIABILITY"]) * 70
        + (suppliers["AVERAGE_LEAD_TIME"] / 60) * 30
    ).clip(0, 100)

    suppliers["RISK_LEVEL"] = np.where(
        suppliers["RISK_SCORE"] >= 50,
        "HIGH",
        np.where(
            suppliers["RISK_SCORE"] >= 25,
            "MEDIUM",
            "LOW"
        )
    )

    return suppliers.sort_values(
        "RISK_SCORE",
        ascending=False
    )


def generate_recommendations(inventory, suppliers):
    recommendations = []

    for _, row in inventory.iterrows():
        if row["STOCKOUT_RISK"] == "HIGH":
            action = (
                f"Urgently replenish {row['PART_ID']} with "
                f"{int(row['ADDITIONAL_STOCK_REQUIRED'])} additional units."
            )
        elif row["STOCKOUT_RISK"] == "MEDIUM":
            action = (
                f"Schedule replenishment for {row['PART_ID']} "
                "within the next planning cycle."
            )
        else:
            action = (
                f"Maintain current inventory for {row['PART_ID']} "
                "and monitor weekly demand."
            )

        recommendations.append({
            "TYPE": "INVENTORY",
            "ENTITY": row["PART_ID"],
            "RISK_LEVEL": row["STOCKOUT_RISK"],
            "RECOMMENDATION": action
        })

    for _, row in suppliers.iterrows():
        if row["RISK_LEVEL"] == "HIGH":
            action = (
                f"Review supplier {row['SUPPLIER_ID']} and "
                "prepare an alternative sourcing plan."
            )
        elif row["RISK_LEVEL"] == "MEDIUM":
            action = (
                f"Monitor supplier {row['SUPPLIER_ID']} "
                "and track delivery performance."
            )
        else:
            action = (
                f"Continue regular monitoring of supplier "
                f"{row['SUPPLIER_ID']}."
            )

        recommendations.append({
            "TYPE": "SUPPLIER",
            "ENTITY": row["SUPPLIER_ID"],
            "RISK_LEVEL": row["RISK_LEVEL"],
            "RECOMMENDATION": action
        })

    return pd.DataFrame(recommendations)
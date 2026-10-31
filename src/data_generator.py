import numpy as np
import pandas as pd


def generate_supply_chain_data():
    """
    Generate synthetic weekly demand and supplier data
    for 60 components over 80 weeks.
    """

    np.random.seed(42)

    parts = [f"PART_{i:03d}" for i in range(1, 61)]
    suppliers = [f"SUP_{i:02d}" for i in range(1, 11)]

    weeks = pd.date_range(
        end=pd.Timestamp.today().normalize(),
        periods=80,
        freq="W"
    )

    records = []

    for part in parts:
        base_demand = np.random.randint(40, 180)
        supplier = np.random.choice(suppliers)

        reliability = np.random.uniform(0.75, 0.99)
        lead_time = np.random.randint(5, 35)
        unit_price = np.random.uniform(100, 2000)

        for index, week in enumerate(weeks):
            seasonal_effect = 1 + 0.15 * np.sin(
                2 * np.pi * index / 52
            )

            trend_effect = 1 + (0.0015 * index)

            promotion = np.random.choice(
                [0, 1],
                p=[0.85, 0.15]
            )

            promotion_effect = 1.25 if promotion else 1.0

            demand = (
                base_demand
                * seasonal_effect
                * trend_effect
                * promotion_effect
                + np.random.normal(0, base_demand * 0.12)
            )

            demand = max(1, int(round(demand)))

            inventory = max(
                0,
                int(demand * np.random.uniform(1.0, 4.0))
            )

            records.append({
                "PART_ID": part,
                "WEEK_DATE": week,
                "DEMAND": demand,
                "UNIT_PRICE": round(unit_price, 2),
                "PROMOTION": promotion,
                "LEAD_TIME_DAYS": lead_time,
                "SUPPLIER_ID": supplier,
                "SUPPLIER_RELIABILITY": round(reliability, 3),
                "INVENTORY_LEVEL": inventory
            })

    df = pd.DataFrame(records)

    return df
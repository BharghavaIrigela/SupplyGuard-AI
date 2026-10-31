import os
import pandas as pd
import pyexasol


def get_connection():
    """Connect to the local or configured Exasol database."""

    host = os.getenv("EXASOL_HOST")
    user = os.getenv("EXASOL_USER")
    password = os.getenv("EXASOL_PASSWORD")

    if not all([host, user, password]):
        raise ValueError(
            "Missing Exasol credentials. "
            "Set EXASOL_HOST, EXASOL_USER and EXASOL_PASSWORD."
        )

    return pyexasol.connect(
        dsn=host,
        user=user,
        password=password,
        compression=True,
        encryption=True,
        websocket_sslopt={"cert_reqs": 0}
    )


def load_demand_data(df):
    """Create the Exasol table and upload demand history."""

    connection = get_connection()

    try:
        connection.execute(
            "CREATE SCHEMA IF NOT EXISTS SUPPLYGUARD"
        )

        connection.execute("""
            CREATE TABLE IF NOT EXISTS SUPPLYGUARD.DEMAND_HISTORY (
                PART_ID VARCHAR(50),
                WEEK_DATE DATE,
                DEMAND INTEGER,
                UNIT_PRICE DOUBLE,
                PROMOTION INTEGER,
                LEAD_TIME_DAYS INTEGER,
                SUPPLIER_ID VARCHAR(50),
                SUPPLIER_RELIABILITY DOUBLE,
                INVENTORY_LEVEL INTEGER
            )
        """)

        connection.execute(
            "TRUNCATE TABLE SUPPLYGUARD.DEMAND_HISTORY"
        )

        connection.import_from_pandas(
            df,
            ("SUPPLYGUARD", "DEMAND_HISTORY")
        )

        result = connection.execute("""
            SELECT COUNT(*)
            FROM SUPPLYGUARD.DEMAND_HISTORY
        """).fetchone()

        print(f"Successfully uploaded {result[0]} records to Exasol.")

    finally:
        connection.close()


def fetch_demand_data():
    """Retrieve demand history from Exasol."""

    connection = get_connection()

    try:
        df = connection.export_to_pandas("""
            SELECT *
            FROM SUPPLYGUARD.DEMAND_HISTORY
            ORDER BY PART_ID, WEEK_DATE
        """)

        return df

    finally:
        connection.close()

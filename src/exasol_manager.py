import os
import pandas as pd
import pyexasol


def get_connection():
    """
    Connect to Exasol using environment variables.
    """

    host = os.getenv("EXASOL_HOST")
    user = os.getenv("EXASOL_USER")
    password = os.getenv("EXASOL_PASSWORD")

    if not all([host, user, password]):
        raise ValueError(
            "Missing Exasol credentials. "
            "Set EXASOL_HOST, EXASOL_USER and EXASOL_PASSWORD."
        )

    connection = pyexasol.connect(
        dsn=host,
        user=user,
        password=password,
        compression=True
    )

    return connection


def load_demand_data(df):
    """
    Load synthetic demand data into Exasol.
    """

    connection = get_connection()

    try:
        connection.execute("""
            CREATE SCHEMA IF NOT EXISTS SUPPLYGUARD
        """)

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

        connection.execute("""
            TRUNCATE TABLE SUPPLYGUARD.DEMAND_HISTORY
        """)

        connection.import_from_pandas(
            df,
            "SUPPLYGUARD.DEMAND_HISTORY"
        )

    finally:
        connection.close()


def fetch_demand_data():
    """
    Retrieve demand history from Exasol.
    """

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
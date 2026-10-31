import streamlit as st
import pandas as pd
import plotly.express as px

from src.data_generator import generate_supply_chain_data
from src.analysis_engine import (
    train_forecasting_model,
    generate_forecasts,
    calculate_inventory_risk,
    calculate_supplier_risk,
    generate_recommendations
)

from src.exasol_manager import (
    load_demand_data,
    fetch_demand_data
)


st.set_page_config(
    page_title="SupplyGuard AI",
    page_icon="📦",
    layout="wide"
)


st.markdown(
    """
    <style>
    .main {
        background-color: #0e1117;
    }
    [data-testid="stMetric"] {
        background-color: #1b2430;
        padding: 18px;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.title("📦 SupplyGuard AI")
st.subheader("Predictive Supply Chain Risk & Decision Intelligence")

st.caption(
    "AI-powered demand forecasting, inventory monitoring "
    "and supplier risk intelligence."
)


with st.sidebar:
    st.header("Data Configuration")

    data_source = st.radio(
        "Select data source",
        ["Synthetic Demo Data", "Exasol Database"]
    )

    st.divider()

    st.caption("ExaThon India 2026")
    st.caption("Track: Factories, Fleets & Supply Chains")


@st.cache_data
def get_demo_data():
    return generate_supply_chain_data()


@st.cache_resource
def build_model(data):
    return train_forecasting_model(data)


if st.button("Run Supply Chain Analysis", type="primary"):

    try:
        with st.spinner("Preparing supply chain data..."):

            if data_source == "Synthetic Demo Data":
                history = get_demo_data()

            else:
                history = fetch_demand_data()

            st.session_state["history"] = history

        with st.spinner("Training demand forecasting model..."):

            model, feature_columns, _ = build_model(history)

            forecasts = generate_forecasts(
                model,
                feature_columns,
                history,
                weeks=12
            )

        with st.spinner("Calculating inventory and supplier risks..."):

            inventory = calculate_inventory_risk(
                history,
                forecasts
            )

            suppliers = calculate_supplier_risk(history)

            recommendations = generate_recommendations(
                inventory,
                suppliers
            )

            st.session_state["forecasts"] = forecasts
            st.session_state["inventory"] = inventory
            st.session_state["suppliers"] = suppliers
            st.session_state["recommendations"] = recommendations

        st.success("Supply chain analysis completed successfully!")

    except Exception as error:
        st.error(f"Analysis failed: {error}")


if "forecasts" not in st.session_state:

    st.info(
        "Select a data source and click "
        "'Run Supply Chain Analysis' to begin."
    )

else:

    history = st.session_state["history"]
    forecasts = st.session_state["forecasts"]
    inventory = st.session_state["inventory"]
    suppliers = st.session_state["suppliers"]
    recommendations = st.session_state["recommendations"]

    st.divider()

    st.header("Executive Overview")

    total_parts = inventory["PART_ID"].nunique()

    high_risk_parts = (
        inventory["STOCKOUT_RISK"] == "HIGH"
    ).sum()

    high_risk_suppliers = (
        suppliers["RISK_LEVEL"] == "HIGH"
    ).sum()

    total_demand = forecasts["PREDICTED_DEMAND"].sum()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Components Monitored", total_parts)

    col2.metric("High Stock-out Risk", int(high_risk_parts))

    col3.metric("High-risk Suppliers", int(high_risk_suppliers))

    col4.metric(
        "12-week Forecast Demand",
        f"{total_demand:,.0f}"
    )

    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs([
        "Demand Forecast",
        "Inventory Risk",
        "Supplier Intelligence",
        "Action Center"
    ])

    with tab1:

        st.header("12-Week Demand Forecast")

        selected_part = st.selectbox(
            "Select component",
            sorted(forecasts["PART_ID"].unique())
        )

        actual = history[
            history["PART_ID"] == selected_part
        ][["WEEK_DATE", "DEMAND"]].copy()

        actual["TYPE"] = "Historical Demand"

        predicted = forecasts[
            forecasts["PART_ID"] == selected_part
        ][["FORECAST_DATE", "PREDICTED_DEMAND"]].copy()

        predicted.columns = ["WEEK_DATE", "DEMAND"]
        predicted["TYPE"] = "Predicted Demand"

        chart_data = pd.concat(
            [actual, predicted],
            ignore_index=True
        )

        fig = px.line(
            chart_data,
            x="WEEK_DATE",
            y="DEMAND",
            color="TYPE",
            markers=True,
            title=f"Demand Forecast — {selected_part}"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Forecast Details")
        st.dataframe(
            forecasts[
                forecasts["PART_ID"] == selected_part
            ],
            use_container_width=True
        )

    with tab2:

        st.header("Inventory Stock-out Risk")

        risk_counts = (
            inventory["STOCKOUT_RISK"]
            .value_counts()
            .reset_index()
        )

        risk_counts.columns = ["Risk Level", "Components"]

        fig = px.pie(
            risk_counts,
            names="Risk Level",
            values="Components",
            hole=0.45,
            title="Inventory Risk Distribution"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Component Risk Details")

        st.dataframe(
            inventory[
                [
                    "PART_ID",
                    "INVENTORY_LEVEL",
                    "EXPECTED_WEEKLY_DEMAND",
                    "LEAD_TIME_DAYS",
                    "COVERAGE_DAYS",
                    "STOCKOUT_RISK",
                    "ADDITIONAL_STOCK_REQUIRED"
                ]
            ].sort_values(
                "COVERAGE_DAYS"
            ),
            use_container_width=True
        )

    with tab3:

        st.header("Supplier Risk Intelligence")

        fig = px.bar(
            suppliers,
            x="SUPPLIER_ID",
            y="RISK_SCORE",
            color="RISK_LEVEL",
            title="Supplier Risk Score",
            range_y=[0, 100]
        )

        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            suppliers,
            use_container_width=True
        )

    with tab4:

        st.header("AI-Assisted Decision Center")

        st.caption(
            "Transparent recommendations based on forecast demand, "
            "inventory coverage and supplier reliability."
        )

        filter_risk = st.selectbox(
            "Filter by risk",
            ["ALL", "HIGH", "MEDIUM", "LOW"]
        )

        displayed = recommendations.copy()

        if filter_risk != "ALL":
            displayed = displayed[
                displayed["RISK_LEVEL"] == filter_risk
            ]

        st.dataframe(
            displayed,
            use_container_width=True,
            hide_index=True
        )

        csv = displayed.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Recommendations CSV",
            data=csv,
            file_name="supplyguard_recommendations.csv",
            mime="text/csv"
        )

    st.divider()

    st.caption(
        "SupplyGuard AI | ExaThon India 2026 | "
        "Predictions are decision-support estimates, not guarantees."
    )
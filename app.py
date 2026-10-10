import json
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Maharashtra Climate Change Index",
    page_icon="🌦️",
    layout="wide"
)

# ==========================================
# 2. PROJECT TITLE
# ==========================================
st.title("🌦️ Maharashtra Climate Change Index")
st.subheader("PCA-Based Analysis of 36 Districts")

st.markdown("""
This project develops a Climate Change Index (CCI)
for Maharashtra districts using climate indicators:

- Rainfall
- Maximum Temperature (Tmax)
- Minimum Temperature (Tmin)
- Diurnal Temperature Range (DTR)

Principal Component Analysis (PCA) is used in the
project methodology to calculate the index and rank
districts.
""")

st.divider()

# ==========================================
# 3. LOAD RANKING DATA
# ==========================================
st.sidebar.header("Climate Data")

uploaded_file = st.sidebar.file_uploader(
    "Upload CCI_Ranking.csv",
    type=["csv"]
)

if uploaded_file is not None:
    ranking = pd.read_csv(uploaded_file)

elif Path("CCI_Ranking.csv").exists():
    ranking = pd.read_csv("CCI_Ranking.csv")

else:
    ranking = None

# ==========================================
# 4. DISPLAY RESULTS
# ==========================================
if ranking is not None:

    required_columns = ["Rank", "District", "PCA_CCI"]

    if all(col in ranking.columns for col in required_columns):

        ranking = ranking[required_columns].copy()

        ranking["Rank"] = pd.to_numeric(
            ranking["Rank"], errors="coerce"
        )

        ranking["PCA_CCI"] = pd.to_numeric(
            ranking["PCA_CCI"], errors="coerce"
        )

        ranking = ranking.dropna(
            subset=["Rank", "District", "PCA_CCI"]
        )

        ranking["Rank"] = ranking["Rank"].astype(int)

        ranking = ranking.sort_values("Rank")

        st.success("Climate index ranking loaded successfully!")

        # Summary metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Districts Included",
                ranking["District"].nunique()
            )

        with col2:
            st.metric(
                "Highest PCA CCI",
                f"{ranking['PCA_CCI'].max():.4f}"
            )

        with col3:
            st.metric(
                "Lowest PCA CCI",
                f"{ranking['PCA_CCI'].min():.4f}"
            )

        st.divider()

        # District ranking table
        st.header("District-Wise CCI Ranking")

        st.dataframe(
            ranking,
            use_container_width=True,
            hide_index=True
        )

        # Interactive chart
        st.header("Compare Districts")

        chart_data = ranking.sort_values(
            "PCA_CCI", ascending=False
        )

        fig = px.bar(
            chart_data,
            x="District",
            y="PCA_CCI",
            hover_data=["Rank"],
            title="PCA-Based Climate Change Index by District",
            labels={
                "District": "District",
                "PCA_CCI": "PCA-Based CCI"
            }
        )

        fig.update_layout(
            xaxis_tickangle=-60,
            height=600
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

# ==========================================
# INTERACTIVE MAHARASHTRA DISTRICT CCI MAP
# ==========================================

st.subheader("Maharashtra Climate Change Index Map")

try:
    with open("Maharashtra_CCI_Map.geojson", "r", encoding="utf-8") as f:
        geojson_data = json.load(f)

    map_df = pd.DataFrame([
        feature["properties"]
        for feature in geojson_data["features"]
    ])

    fig_map = px.choropleth(
        map_df,
        geojson=geojson_data,
        locations="Map_District",
        featureidkey="properties.Map_District",
        color="PCA_CCI",
        hover_name="District",
        hover_data={
            "Rank": True,
            "PCA_CCI": ":.2f",
            "Category": True,
            "Map_District": False
        },
        color_continuous_scale="YlOrRd",
        projection="mercator",
        title="District-wise Climate Change Index"
    )

    fig_map.update_geos(
        fitbounds="locations",
        visible=False
    )

    fig_map.update_layout(
        height=650,
        margin={"r": 10, "t": 60, "l": 10, "b": 10},
        coloraxis_colorbar_title="PCA CCI"
    )

    st.plotly_chart(fig_map, use_container_width=True)

except FileNotFoundError:
    st.error(
        "Map file not found. Please upload "
        "Maharashtra_CCI_Map.geojson to your GitHub repository."
    )

        # Download results
        st.download_button(
            label="Download CCI Ranking CSV",
            data=ranking.to_csv(index=False).encode("utf-8"),
            file_name="Maharashtra_CCI_Ranking.csv",
            mime="text/csv"
        )

    else:
        st.error(
            "The CSV must contain these columns: "
            "Rank, District, PCA_CCI. "
            f"Columns found: {list(ranking.columns)}"
        )

else:
    st.warning(
        "CCI_Ranking.csv was not found. "
        "Upload it using the sidebar or add it "
        "to the GitHub repository."
    )

# ==========================================
# 5. PROJECT METHODOLOGY
# ==========================================
st.divider()

st.header("Methodology")

st.markdown("""
1. Calculate monthly climate trends.
2. Normalize the climate trend indicators.
3. Calculate climate variable scores.
4. Standardize the selected variables.
5. Perform correlation-based PCA.
6. Calculate PCA-based weights and the final CCI.
7. Rank the districts using the final index.

The index scores and ranking shown above are loaded
from the project's exported results.
""")

st.caption(
    "Maharashtra Climate Change Index | "
    "Statistical analysis using Python and PCA"
)

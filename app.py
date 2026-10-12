
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

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
for Maharashtra districts using:

- Rainfall
- Maximum Temperature (Tmax)
- Minimum Temperature (Tmin)
- Diurnal Temperature Range (DTR)

Principal Component Analysis (PCA) is used to calculate
the index weights and rank districts.
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
# 4. DISPLAY RANKING RESULTS
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

        # Ranking table
        st.header("District-Wise CCI Ranking")

        st.dataframe(
            ranking,
            use_container_width=True,
            hide_index=True
        )

        # Bar chart
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
# 5. MAHARASHTRA DISTRICT MAP
# ==========================================
st.divider()
st.header("Maharashtra Climate Change Index Map")

geojson_path = Path("Maharashtra_CCI_Map.geojson")

if geojson_path.exists():

    try:
        with open(
            geojson_path, "r", encoding="utf-8"
        ) as file:
            geojson_data = json.load(file)

        features = geojson_data.get("features", [])

        # Read district attributes from GeoJSON
        map_df = pd.DataFrame([
            feature.get("properties", {})
            for feature in features
        ])

        required_map_columns = [
            "Map_District",
            "District",
            "Rank",
            "PCA_CCI",
            "Category"
        ]

        if not all(
            col in map_df.columns
            for col in required_map_columns
        ):
            st.error(
                "The GeoJSON file is missing required columns. "
                "Please check Map_District, District, Rank, "
                "PCA_CCI, and Category."
            )

        else:
            map_df["PCA_CCI"] = pd.to_numeric(
                map_df["PCA_CCI"], errors="coerce"
            )
            map_df["Rank"] = pd.to_numeric(
                map_df["Rank"], errors="coerce"
            )

            map_df = map_df.dropna(
                subset=[
                    "Map_District",
                    "District",
                    "Rank",
                    "PCA_CCI"
                ]
            ).copy()

            # Ensure feature identifiers are strings
            map_df["Map_District"] = (
                map_df["Map_District"].astype(str)
            )

            for feature in features:
                props = feature.get("properties", {})
                if "Map_District" in props:
                    props["Map_District"] = str(
                        props["Map_District"]
                    )

            # Build the district choropleth map
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

            # Fit the geographic view to the district shapes
            fig_map.update_geos(
                fitbounds="locations",
                visible=False
            )

            # Add district names using geographic label points.
            # For polygons, use the mean of their exterior vertices
            # as an approximate label position.
            label_lons = []
            label_lats = []
            label_names = []

            for feature in features:
                props = feature.get("properties", {})
                district_name = props.get("District")

                if not district_name:
                    district_name = props.get("Map_District")

                geometry = feature.get("geometry", {})
                geom_type = geometry.get("type")
                coords = geometry.get("coordinates", [])

                # Handle Polygon and MultiPolygon geometries
                polygons = []

                if geom_type == "Polygon":
                    polygons = [coords]
                elif geom_type == "MultiPolygon":
                    polygons = coords

                # Use the largest exterior ring for label position
                best_ring = None
                best_size = 0

                for polygon in polygons:
                    if polygon and polygon[0]:
                        ring = polygon[0]
                        if len(ring) > best_size:
                            best_ring = ring
                            best_size = len(ring)

                if best_ring and district_name:
                    lon = sum(point[0] for point in best_ring) / len(best_ring)
                    lat = sum(point[1] for point in best_ring) / len(best_ring)

                    label_lons.append(lon)
                    label_lats.append(lat)
                    label_names.append(str(district_name))

            # Overlay labels on the map
            if label_names:
                fig_map.add_trace(
                    go.Scattergeo(
                        lon=label_lons,
                        lat=label_lats,
                        text=label_names,
                        mode="text",
                        textfont=dict(
                            size=8,
                            color="black"
                        ),
                        hoverinfo="skip",
                        showlegend=False
                    )
                )

            fig_map.update_layout(
                height=750,
                margin=dict(
                    l=10, r=10, t=60, b=10
                ),
                coloraxis_colorbar=dict(
                    title="PCA CCI"
                )
            )

            st.plotly_chart(
                fig_map,
                use_container_width=True
            )

            st.caption(
                "District colors represent PCA CCI scores. "
                "Hover over a district to see its score, "
                "rank, and category. Text labels identify districts."
            )

    except Exception as error:
        st.error(
            f"Unable to display the district map: {error}"
        )

else:
    st.warning(
        "Maharashtra_CCI_Map.geojson was not found. "
        "Upload this file to the same GitHub folder as app.py."
    )
    
# ============================================================
# PCA ANALYSIS RESULTS
# ============================================================

st.header("PCA Analysis Results")

# 1. Variance explained by each principal component
eigenvalue_df = pd.DataFrame({
    "Principal_Component": ["PC1", "PC2", "PC3", "PC4"],
    "Variance_Explained": [
        51.246036, 31.057649, 13.824048, 3.872267
    ],
    "Cumulative_Variance": [
        51.246036, 82.303685, 96.127733, 100.000000
    ]
})

st.subheader("1. Variance Explained by PCA")

fig_variance = px.bar(
    eigenvalue_df,
    x="Principal_Component",
    y="Variance_Explained",
    text="Variance_Explained",
    title="Variance Explained by Each Principal Component",
    labels={
        "Principal_Component": "Principal Component",
        "Variance_Explained": "Variance Explained (%)"
    }
)

fig_variance.update_traces(
    texttemplate="%{text:.2f}%",
    textposition="outside"
)
fig_variance.update_layout(yaxis_range=[0, 60])
st.plotly_chart(fig_variance, use_container_width=True)

st.write(
    "PC1 explains 51.25% of the total variation. "
    "PC1 and PC2 together explain 82.30%."
)

# 2. Cumulative variance explained
st.subheader("2. Cumulative Variance Explained")

fig_cumulative = px.line(
    eigenvalue_df,
    x="Principal_Component",
    y="Cumulative_Variance",
    markers=True,
    text="Cumulative_Variance",
    title="Cumulative Variance Explained by PCA",
    labels={
        "Principal_Component": "Principal Component",
        "Cumulative_Variance": "Cumulative Variance (%)"
    }
)

fig_cumulative.update_traces(
    texttemplate="%{text:.2f}%",
    textposition="top center"
)
fig_cumulative.update_layout(yaxis_range=[0, 105])
st.plotly_chart(fig_cumulative, use_container_width=True)

# 3. PCA loadings
loading_df = pd.DataFrame({
    "Variable": ["Rainfall", "Tmax", "Tmin", "DTR"],
    "PC1": [-0.125557, 0.667279, 0.843112, 0.967752],
    "PC2": [0.909859, 0.606492, -0.281217, -0.055141],
    "PC3": [0.425038, -0.434115, 0.445547, -0.033690],
    "PC4": [0.065614, -0.164381, -0.200345, 0.296398]
})

st.subheader("3. PCA Loadings")

loading_long = loading_df.melt(
    id_vars="Variable",
    var_name="Principal_Component",
    value_name="Loading"
)

fig_loadings = px.bar(
    loading_long,
    x="Variable",
    y="Loading",
    color="Principal_Component",
    barmode="group",
    title="PCA Loadings of Climate Variables",
    labels={"Loading": "PCA Loading"}
)

fig_loadings.add_hline(y=0, line_dash="dash")
st.plotly_chart(fig_loadings, use_container_width=True)

# 4. PCA weights
weights_df = pd.DataFrame({
    "Variable": ["Rainfall", "Tmax", "Tmin", "DTR"],
    "PCA_Weight": [0.048223, 0.256281, 0.323813, 0.371683]
})

weights_df["Weight_Percent"] = weights_df["PCA_Weight"] * 100
weights_df = weights_df.sort_values(
    "PCA_Weight", ascending=False
)

st.subheader("4. PCA-Based Climate Variable Weights")

fig_weights = px.bar(
    weights_df,
    x="Variable",
    y="Weight_Percent",
    text="Weight_Percent",
    title="Relative Weights of Climate Variables",
    labels={
        "Variable": "Climate Variable",
        "Weight_Percent": "PCA Weight (%)"
    }
)

fig_weights.update_traces(
    texttemplate="%{text:.2f}%",
    textposition="outside"
)
fig_weights.update_layout(yaxis_range=[0, 45])
st.plotly_chart(fig_weights, use_container_width=True)

st.dataframe(
    weights_df[["Variable", "Weight_Percent"]],
    use_container_width=True,
    hide_index=True
)

# ==========================================
# 6. PROJECT METHODOLOGY
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

The scores and rankings displayed above are loaded
from the project's exported results.
""")


# ==========================================
# ADDITIONAL PROJECT DOCUMENTATION
# ==========================================

st.divider()
st.header("Interpretation of PCA Results")

st.markdown("""
### What do the PCA results tell us?

- **PC1 explains 51.25%** of the variation in the four
  standardized climate indicators.
- **PC1 and PC2 together explain 82.30%** of the variation.
- DTR has the highest weight in the selected PCA weighting
  method, followed by Tmin, Tmax, and Rainfall.
- Rainfall has a smaller weight in this particular index.

These results describe the statistical structure of the
selected indicators. They do not prove that one variable
is the most important driver of climate change everywhere.
""")

st.subheader("How to Interpret the CCI Ranking")

st.markdown("""
A higher CCI score indicates a greater combined magnitude
of the selected climate-trend indicators under this
project's scoring and weighting procedure.

The index is designed to compare districts. It should not
be interpreted as a direct probability of a climate disaster
or as a complete measure of climate vulnerability.
""")

# ==========================================
# DATA SOURCES
# ==========================================

st.divider()
st.header("Data Sources")

st.markdown("""
The project uses historical climate data obtained from
Open-Meteo.

- **Provider:** Open-Meteo
- **Historical Weather API documentation:**
  https://open-meteo.com/en/docs/historical-weather-api
- **Geographical coverage:** 36 districts of Maharashtra
- **Study period:** 2000–2025
- **Indicators:** Rainfall, maximum temperature (Tmax),
  minimum temperature (Tmin), and Diurnal Temperature
  Range (DTR)

The data were processed using Python before calculating
the district-level scores and PCA-based index.
""")

# ==========================================
# LIMITATIONS
# ==========================================

st.divider()
st.header("Limitations")

st.markdown("""
1. The index depends on the quality and completeness of
   the underlying climate data.
2. Absolute trend magnitudes do not preserve direction:
   an increasing trend and a decreasing trend can both
   contribute positively to the score.
3. PCA-based weights depend on the selected variables,
   data, and preprocessing method.
4. The index does not directly include population exposure,
   infrastructure, socioeconomic vulnerability, or
   adaptive capacity.
5. The district ranking is a comparative statistical
   result, not a complete climate-risk assessment.
6. Further validation against independent climate studies
   is needed before using the rankings for policy decisions.
""")

# ==========================================
# SENSITIVITY ANALYSIS
# ==========================================

st.divider()
st.header("Sensitivity Analysis")

st.markdown("""
The PCA-based ranking was compared with a ranking
constructed using equal weights for the four climate
variables.

- **Spearman rank correlation:** 0.9807
- **Average absolute rank difference:** approximately 1.56
  positions
- **Largest rank change in the comparison:** Ratnagiri,
  which moved from rank 27 under PCA weighting to rank 21
  under equal weighting.

The high rank correlation suggests that the two weighting
approaches produce broadly similar district rankings.
However, this comparison is not independent validation
of the index.
""")

# ==========================================
# FUTURE IMPROVEMENTS
# ==========================================

st.divider()
st.header("Future Improvements")

st.markdown("""
- Validate climate data coverage and document data
  preprocessing in detail.
- Compare alternative normalization and weighting methods.
- Study increasing and decreasing climate trends separately.
- Include additional reliable climate indicators.
- Compare results with independent climate assessments.
- Add more downloadable outputs and interactive filters.
""")

# ==========================================
# TECHNOLOGY STACK
# ==========================================

st.divider()
st.header("Technology Stack")

st.markdown("""
- **Python:** Data processing and statistical analysis
- **Pandas and NumPy:** Data manipulation and calculations
- **Scikit-learn:** Standardization and PCA
- **Plotly:** Interactive charts and maps
- **Streamlit:** Interactive dashboard
- **Google Colab:** Analysis and experimentation
- **GitHub:** Source code and project documentation
""")

# ==========================================
# PROJECT LINKS
# ==========================================

st.divider()
st.header("Project Links")

st.markdown("""
- [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Scikit-learn PCA Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
""")

st.caption(
    "Maharashtra Climate Change Index | "
    "Statistical analysis using Python and PCA"
)


# ============================================================
# MAHARASHTRA CLIMATE CHANGE INDEX (CCI)
# PCA-BASED DISTRICT ANALYSIS AND INTERACTIVE DASHBOARD
# Streamlit Application
# ============================================================

import io
import json
import re
import urllib.request
from pathlib import Path
import os

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Maharashtra Climate Change Index",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# 2. CONSTANTS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

RANKING_FILE = BASE_DIR / "CCI_Ranking.csv"

# Add your district boundary GeoJSON to GitHub using either name.
MAP_CANDIDATES = [
    BASE_DIR / "Maharashtra_District_Boundaries.geojson",
    BASE_DIR / "Maharashtra_CCI_Map.geojson",
    BASE_DIR / "Maharashtra.geojson",
]

MAP_URL = (
    "https://gist.githubusercontent.com/"
    "saketkc/7c1ce9d993773ced9974f6c57730b7df/"
    "raw/Maharashtra.geojson"
)

# Exact requested colors
COLORS = {
    "Q1 - Highest CCI": "#8B0000",
    "Q2 - Medium CCI": "#FDBB84",
    "Q3 - Lowest CCI": "#FFF2AE",
}

GROUP_LABELS = list(COLORS.keys())

# PCA results supplied in your existing project
VARIANCE_DF = pd.DataFrame({
    "Principal_Component": ["PC1", "PC2", "PC3", "PC4"],
    "Variance_Explained": [
        51.246036, 31.057649, 13.824048, 3.872267
    ],
    "Cumulative_Variance": [
        51.246036, 82.303685, 96.127733, 100.000000
    ],
})

LOADING_DF = pd.DataFrame({
    "Variable": ["Rainfall", "Tmax", "Tmin", "DTR"],
    "PC1": [-0.125557, 0.667279, 0.843112, 0.967752],
    "PC2": [0.909859, 0.606492, -0.281217, -0.055141],
    "PC3": [0.425038, -0.434115, 0.445547, -0.033690],
    "PC4": [0.065614, -0.164381, -0.200345, 0.296398],
})

WEIGHTS_DF = pd.DataFrame({
    "Variable": ["Rainfall", "Tmax", "Tmin", "DTR"],
    "PCA_Weight": [0.048223, 0.256281, 0.323813, 0.371683],
})

# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def normalize_name(name):
    """Normalize district names for geographic matching."""

    name = str(name).strip().lower()
    name = re.sub(r"[^a-z0-9]", "", name)

    aliases = {
        "mumbaicity": "mumbai",
        "mumbai": "mumbai",
        "aurangabad": "chhatrapatisambhajinagar",
        "chhatrapatisambhajinagar": "chhatrapatisambhajinagar",
        "osmanabad": "dharashiv",
        "dharashiv": "dharashiv",
        "ahmednagar": "ahilyanagar",
        "ahmadnagar": "ahilyanagar",
        "ahilyanagar": "ahilyanagar",
        "raigarh": "raigad",
    }

    return aliases.get(name, name)


def prepare_ranking(data):
    """Validate ranking data and assign three equal rank groups."""

    data = data.copy()
    data.columns = [str(c).strip() for c in data.columns]

    # Recognize common column-name alternatives.
    aliases = {
        "district": ["District", "District_Name", "District Name"],
        "score": ["PCA_CCI", "PCA CCI", "CCI", "CCI_Score"],
        "rank": ["Rank", "CCI_Rank", "Final_Rank"],
    }

    def locate(candidates):
        lookup = {str(c).strip().lower(): c for c in data.columns}
        for candidate in candidates:
            if candidate.lower() in lookup:
                return lookup[candidate.lower()]
        return None

    district_col = locate(aliases["district"])
    score_col = locate(aliases["score"])
    rank_col = locate(aliases["rank"])

    if district_col is None:
        raise ValueError(
            "Ranking data must contain a District column."
        )

    if score_col is None and rank_col is None:
        raise ValueError(
            "Ranking data must contain PCA_CCI or Rank."
        )

    result = pd.DataFrame()
    result["District"] = data[district_col].astype(str).str.strip()

    if score_col is not None:
        result["PCA_CCI"] = pd.to_numeric(
            data[score_col], errors="coerce"
        )
    else:
        result["PCA_CCI"] = np.nan

    if rank_col is not None:
        result["Rank"] = pd.to_numeric(
            data[rank_col], errors="coerce"
        )
    else:
        result["Rank"] = np.nan

    result = result.replace([np.inf, -np.inf], np.nan)
    result = result.dropna(subset=["District"])

    result = result[
        result["District"].ne("")
        & result["District"].str.lower().ne("nan")
    ].copy()

    if score_col is not None and result["PCA_CCI"].isna().any():
        result = result.dropna(subset=["PCA_CCI"]).copy()

    if result.empty:
        raise ValueError("No valid ranking records were found.")

    # Recalculate ranks when the CCI score is available.
    # This keeps rank 1 = highest CCI.
    if score_col is not None:
        result = result.sort_values(
            "PCA_CCI", ascending=False
        ).reset_index(drop=True)

        result["Rank"] = result["PCA_CCI"].rank(
            ascending=False, method="first"
        ).astype(int)

    else:
        result = result.dropna(subset=["Rank"]).copy()
        result = result.sort_values("Rank").reset_index(drop=True)
        result["Rank"] = result["Rank"].rank(
            method="first", ascending=True
        ).astype(int)

    result["Map_Key"] = result["District"].apply(normalize_name)

    if result["Map_Key"].duplicated().any():
        duplicates = result.loc[
            result["Map_Key"].duplicated(keep=False), "District"
        ].tolist()

        raise ValueError(
            f"Duplicate district records detected: {duplicates}"
        )

    # Three approximately equal groups based on ranking position.
    result = result.sort_values("Rank").reset_index(drop=True)

    result["Rank_Position"] = np.arange(1, len(result) + 1)

    result["CCI_Group"] = pd.cut(
        result["Rank_Position"],
        bins=[
            0,
            len(result) / 3,
            2 * len(result) / 3,
            len(result),
        ],
        labels=GROUP_LABELS,
        include_lowest=True,
    )

    return result


@st.cache_data(show_spinner=False)
def read_ranking_csv(file_bytes):
    return pd.read_csv(io.BytesIO(file_bytes))


@st.cache_data(show_spinner=False)
def read_geojson_bytes(file_bytes):
    return gpd.read_file(io.BytesIO(file_bytes))


@st.cache_data(show_spinner=False)
def read_geojson_path(path_string, modified_time):
    return gpd.read_file(path_string)


def load_local_map():
    """Load the first available GeoJSON from the project directory."""

    for path in MAP_CANDIDATES:
        if path.exists():
            try:
                return read_geojson_path(
                    str(path), path.stat().st_mtime
                ), str(path.name)
            except Exception as exc:
                st.warning(f"Could not read {path.name}: {exc}")

    return None, None


def merge_map(boundaries, ranking):
    """Join district geometries with CCI ranks."""

    boundaries = boundaries.copy()

    possible_names = [
        "district_name", "District", "district",
        "DISTRICT", "DIST_NAME", "dtname",
        "NAME_2", "name_2", "name"
    ]

    district_col = next(
        (col for col in possible_names if col in boundaries.columns),
        None,
    )

    if district_col is None:
        raise ValueError(
            "District-name field not found in the boundary file. "
            f"Available fields: {boundaries.columns.tolist()}"
        )

    boundaries["Map_Key"] = (
        boundaries[district_col].astype(str).apply(normalize_name)
    )

    if boundaries["Map_Key"].duplicated().any():
        duplicates = boundaries.loc[
            boundaries["Map_Key"].duplicated(keep=False),
            district_col
        ].tolist()

        raise ValueError(
            "Duplicate district names in boundary data: "
            f"{duplicates}"
        )

    if boundaries.crs is None:
        raise ValueError(
            "The boundary file has no CRS. Set the correct CRS "
            "before using this file."
        )

    merged = boundaries.merge(
        ranking,
        on="Map_Key",
        how="left",
        validate="one_to_one",
    )

    unmatched_map = merged.loc[
        merged["Rank"].isna(), district_col
    ].tolist()

    unmatched_ranking = ranking.loc[
        ~ranking["Map_Key"].isin(boundaries["Map_Key"]),
        "District",
    ].tolist()

    return merged, district_col, unmatched_map, unmatched_ranking


def draw_district_map(map_data, district_col):
    """Draw a static district choropleth with names and ranks."""

    fig, ax = plt.subplots(figsize=(12, 13))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for group, color in COLORS.items():
        subset = map_data[map_data["CCI_Group"] == group]

        if not subset.empty:
            subset.plot(
                ax=ax,
                color=color,
                edgecolor="black",
                linewidth=0.65,
                zorder=2,
            )

    # Draw missing data, if any.
    missing = map_data[map_data["Rank"].isna()]

    if not missing.empty:
        missing.plot(
            ax=ax,
            color="#D3D3D3",
            edgecolor="black",
            linewidth=0.65,
            hatch="///",
            zorder=1,
        )

    short_names = {
        "Mumbai Suburban": "Mumbai Sub.",
        "Chhatrapati Sambhajinagar": "Ch. Sambhajinagar",
    }

    for _, row in map_data.iterrows():
        if row.geometry is None or row.geometry.is_empty:
            continue

        point = row.geometry.representative_point()

        if pd.notna(row["Rank"]):
            district = str(row["District"])
            district = short_names.get(district, district)
            label = f"{district}\n({int(row['Rank'])})"
        else:
            label = str(row[district_col])

        ax.annotate(
            label,
            (point.x, point.y),
            ha="center",
            va="center",
            fontsize=6.5,
            fontweight="bold",
            color="black",
            zorder=5,
            bbox=dict(
                boxstyle="round,pad=0.12",
                facecolor="white",
                edgecolor="none",
                alpha=0.70,
            ),
        )

    legend_items = [
        mpatches.Patch(
            facecolor=color,
            edgecolor="black",
            label=group,
        )
        for group, color in COLORS.items()
    ]

    if not missing.empty:
        legend_items.append(
            mpatches.Patch(
                facecolor="#D3D3D3",
                edgecolor="black",
                hatch="///",
                label="No CCI data",
            )
        )

    ax.legend(
        handles=legend_items,
        title="CCI Rank Groups",
        loc="lower left",
        fontsize=8,
        title_fontsize=9,
        framealpha=0.95,
    )

    ax.set_title(
        "Maharashtra Climate Change Index (CCI)\n"
        "PCA-Based District Ranking: Three Equal Groups",
        fontsize=14,
        fontweight="bold",
        pad=16,
    )

    ax.set_axis_off()
    ax.set_aspect("equal")
    fig.tight_layout()

    return fig


# ============================================================
# 4. PAGE HEADER
# ============================================================

st.title("🌦️ Maharashtra Climate Change Index")
st.subheader("PCA-Based Analysis of 36 Districts")

st.markdown(
    """
    This dashboard presents the district-wise Climate Change Index (CCI)
    for Maharashtra, based on PCA-weighted climate indicators.

    **Indicators:** Rainfall, maximum temperature (Tmax), minimum
    temperature (Tmin), and diurnal temperature range (DTR).

    Districts are grouped into three equal rank-based groups:
    **Q1 – Highest CCI**, **Q2 – Medium CCI**, and **Q3 – Lowest CCI**.
    """
)

st.info(
    "Interpretation: Rank 1 represents the highest CCI score. "
    "The three groups contain approximately equal numbers of districts."
)

st.divider()

# ============================================================
# 5. LOAD RANKING DATA
# ============================================================

st.sidebar.header("📂 Project Data")

uploaded_ranking = st.sidebar.file_uploader(
    "Upload CCI ranking CSV or Excel",
    type=["csv", "xlsx"],
    key="ranking_upload",
)

ranking = None

try:
    if uploaded_ranking is not None:
        if uploaded_ranking.name.lower().endswith(".csv"):
            raw_ranking = read_ranking_csv(
                uploaded_ranking.getvalue()
            )
        else:
            raw_ranking = pd.read_excel(
                uploaded_ranking, sheet_name=0
            )

        ranking = prepare_ranking(raw_ranking)

    elif RANKING_FILE.exists():
        raw_ranking = pd.read_csv(RANKING_FILE)
        ranking = prepare_ranking(raw_ranking)

    else:
        st.warning(
            "CCI_Ranking.csv was not found. Upload your ranking file "
            "using the sidebar or add CCI_Ranking.csv to your GitHub "
            "repository."
        )

except Exception as exc:
    st.error(f"Unable to load the ranking data: {exc}")

# ============================================================
# 6. RANKING TABLE, SUMMARY, AND BAR CHART
# ============================================================

if ranking is not None:

    st.success(
        f"CCI ranking loaded successfully: "
        f"{len(ranking)} district records."
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Districts", ranking["District"].nunique())

    with col2:
        st.metric(
            "Highest CCI",
            (
                f"{ranking['PCA_CCI'].max():.4f}"
                if ranking["PCA_CCI"].notna().any()
                else "N/A"
            ),
        )

    with col3:
        st.metric(
            "Lowest CCI",
            (
                f"{ranking['PCA_CCI'].min():.4f}"
                if ranking["PCA_CCI"].notna().any()
                else "N/A"
            ),
        )

    with col4:
        st.metric("Rank Groups", ranking["CCI_Group"].nunique())

    st.divider()
    st.header("📊 District-Wise CCI Ranking")

    display_cols = [
        col for col in
        ["Rank", "District", "PCA_CCI", "CCI_Group"]
        if col in ranking.columns
    ]

    st.dataframe(
        ranking[display_cols].sort_values("Rank"),
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "⬇️ Download CCI Ranking CSV",
        data=ranking[display_cols].to_csv(index=False).encode("utf-8"),
        file_name="Maharashtra_CCI_Ranking.csv",
        mime="text/csv",
    )

    if ranking["PCA_CCI"].notna().any():
        st.subheader("CCI Score by District")

        chart_data = ranking.sort_values(
            "PCA_CCI", ascending=False
        ).copy()

        fig_bar = px.bar(
            chart_data,
            x="District",
            y="PCA_CCI",
            color="CCI_Group",
            color_discrete_map=COLORS,
            hover_data=["Rank", "CCI_Group"],
            title="PCA-Based Climate Change Index by District",
            labels={
                "District": "District",
                "PCA_CCI": "PCA-Based CCI",
                "CCI_Group": "CCI Group",
            },
            category_orders={"CCI_Group": GROUP_LABELS},
        )

        fig_bar.update_layout(
            xaxis_tickangle=-60,
            height=580,
            margin=dict(l=20, r=20, t=70, b=130),
            legend_title_text="CCI Rank Groups",
        )

        st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("District Counts by CCI Group")

    group_counts = (
        ranking["CCI_Group"]
        .value_counts()
        .reindex(GROUP_LABELS, fill_value=0)
        .rename_axis("CCI_Group")
        .reset_index(name="District_Count")
    )

    fig_groups = px.bar(
        group_counts,
        x="CCI_Group",
        y="District_Count",
        color="CCI_Group",
        color_discrete_map=COLORS,
        category_orders={"CCI_Group": GROUP_LABELS},
        text="District_Count",
        title="Number of Districts in Each CCI Group",
    )

    fig_groups.update_layout(
        height=380,
        showlegend=False,
        xaxis_title="CCI Group",
        yaxis_title="Number of Districts",
    )

    st.plotly_chart(fig_groups, use_container_width=True)

# ============================================================

```python
# ============================================================
# SECTION 7: MAHARASHTRA DISTRICT CCI MAP
# ============================================================

import streamlit as st
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import re
import unicodedata


# ------------------------------------------------------------
# 7.1 LOAD DISTRICT BOUNDARIES
# ------------------------------------------------------------

districts_gdf = load_district_boundaries()

if districts_gdf is None:
    st.error(
        "District boundaries are not loaded. "
        "Please check your existing boundary-loading function "
        "and boundary data file."
    )
    st.stop()

elif districts_gdf.empty:
    st.error("The district boundary file contains no records.")
    st.stop()

else:
    st.success(
        f"Loaded {len(districts_gdf)} district boundary records."
    )

    st.write(
        "Available boundary columns:",
        districts_gdf.columns.tolist()
    )


# ------------------------------------------------------------
# 7.2 FIND THE CCI RANKING DATAFRAME
# ------------------------------------------------------------

if "final_ranking" in globals() and isinstance(
    final_ranking, pd.DataFrame
):
    ranking_df = final_ranking.copy()

elif "ranking" in globals() and isinstance(
    ranking, pd.DataFrame
):
    ranking_df = ranking.copy()

elif "df" in globals() and isinstance(df, pd.DataFrame):
    ranking_df = df.copy()

else:
    st.error(
        "CCI ranking data was not found. "
        "Please ensure that your district ranking dataframe "
        "is loaded before the map section."
    )
    st.stop()

if ranking_df.empty:
    st.error("The CCI ranking dataframe is empty.")
    st.stop()


# ------------------------------------------------------------
# 7.3 HELPER FUNCTION TO NORMALIZE DISTRICT NAMES
# ------------------------------------------------------------

def normalize_district_name(value):
    """Normalize district names before merging the datasets."""

    if pd.isna(value):
        return ""

    value = unicodedata.normalize("NFKD", str(value))
    value = value.encode("ascii", "ignore").decode("utf-8")
    value = value.lower().strip()

    # Remove common words and punctuation differences
    value = re.sub(r"\bdistrict\b", "", value)
    value = re.sub(r"[^a-z0-9]", "", value)

    # Handle common boundary/data spelling variations
    aliases = {
        "ahmednagar": "ahilyanagar",
        "aurangabad": "chhatrapatisambhajinagar",
        "osmanabad": "dharashiv",
        "buldana": "buldhana",
        "garhchiroli": "gadchiroli",
    }

    return aliases.get(value, value)


# ------------------------------------------------------------
# 7.4 DETECT DISTRICT AND RANK COLUMNS
# ------------------------------------------------------------

boundary_district_candidates = [
    "District",
    "District_Name",
    "district_name",
    "district",
    "DISTRICT",
    "NAME_2",
    "NAME2",
    "dtname",
    "DTNAME",
    "DIST_NAME",
    "Map_District",
]

ranking_district_candidates = [
    "District",
    "District_Name",
    "district_name",
    "district",
    "DISTRICT",
    "Map_District",
]

rank_candidates = [
    "Rank",
    "CCI_Rank",
    "Overall_Rank",
    "Final_Rank",
    "rank",
    "rank_position",
]

score_candidates = [
    "PCA_CCI",
    "CCI",
    "CCI_Score",
    "CCI_Score_Weighted",
    "Final_CCI",
    "Climate_Change_Index",
]

boundary_district_col = next(
    (
        col for col in boundary_district_candidates
        if col in districts_gdf.columns
    ),
    None,
)

ranking_district_col = next(
    (
        col for col in ranking_district_candidates
        if col in ranking_df.columns
    ),
    None,
)

rank_col = next(
    (
        col for col in rank_candidates
        if col in ranking_df.columns
    ),
    None,
)

score_col = next(
    (
        col for col in score_candidates
        if col in ranking_df.columns
    ),
    None,
)

if boundary_district_col is None:
    st.error(
        "Could not identify the district-name column in the "
        "boundary data. Check the boundary columns displayed above."
    )
    st.stop()

if ranking_district_col is None:
    st.error(
        "Could not identify the district-name column in the "
        "CCI ranking dataframe."
    )
    st.write("Ranking columns:", ranking_df.columns.tolist())
    st.stop()

if rank_col is None:
    st.warning(
        "No existing rank column was found. "
        "Ranks will be calculated from the CCI score if available."
    )

    if score_col is not None:
        ranking_df[score_col] = pd.to_numeric(
            ranking_df[score_col], errors="coerce"
        )

        # Higher CCI score receives a better rank.
        ranking_df["Rank"] = (
            ranking_df[score_col]
            .rank(ascending=False, method="min")
        )

        rank_col = "Rank"

    else:
        st.error(
            "Neither a rank column nor a recognized CCI score "
            "column was found. Please check your ranking dataframe."
        )
        st.stop()

# Standardize the rank field
if rank_col != "Rank":
    ranking_df["Rank"] = ranking_df[rank_col]

ranking_df["Rank"] = pd.to_numeric(
    ranking_df["Rank"], errors="coerce"
)


# ------------------------------------------------------------
# 7.5 PREPARE THE BOUNDARY AND RANKING DATA
# ------------------------------------------------------------

# Keep only records with valid district names
districts_gdf = districts_gdf.copy()
ranking_df = ranking_df.copy()

districts_gdf = districts_gdf[
    districts_gdf[boundary_district_col].notna()
].copy()

ranking_df = ranking_df[
    ranking_df[ranking_district_col].notna()
].copy()

# Create matching district-name keys
districts_gdf["_district_key"] = (
    districts_gdf[boundary_district_col]
    .apply(normalize_district_name)
)

ranking_df["_district_key"] = (
    ranking_df[ranking_district_col]
    .apply(normalize_district_name)
)

# Remove blank district keys
districts_gdf = districts_gdf[
    districts_gdf["_district_key"] != ""
].copy()

ranking_df = ranking_df[
    ranking_df["_district_key"] != ""
].copy()

# Avoid duplicate district records in ranking data
ranking_df = ranking_df.drop_duplicates(
    subset="_district_key",
    keep="first"
)

# Select the attributes to merge
columns_to_merge = [
    "_district_key",
    "Rank",
]

if score_col is not None and score_col in ranking_df.columns:
    columns_to_merge.append(score_col)

ranking_for_map = ranking_df[columns_to_merge].copy()

# Merge ranking attributes into district geometries
map_data = districts_gdf.merge(
    ranking_for_map,
    on="_district_key",
    how="left",
    validate="many_to_one",
)

# Use a consistent name for displaying districts
map_data["Map_District"] = (
    map_data[boundary_district_col].astype(str)
)

# Remove records without usable geometries
map_data = map_data[
    map_data.geometry.notna()
    & ~map_data.geometry.is_empty
].copy()

if map_data.empty:
    st.error(
        "No valid district geometries remain after preparing the map."
    )
    st.stop()


# ------------------------------------------------------------
# 7.6 CREATE CCI GROUPS
# ------------------------------------------------------------

# The previous KeyError occurred because CCI_Group was missing.
# Create it explicitly before the map renderer uses it.

map_data["CCI_Group"] = "Unranked"

valid_rank_mask = (
    map_data["Rank"].notna()
    & (map_data["Rank"] > 0)
)

valid_ranks = map_data.loc[valid_rank_mask, "Rank"]

if not valid_ranks.empty:
    # Assign groups according to rank order:
    # Group 1 = highest-ranked districts
    # Group 2 = middle-ranked districts
    # Group 3 = lowest-ranked districts

    number_of_ranked_districts = len(valid_ranks)

    rank_position = (
        map_data.loc[valid_rank_mask, "Rank"]
        .rank(method="first", ascending=True)
    )

    group_1_limit = number_of_ranked_districts / 3
    group_2_limit = 2 * number_of_ranked_districts / 3

    map_data.loc[valid_rank_mask, "CCI_Group"] = (
        rank_position.apply(
            lambda position: (
                "Group 1"
                if position <= group_1_limit
                else (
                    "Group 2"
                    if position <= group_2_limit
                    else "Group 3"
                )
            )
        )
    )

# Colors for the district groups
COLORS = {
    "Group 1": "#8B0000",
    "Group 2": "#FDBB84",
    "Group 3": "#FFF2AE",
    "Unranked": "#D9D9D9",
}

map_data["Map_Color"] = map_data["CCI_Group"].map(COLORS)

# Keep a copy of the merged map for inspection/debugging
st.session_state["maharashtra_cci_map_data"] = map_data


# ------------------------------------------------------------
# 7.7 MAP TITLE AND DISPLAY OPTIONS
# ------------------------------------------------------------

st.subheader("Maharashtra District Climate Change Index Map")

st.caption(
    "District colours represent relative CCI rank groups. "
    "District labels show the district name and rank."
)

show_district_labels = st.checkbox(
    "Show district names and ranks on the map",
    value=True,
    key="show_cci_district_labels",
)

show_map_table = st.checkbox(
    "Show district ranking data below the map",
    value=True,
    key="show_cci_map_table",
)


# ------------------------------------------------------------
# 7.8 DRAW THE DISTRICT MAP
# ------------------------------------------------------------

try:
    # Matplotlib works best when the map uses a projected CRS.
    plot_data = map_data.copy()

    if plot_data.crs is None:
        plot_data = plot_data.set_crs("EPSG:4326")

    try:
        plot_data = plot_data.to_crs("EPSG:32643")
    except Exception:
        # Keep the original CRS if reprojection fails.
        pass

    fig, ax = plt.subplots(figsize=(12, 13))

    for group_name, group_color in COLORS.items():
        group_data = plot_data[
            plot_data["CCI_Group"] == group_name
        ]

        if not group_data.empty:
            group_data.plot(
                ax=ax,
                color=group_color,
                edgecolor="#333333",
                linewidth=0.8,
                missing_kwds={
                    "color": COLORS["Unranked"],
                },
            )

    # Add district names and rank numbers
    if show_district_labels:
        for _, row in plot_data.iterrows():

            geometry = row.geometry

            if geometry is None or geometry.is_empty:
                continue

            # Representative points are guaranteed to lie
            # within the geometry more often than centroids.
            point = geometry.representative_point()

            district_name = str(row["Map_District"])

            if pd.notna(row["Rank"]):
                rank_text = f"Rank {int(row['Rank'])}"
            else:
                rank_text = "Unranked"

            label_text = (
                f"{district_name}\n{rank_text}"
            )

            ax.annotate(
                label_text,
                xy=(point.x, point.y),
                ha="center",
                va="center",
                fontsize=6.5,
                color="black",
                fontweight="bold",
                bbox={
                    "boxstyle": "round,pad=0.2",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.75,
                },
            )

    # Add map title and legend
    ax.set_title(
        "Maharashtra Climate Change Index (CCI)\n"
        "District-wise Ranking",
        fontsize=16,
        fontweight="bold",
        pad=18,
    )

    legend_items = [
        mpatches.Patch(
            color=COLORS["Group 1"],
            label="Group 1 - Highest-ranked districts",
        ),
        mpatches.Patch(
            color=COLORS["Group 2"],
            label="Group 2 - Middle-ranked districts",
        ),
        mpatches.Patch(
            color=COLORS["Group 3"],
            label="Group 3 - Lowest-ranked districts",
        ),
        mpatches.Patch(
            color=COLORS["Unranked"],
            label="Unranked / No matching CCI record",
        ),
    ]

    ax.legend(
        handles=legend_items,
        loc="lower left",
        fontsize=9,
        frameon=True,
        title="CCI Rank Groups",
    )

    ax.set_axis_off()

    fig.tight_layout()

    st.pyplot(fig, use_container_width=True)

    plt.close(fig)

except Exception as e:
    st.error(f"Unable to draw the district map: {e}")
    st.exception(e)


# ------------------------------------------------------------
# 7.9 DISPLAY DISTRICT RANKING TABLE
# ------------------------------------------------------------

if show_map_table:

    st.subheader("District-wise CCI Ranking")

    display_columns = ["Map_District", "Rank", "CCI_Group"]

    if score_col is not None and score_col in map_data.columns:
        display_columns.append(score_col)

    display_df = (
        map_data[display_columns]
        .drop_duplicates(subset=["Map_District"])
        .sort_values(
            by="Rank",
            ascending=True,
            na_position="last",
        )
        .reset_index(drop=True)
    )

    display_df = display_df.rename(
        columns={
            "Map_District": "District",
            "CCI_Group": "CCI Rank Group",
        }
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )

    ranked_count = int(display_df["Rank"].notna().sum())
    unranked_count = len(display_df) - ranked_count

    col1, col2, col3 = st.columns(3)

    col1.metric("Districts on map", len(display_df))
    col2.metric("Districts with ranks", ranked_count)
    col3.metric("Districts without ranks", unranked_count)

# ============================================================
# END OF SECTION 7
# ============================================================
```
# ============================================================
# 8. PCA ANALYSIS RESULTS
# ============================================================

st.divider()
st.header("📈 PCA Analysis Results")

st.subheader("1. Variance Explained by PCA")

fig_variance = px.bar(
    VARIANCE_DF,
    x="Principal_Component",
    y="Variance_Explained",
    text="Variance_Explained",
    title="Variance Explained by Each Principal Component",
    labels={
        "Principal_Component": "Principal Component",
        "Variance_Explained": "Variance Explained (%)",
    },
)

fig_variance.update_traces(
    texttemplate="%{text:.2f}%",
    textposition="outside",
)

fig_variance.update_layout(
    yaxis_range=[0, 60],
    height=430,
)

st.plotly_chart(fig_variance, use_container_width=True)

st.write(
    "PC1 explains 51.25% of the total variation. "
    "PC1 and PC2 together explain 82.30%."
)

st.subheader("2. Cumulative Variance Explained")

fig_cumulative = px.line(
    VARIANCE_DF,
    x="Principal_Component",
    y="Cumulative_Variance",
    markers=True,
    text="Cumulative_Variance",
    title="Cumulative Variance Explained by PCA",
    labels={
        "Principal_Component": "Principal Component",
        "Cumulative_Variance": "Cumulative Variance (%)",
    },
)

fig_cumulative.update_traces(
    texttemplate="%{text:.2f}%",
    textposition="top center",
)

fig_cumulative.update_layout(
    yaxis_range=[0, 105],
    height=430,
)

st.plotly_chart(fig_cumulative, use_container_width=True)

st.subheader("3. PCA Loadings")

loading_long = LOADING_DF.melt(
    id_vars="Variable",
    var_name="Principal_Component",
    value_name="Loading",
)

fig_loadings = px.bar(
    loading_long,
    x="Variable",
    y="Loading",
    color="Principal_Component",
    barmode="group",
    title="PCA Loadings of Climate Variables",
    labels={"Loading": "PCA Loading"},
)

fig_loadings.add_hline(y=0, line_dash="dash")

fig_loadings.update_layout(height=500)

st.plotly_chart(fig_loadings, use_container_width=True)

st.subheader("4. PCA-Based Climate Variable Weights")

weights_display = WEIGHTS_DF.copy()
weights_display["Weight_Percent"] = (
    weights_display["PCA_Weight"] * 100
)

weights_display = weights_display.sort_values(
    "PCA_Weight", ascending=False
)

fig_weights = px.bar(
    weights_display,
    x="Variable",
    y="Weight_Percent",
    text="Weight_Percent",
    title="Relative Weights of Climate Variables",
    labels={
        "Variable": "Climate Variable",
        "Weight_Percent": "PCA Weight (%)",
    },
)

fig_weights.update_traces(
    texttemplate="%{text:.2f}%",
    textposition="outside",
)

fig_weights.update_layout(
    yaxis_range=[0, 45],
    height=430,
)

st.plotly_chart(fig_weights, use_container_width=True)

st.dataframe(
    weights_display[["Variable", "Weight_Percent"]],
    use_container_width=True,
    hide_index=True,
)

# ============================================================
# 9. METHODOLOGY
# ============================================================

st.divider()
st.header("📚 Methodology")

st.markdown(
    """
    1. Calculate climate trend indicators for each district.
    2. Normalize the selected climate indicators.
    3. Standardize the selected variables.
    4. Perform Principal Component Analysis (PCA).
    5. Derive PCA-based weights according to the project's
       selected weighting method.
    6. Calculate the district-level Climate Change Index.
    7. Rank districts by the final CCI score.
    8. Divide the ranking into three approximately equal groups.

    The ranking file contains the precomputed scores. This dashboard
    visualizes those results; it does not recalculate the underlying
    climate trends or PCA model.
    """
)

# ============================================================
# 10. INTERPRETATION
# ============================================================

st.divider()
st.header("Interpretation of Results")

st.markdown(
    """
    - **Q1 – Highest CCI:** ranks 1–12, the highest index scores.
    - **Q2 – Medium CCI:** ranks 13–24, the middle group.
    - **Q3 – Lowest CCI:** ranks 25–36, the lowest index scores.

    A higher CCI score indicates a greater combined magnitude of the
    selected climate-trend indicators under this project's scoring
    and weighting procedure.

    This index is comparative. It is not a direct probability of a
    climate disaster or a complete measure of climate vulnerability.
    """
)

# ============================================================
# 11. DATA SOURCES AND LIMITATIONS
# ============================================================

st.divider()
st.header("Data Sources")

st.markdown(
    """
    **Historical weather data:** Open-Meteo Historical Weather API

    https://open-meteo.com/en/docs/historical-weather-api

    **Study period:** 2000–2025 (as specified in the project)

    **Indicators:** Rainfall, Tmax, Tmin, and DTR

    **Geographical coverage:** Maharashtra districts
    """
)

st.header("Limitations")

st.markdown(
    """
    1. Results depend on the quality and completeness of the source data.
    2. PCA weights depend on the variables and preprocessing choices.
    3. The index does not directly measure population exposure,
       socioeconomic vulnerability, or adaptive capacity.
    4. Statistical rankings should be validated against independent
       climate assessments before being used for policy decisions.
    """
)

# ============================================================
# 12. SENSITIVITY ANALYSIS
# ============================================================

st.divider()
st.header("Sensitivity Analysis")

st.markdown(
    """
    The project's previously reported comparison with an equal-weight
    ranking gave a Spearman rank correlation of **0.9807**, with an
    average absolute rank difference of approximately **1.56** places.

    These values are reported project results, not recalculated from
    the uploaded file. Recalculate them if the ranking or input data
    change. High rank correlation alone is not independent validation
    of the index.
    """
)

# ============================================================
# 13. FUTURE IMPROVEMENTS
# ============================================================

st.divider()
st.header("Future Improvements")

st.markdown(
    """
    - Validate climate data coverage and preprocessing.
    - Compare alternative normalization and weighting methods.
    - Analyze increasing and decreasing climate trends separately.
    - Include additional reliable climate indicators.
    - Validate findings against independent climate assessments.
    - Add further downloadable datasets and interactive filters.
    """
)

# ============================================================
# 14. TECHNOLOGY STACK
# ============================================================

st.divider()
st.header("Technology Stack")

st.markdown(
    """
    - **Python:** Data processing and statistical analysis
    - **Pandas and NumPy:** Data manipulation
    - **GeoPandas:** District geographic boundaries
    - **Matplotlib:** District map rendering
    - **Plotly:** Interactive statistical charts
    - **Streamlit:** Web dashboard
    - **GitHub:** Source code and project version control
    """
)

# ============================================================
# 15. PROJECT LINKS
# ============================================================

st.divider()
st.header("Project Links")

st.markdown(
    """
    - [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api)
    - [Streamlit Documentation](https://docs.streamlit.io/)
    - [Pandas Documentation](https://pandas.pydata.org/docs/)
    - [GeoPandas Documentation](https://geopandas.org/en/stable/)
    - [Scikit-learn PCA Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
    """
)

st.caption(
    "Maharashtra Climate Change Index | "
    "PCA-based district comparison | Python + Streamlit"
)

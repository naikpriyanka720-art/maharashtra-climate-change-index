
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

# ============================================================
# ============================================================
# MAHARASHTRA CLIMATE CHANGE INDEX (CCI) DISTRICT MAP
# ============================================================

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import re
import unicodedata

# ------------------------------------------------------------
# 1. CHECK REQUIRED DATA
# ------------------------------------------------------------

if "boundaries" not in globals() or boundaries is None:
    st.error(
        "District boundaries are not loaded. "
        "Please check your existing boundary-loading function."
    )
    st.stop()

if not isinstance(boundaries, gpd.GeoDataFrame):
    st.error("The district boundaries must be a GeoDataFrame.")
    st.stop()

if boundaries.empty:
    st.error("The district boundary dataset is empty.")
    st.stop()

# Find the dataframe containing district rankings
ranking_source = globals().get("ranking_source")

if ranking_source is None or not isinstance(ranking_source, pd.DataFrame):
    possible_sources = ["final_ranking", "ranking", "df"]

    for source_name in possible_sources:
        candidate = globals().get(source_name)

        if isinstance(candidate, pd.DataFrame) and not candidate.empty:
            ranking_source = candidate.copy()
            break

if ranking_source is None or not isinstance(ranking_source, pd.DataFrame):
    st.error(
        "CCI ranking data was not found. "
        "Please check that your district ranking dataframe is created "
        "before this map section."
    )
    st.stop()

# ------------------------------------------------------------
# 2. FIND DISTRICT COLUMNS
# ------------------------------------------------------------

def find_map_column(dataframe, candidates):
    """Find a column while ignoring capitalization and spaces."""
    lookup = {
        str(column).strip().lower().replace("_", " "): column
        for column in dataframe.columns
    }

    for candidate in candidates:
        key = candidate.strip().lower().replace("_", " ")
        if key in lookup:
            return lookup[key]

    return None


ranking_district_col = find_map_column(
    ranking_source,
    [
        "District",
        "District Name",
        "District_Name",
        "district_name",
        "Name of District",
    ],
)

boundary_district_col = find_map_column(
    boundaries,
    [
        "District",
        "District Name",
        "DISTRICT",
        "district_name",
        "District_Name",
        "dtname",
        "DIST_NAME",
        "NAME_2",
        "name",
    ],
)

if ranking_district_col is None:
    st.error(
        "Could not find the district-name column in the ranking data. "
        f"Available columns: {list(ranking_source.columns)}"
    )
    st.stop()

if boundary_district_col is None:
    st.error(
        "Could not find the district-name column in the boundary data. "
        f"Available columns: {list(boundaries.columns)}"
    )
    st.stop()

# ------------------------------------------------------------
# 3. NORMALIZE DISTRICT NAMES FOR MATCHING
# ------------------------------------------------------------

def normalize_map_district(value):
    if pd.isna(value):
        return ""

    name = unicodedata.normalize("NFKD", str(value))
    name = "".join(
        character for character in name
        if not unicodedata.combining(character)
    )
    name = name.strip().lower()
    name = re.sub(r"^(district|dist)\s+", "", name)
    name = re.sub(r"[^a-z0-9]", "", name)

    # Common old and alternative district names
    aliases = {
        "aurangabad": "chhatrapatisambhajinagar",
        "chhatrapatisambhajinagar": "chhatrapatisambhajinagar",
        "osmanabad": "dharashiv",
        "dharashiv": "dharashiv",
        "ahmednagar": "ahilyanagar",
        "ahilyanagar": "ahilyanagar",
        "buldhana": "buldhana",
        "mumbai": "mumbaicity",
        "mumbaicity": "mumbaicity",
        "greatermumbai": "mumbaicity",
        "mumbaisuburban": "mumbaisuburban",
        "mumbaisuburb": "mumbaisuburban",
    }

    return aliases.get(name, name)


# ------------------------------------------------------------
# 4. PREPARE THE RANKING DATA
# ------------------------------------------------------------

ranking_for_map = ranking_source.copy()

rank_col = find_map_column(
    ranking_for_map,
    ["Rank", "CCI Rank", "CCI_Rank", "Overall Rank", "Ranking"],
)

score_col = find_map_column(
    ranking_for_map,
    [
        "CCI Score",
        "CCI_Score",
        "CCI",
        "Final CCI",
        "Final Score",
        "Composite Score",
        "Index Score",
    ],
)

ranking_for_map["_district_key"] = (
    ranking_for_map[ranking_district_col]
    .apply(normalize_map_district)
)

# Prefer an existing rank; otherwise calculate rank from CCI score.
if rank_col is not None:
    ranking_for_map["_map_rank"] = pd.to_numeric(
        ranking_for_map[rank_col], errors="coerce"
    )
elif score_col is not None:
    ranking_for_map["_map_score"] = pd.to_numeric(
        ranking_for_map[score_col], errors="coerce"
    )

    ranking_for_map["_map_rank"] = (
        ranking_for_map["_map_score"]
        .rank(method="min", ascending=False)
    )
else:
    st.error(
        "Could not find a Rank or CCI Score column in the ranking data. "
        f"Available columns: {list(ranking_for_map.columns)}"
    )
    st.stop()

ranking_for_map = ranking_for_map[
    ["_district_key", "_map_rank"]
].copy()

ranking_for_map = ranking_for_map.dropna(
    subset=["_district_key", "_map_rank"]
)

# Keep only one ranking record per district.
ranking_for_map = ranking_for_map.drop_duplicates(
    subset=["_district_key"],
    keep="first",
)

# Assign three groups using rank.
def get_cci_group(rank):
    if pd.isna(rank):
        return None

    if rank <= 12:
        return "Q1 - Highest CCI"
    elif rank <= 24:
        return "Q2 - Medium CCI"
    else:
        return "Q3 - Lowest CCI"


ranking_for_map["_map_group"] = ranking_for_map["_map_rank"].apply(
    get_cci_group
)

# ------------------------------------------------------------
# 5. PREPARE DISTRICT BOUNDARIES
# ------------------------------------------------------------

map_boundaries = boundaries.copy()

map_boundaries["_district_key"] = (
    map_boundaries[boundary_district_col]
    .apply(normalize_map_district)
)

# Remove any previous rank/group columns to avoid merge conflicts.
columns_to_remove = [
    column
    for column in map_boundaries.columns
    if str(column).strip().lower().replace("_", " ")
    in ["rank", "cci rank", "cci group", "cci score"]
]

map_boundaries = map_boundaries.drop(
    columns=columns_to_remove,
    errors="ignore",
)

# Merge rankings onto the boundary geometry.
map_data = map_boundaries.merge(
    ranking_for_map,
    on="_district_key",
    how="left",
    validate="many_to_one",
)

map_data = map_data.rename(
    columns={
        "_map_rank": "Rank",
        "_map_group": "CCI_Group",
    }
)

# ------------------------------------------------------------
# 6. REMOVE UNMATCHED DISTRICTS
#    This prevents a "No data" category from appearing.
# ------------------------------------------------------------

cci_colors = {
    "Q1 - Highest CCI": "#8B0000",
    "Q2 - Medium CCI": "#FDBB84",
    "Q3 - Lowest CCI": "#FFF2AE",
}

map_data["Rank"] = pd.to_numeric(
    map_data["Rank"], errors="coerce"
)

map_data = map_data[
    map_data["Rank"].notna()
    & map_data["CCI_Group"].isin(cci_colors.keys())
    & map_data.geometry.notna()
    & ~map_data.geometry.is_empty
].copy()

if map_data.empty:
    st.error(
        "No districts matched between the ranking data and the boundary "
        "file. Check the district names in both datasets."
    )

    st.write("District names in ranking data:")
    st.write(
        ranking_source[ranking_district_col]
        .dropna()
        .astype(str)
        .sort_values()
        .unique()
        .tolist()
    )

    st.write("District names in boundary data:")
    st.write(
        boundaries[boundary_district_col]
        .dropna()
        .astype(str)
        .sort_values()
        .unique()
        .tolist()
    )

    st.stop()

# ------------------------------------------------------------
# 7. DRAW THE MAP WITH DISTRICT NAMES AND RANKS
# ------------------------------------------------------------

st.subheader("Maharashtra Climate Change Index — District Map")

# Ensure the geometries have a CRS.
if map_data.crs is None:
    map_data = map_data.set_crs("EPSG:4326")

# Project to a coordinate system suitable for Maharashtra.
try:
    map_projected = map_data.to_crs("EPSG:32643")
except Exception:
    map_projected = map_data.to_crs("EPSG:4326")

fig, ax = plt.subplots(figsize=(13, 13))

# Draw only the three CCI groups.
for group_name, group_color in cci_colors.items():
    group_data = map_projected[
        map_projected["CCI_Group"] == group_name
    ]

    if not group_data.empty:
        group_data.plot(
            ax=ax,
            color=group_color,
            edgecolor="#333333",
            linewidth=0.65,
        )

# Place district labels inside the polygons.
label_points = map_projected.geometry.representative_point()

for index, point in zip(map_projected.index, label_points):
    district_name = str(
        map_projected.at[index, boundary_district_col]
    ).strip()

    rank_value = map_projected.at[index, "Rank"]

    if pd.isna(rank_value):
        continue

    rank_text = f"Rank {int(round(rank_value))}"

    ax.annotate(
        f"{district_name}\n{rank_text}",
        xy=(point.x, point.y),
        ha="center",
        va="center",
        fontsize=7,
        fontweight="bold",
        color="black",
        bbox=dict(
            boxstyle="round,pad=0.18",
            facecolor="white",
            edgecolor="none",
            alpha=0.72,
        ),
    )

# Legend contains only the three CCI groups.
legend_handles = [
    mpatches.Patch(
        facecolor=color,
        edgecolor="#333333",
        label=label,
    )
    for label, color in cci_colors.items()
]

ax.legend(
    handles=legend_handles,
    title="CCI Category",
    loc="lower left",
    frameon=True,
    fontsize=9,
    title_fontsize=10,
)

ax.set_title(
    "Maharashtra Climate Change Index (CCI)\n"
    "District-wise Ranking",
    fontsize=16,
    fontweight="bold",
    pad=15,
)

ax.set_axis_off()
plt.tight_layout()

st.pyplot(fig, use_container_width=True)
plt.close(fig)

# ------------------------------------------------------------
# 8. DISPLAY MATCHED DISTRICT RANKINGS
# ------------------------------------------------------------

st.subheader("District-wise CCI Ranking")

ranking_table = map_data[
    [boundary_district_col, "Rank", "CCI_Group"]
].copy()

ranking_table = ranking_table.rename(
    columns={
        boundary_district_col: "District",
        "CCI_Group": "CCI Category",
    }
)

ranking_table["Rank"] = ranking_table["Rank"].round().astype(int)

ranking_table = ranking_table.sort_values(
    "Rank"
).reset_index(drop=True)

st.dataframe(
    ranking_table,
    use_container_width=True,
    hide_index=True,
)

st.caption(
    f"Map displays {len(ranking_table)} matched districts. "
    "Districts without matching CCI ranking data are excluded from "
    "the map and legend."
)
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

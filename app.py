
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
# SECTION 7: MAHARASHTRA CCI DISTRICT MAP
# ============================================================

import os
import re
import zipfile
import tempfile
import unicodedata
from pathlib import Path

import streamlit as st
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from matplotlib.patches import Patch


# ------------------------------------------------------------
# 1. MAP COLORS
# ------------------------------------------------------------

CCI_COLORS = {
    "Q1 - Highest CCI": "#8B0000",
    "Q2 - Medium CCI": "#FDBB84",
    "Q3 - Lowest CCI": "#FFF2AE",
}

NO_DATA_COLOR = "#E5E7EB"
BORDER_COLOR = "#FFFFFF"


# ------------------------------------------------------------
# 2. HELPER FUNCTIONS
# ------------------------------------------------------------

def find_column(dataframe, candidates):
    """Find a column using case-insensitive matching."""

    if dataframe is None:
        return None

    columns = list(dataframe.columns)
    normalized = {
        str(column).strip().lower().replace(" ", "_"): column
        for column in columns
    }

    for candidate in candidates:
        key = candidate.strip().lower().replace(" ", "_")

        if key in normalized:
            return normalized[key]

    return None


def normalize_district_name(value):
    """Standardize district names for matching GeoJSON boundaries."""

    if pd.isna(value):
        return None

    name = unicodedata.normalize("NFKD", str(value))
    name = name.encode("ascii", "ignore").decode("ascii")
    name = name.lower().strip()

    # Remove common suffixes and punctuation.
    name = re.sub(r"\bdistrict\b", "", name)
    name = re.sub(r"[^a-z0-9]", "", name)

    # Handle historical and spelling variations.
    aliases = {
        "mumbaicity": "mumbai",
        "mumbaisuburban": "mumbaisuburban",
        "aurangabad": "chhatrapatisambhajinagar",
        "chhatrapatisambhajinagar": "chhatrapatisambhajinagar",
        "osmanabad": "dharashiv",
        "dharashiv": "dharashiv",
        "ahmednagar": "ahilyanagar",
        "ahmadnagar": "ahilyanagar",
        "ahilyanagar": "ahilyanagar",
        "raigarh": "raigad",
        "raigad": "raigad",
    }

    return aliases.get(name, name)


def assign_cci_groups(dataframe):
    """
    Assign groups by rank:
    ranks 1-12  -> Q1 - Highest CCI
    ranks 13-24 -> Q2 - Medium CCI
    ranks 25-36 -> Q3 - Lowest CCI
    """

    dataframe = dataframe.copy()

    rank_column = find_column(
        dataframe,
        ["Rank", "CCI_Rank", "Overall_Rank", "rank_position"],
    )

    score_column = find_column(
        dataframe,
        ["PCA_CCI", "CCI", "CCI_Score", "CCI_Score_Weighted"],
    )

    if rank_column is not None:
        dataframe["Rank"] = pd.to_numeric(
            dataframe[rank_column], errors="coerce"
        )

    elif score_column is not None:
        dataframe[score_column] = pd.to_numeric(
            dataframe[score_column], errors="coerce"
        )

        dataframe["Rank"] = (
            dataframe[score_column]
            .rank(method="min", ascending=False)
        )

    else:
        raise ValueError(
            "No Rank or CCI score column found in the ranking data."
        )

    dataframe["Rank"] = dataframe["Rank"].round().astype("Int64")

    dataframe["CCI_Group"] = pd.cut(
        dataframe["Rank"].astype(float),
        bins=[0, 12, 24, 36],
        labels=[
            "Q1 - Highest CCI",
            "Q2 - Medium CCI",
            "Q3 - Lowest CCI",
        ],
        include_lowest=True,
    )

    return dataframe


# ------------------------------------------------------------
# 3. GET THE EXISTING CCI RANKING DATAFRAME
# ------------------------------------------------------------

ranking_source = None

for variable_name in ["final_ranking", "ranking", "df"]:
    candidate = globals().get(variable_name)

    if isinstance(candidate, pd.DataFrame):
        district_column = find_column(
            candidate,
            ["District", "District_Name", "district_name", "district"],
        )

        score_column = find_column(
            candidate,
            ["PCA_CCI", "CCI", "CCI_Score", "CCI_Score_Weighted"],
        )

        rank_column = find_column(
            candidate,
            ["Rank", "CCI_Rank", "Overall_Rank", "rank_position"],
        )

        if district_column and (score_column or rank_column):
            ranking_source = candidate.copy()
            break


if ranking_source is None:
    st.error(
        "CCI ranking data was not found. Create final_ranking or "
        "ranking with a District column and a Rank or PCA_CCI column "
        "before this map section."
    )

else:
    # Prepare ranking table.
    district_column = find_column(
        ranking_source,
        ["District", "District_Name", "district_name", "district"],
    )

    ranking_source["_district_key"] = (
        ranking_source[district_column].apply(normalize_district_name)
    )

    ranking_source = assign_cci_groups(ranking_source)

    # Remove duplicate district records before merging.
    ranking_source = ranking_source.drop_duplicates(
        subset="_district_key",
        keep="first",
    )


    # --------------------------------------------------------
    # 4. LOAD DISTRICT BOUNDARIES
    # --------------------------------------------------------

    # Reuse an already-loaded GeoDataFrame where possible.
    boundaries = None

    for variable_name in ["boundaries", "maha", "map_boundaries"]:
        candidate = globals().get(variable_name)

        if isinstance(candidate, gpd.GeoDataFrame):
            boundaries = candidate.copy()
            break

    # If boundaries are not already loaded, allow file upload.
    if boundaries is None:
        st.subheader("Maharashtra District Map")

        uploaded_boundary = st.file_uploader(
            "Upload Maharashtra district boundaries",
            type=["geojson", "json", "zip", "gpkg", "shp"],
            key="cci_district_boundary_upload",
            help=(
                "Upload a district GeoJSON, zipped shapefile, "
                "or GeoPackage."
            ),
        )

        if uploaded_boundary is not None:
            try:
                filename = uploaded_boundary.name.lower()

                if filename.endswith(".zip"):
                    with tempfile.TemporaryDirectory() as temp_dir:
                        zip_path = os.path.join(temp_dir, "boundaries.zip")

                        with open(zip_path, "wb") as file:
                            file.write(uploaded_boundary.getvalue())

                        with zipfile.ZipFile(zip_path) as archive:
                            archive.extractall(temp_dir)

                        shp_files = list(
                            Path(temp_dir).rglob("*.shp")
                        )

                        if shp_files:
                            boundaries = gpd.read_file(shp_files[0])

                        else:
                            geojson_files = list(
                                Path(temp_dir).rglob("*.geojson")
                            )

                            if geojson_files:
                                boundaries = gpd.read_file(
                                    geojson_files[0]
                                )
                            else:
                                st.error(
                                    "No shapefile or GeoJSON found "
                                    "inside the ZIP file."
                                )

                elif filename.endswith(".shp"):
                    st.error(
                        "Upload the shapefile components together "
                        "as a ZIP file, including .shp, .shx, .dbf "
                        "and .prj files."
                    )

                else:
                    boundaries = gpd.read_file(uploaded_boundary)

            except Exception as error:
                st.error(f"Could not load district boundaries: {error}")


    # --------------------------------------------------------
    # 5. MATCH DISTRICTS AND CREATE THE MAP DATAFRAME
    # --------------------------------------------------------

    if boundaries is None:
        st.info(
            "Load your Maharashtra district GeoDataFrame before "
            "displaying the map."
        )

    elif boundaries.empty:
        st.error("The district boundary file contains no features.")

    else:
        boundaries = boundaries.copy()

        if boundaries.crs is None:
            # Only use this if the source coordinates are longitude
            # and latitude. Change this if your source uses another CRS.
            boundaries = boundaries.set_crs("EPSG:4326")

        if boundaries.geometry.isna().all():
            st.error("The boundary file contains no valid geometries.")

        else:
            boundary_district_column = find_column(
                boundaries,
                [
                    "District",
                    "District_Name",
                    "district_name",
                    "district",
                    "NAME_2",
                    "NAME2",
                    "dtname",
                    "DIST_NAME",
                    "district_n",
                ],
            )

            if boundary_district_column is None:
                st.error(
                    "Could not identify the district name column in "
                    f"the boundary file. Available columns: "
                    f"{list(boundaries.columns)}"
                )

            else:
                boundaries["_district_key"] = (
                    boundaries[boundary_district_column]
                    .apply(normalize_district_name)
                )

                # Keep only the required ranking fields.
                ranking_for_map = ranking_source[
                    [
                        "_district_key",
                        "Rank",
                        "CCI_Group",
                    ]
                ].copy()

                # Ensure one ranking record per district.
                ranking_for_map = ranking_for_map.drop_duplicates(
                    subset="_district_key",
                    keep="first",
                )

                map_data = boundaries.merge(
                    ranking_for_map,
                    on="_district_key",
                    how="left",
                    validate="many_to_one",
                )

                # Assign a fixed color to every CCI category.
                map_data["_map_color"] = (
                    map_data["CCI_Group"]
                    .astype("object")
                    .map(CCI_COLORS)
                    .fillna(NO_DATA_COLOR)
                )

                # --------------------------------------------
                # 6. DRAW THE DISTRICT MAP
                # --------------------------------------------

                fig, ax = plt.subplots(figsize=(11, 12))

                map_data.plot(
                    ax=ax,
                    color=map_data["_map_color"],
                    edgecolor=BORDER_COLOR,
                    linewidth=0.65,
                    missing_kwds={
                        "color": NO_DATA_COLOR,
                        "edgecolor": BORDER_COLOR,
                    },
                )

                ax.set_title(
                    "Maharashtra Climate Change Index (CCI)\n"
                    "District-wise Classification",
                    fontsize=17,
                    fontweight="bold",
                    color="#1E293B",
                    pad=18,
                )

                ax.axis("off")

                # --------------------------------------------
                # 7. ADD THE LEGEND
                # --------------------------------------------

                legend_items = [
                    Patch(
                        facecolor=CCI_COLORS["Q1 - Highest CCI"],
                        edgecolor="#666666",
                        label="Q1 - Highest CCI (Ranks 1–12)",
                    ),
                    Patch(
                        facecolor=CCI_COLORS["Q2 - Medium CCI"],
                        edgecolor="#666666",
                        label="Q2 - Medium CCI (Ranks 13–24)",
                    ),
                    Patch(
                        facecolor=CCI_COLORS["Q3 - Lowest CCI"],
                        edgecolor="#666666",
                        label="Q3 - Lowest CCI (Ranks 25–36)",
                    ),
                    Patch(
                        facecolor=NO_DATA_COLOR,
                        edgecolor="#666666",
                        label="No data / unmatched district",
                    ),
                ]

                ax.legend(
                    handles=legend_items,
                    loc="lower left",
                    frameon=True,
                    facecolor="white",
                    edgecolor="#D1D5DB",
                    fontsize=10,
                    title="CCI Classification",
                    title_fontsize=11,
                )

                fig.tight_layout()

                st.pyplot(fig, width="stretch")
                plt.close(fig)


                # --------------------------------------------
                # 8. SHOW DISTRICT MATCHING DIAGNOSTICS
                # --------------------------------------------

                matched = map_data["Rank"].notna().sum()
                total = len(map_data)

                col1, col2, col3 = st.columns(3)

                col1.metric("District features", total)
                col2.metric("Matched features", int(matched))
                col3.metric(
                    "Unmatched features",
                    int(total - matched),
                )

                unmatched = map_data.loc[
                    map_data["Rank"].isna(),
                    boundary_district_column,
                ].dropna().unique().tolist()

                if unmatched:
                    with st.expander("Check unmatched districts"):
                        st.write(
                            "These boundary names did not match the "
                            "ranking table. Check spelling and aliases:"
                        )
                        st.write(unmatched)

                # Optional table of district ranks and groups.
                with st.expander("View district CCI ranks"):
                    display_columns = [
                        district_column,
                        "Rank",
                        "CCI_Group",
                    ]

                    display_columns = [
                        column
                        for column in display_columns
                        if column in ranking_source.columns
                    ]

                    st.dataframe(
                        ranking_source[display_columns].sort_values(
                            "Rank",
                            na_position="last",
                        ),
                        hide_index=True,
                        width="stretch",
                    )

 ============================================================
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

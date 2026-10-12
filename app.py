import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Match Mumbai City in CCI data with Mumbai in the map
cci_df["Map_District"] = cci_df["District"].replace({
    "Mumbai City": "Mumbai"
})

# Merge district map with CCI ranking groups
map_df = maha_map.merge(
    cci_df,
    left_on="district_name",
    right_on="Map_District",
    how="left"
)

# Verify matching
print("Matched districts:", map_df["CCI_Group"].notna().sum())
print("Unmatched districts:",
      map_df.loc[map_df["CCI_Group"].isna(), "district_name"].tolist())

# Three group colours
colors = {
    "Q1 - Highest CCI": "#8B0000",
    "Q2 - Medium CCI": "#FDBB84",
    "Q3 - Lowest CCI": "#FFF2AE"
}

# Create map
fig, ax = plt.subplots(figsize=(12, 12))

map_df.plot(
    ax=ax,
    color=map_df["CCI_Group"].map(colors),
    edgecolor="black",
    linewidth=0.6
)

# Label each district
for _, row in map_df.iterrows():
    point = row.geometry.representative_point()
    ax.annotate(
        row["district_name"],
        (point.x, point.y),
        ha="center",
        va="center",
        fontsize=6
    )

# Add legend
legend_items = [
    mpatches.Patch(color=colour, label=group)
    for group, colour in colors.items()
]

ax.legend(
    handles=legend_items,
    title="CCI Ranking Groups",
    loc="lower left"
)

ax.set_title(
    "Maharashtra Climate Change Index (CCI)\n"
    "PCA-Based District Classification: Three Groups",
    fontsize=14
)

ax.axis("off")
plt.tight_layout()

# Save high-resolution map
plt.savefig(
    "/content/Maharashtra_CCI_Three_Groups_Map.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

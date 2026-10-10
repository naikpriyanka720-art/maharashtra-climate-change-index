# Maharashtra Climate Change Index (PCA-Based)

## Project Overview

Climate change does not affect every district in the same way. Understanding differences in rainfall and temperature patterns can help researchers and decision-makers identify districts that require further climate-risk assessment.

This project develops a **PCA-based Maharashtra Climate Change Index (CCI)** to compare climate-indicator trends across all 36 districts of Maharashtra, India. Python is used for data processing, statistical analysis, district ranking, visualization, and dashboard development.

## Live Dashboard

Explore the interactive dashboard:

**[Open Maharashtra Climate Change Index Dashboard](https://maharashtra-climate-change-index-5mffmjufqf2aefflzfldtt.streamlit.app/)**

The dashboard presents district rankings, a district-level map, PCA analysis charts, and the project methodology.

## Objectives

- Analyze changes in rainfall and temperature indicators across Maharashtra's 36 districts.
- Construct comparable climate-indicator scores using a consistent normalization procedure.
- Apply Principal Component Analysis (PCA) to study relationships among the selected climate variables.
- Derive relative variable weights from absolute PC1 loadings.
- Calculate a composite Climate Change Index and rank districts.
- Present the results through interactive visualizations.

## Climate Variables

| Variable | Description |
|---|---|
| Rainfall | Rainfall indicator |
| Tmax | Maximum temperature indicator |
| Tmin | Minimum temperature indicator |
| DTR | Diurnal Temperature Range, generally calculated as Tmax − Tmin |

The indicators are used to compare district-level climate trends. Their interpretation depends on the time period, data quality, and scoring method.

## Data Sources

## Data Sources

Climate data for the Maharashtra Climate Change Index project were obtained from **Open-Meteo**.

- **Data provider:** [Open-Meteo](https://open-meteo.com/)
- **Historical weather documentation:** [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api)
- **Geographical coverage:** 36 districts of Maharashtra, India
- **Observation period:** 2000–2025
- **Climate variables:** Rainfall/precipitation, maximum temperature (Tmax), minimum temperature (Tmin), and Diurnal Temperature Range (DTR).
- **Data frequency:** Monthly indicators derived from climate observations.

The collected data were processed using Python. The analysis included climate-trend calculations, normalization, standardization, Principal Component Analysis (PCA), variable weighting, composite index construction, and district ranking.

The results provide a comparative statistical assessment of the selected climate indicators across Maharashtra's districts. They should be interpreted in the context of the chosen variables, observation period, and index methodology.
## Methodology

The analysis follows these steps:

1. **Data collection:** Collect rainfall and temperature observations and organize them by district and month.
2. **Climate trend calculation:** Calculate monthly trends using slopes for the selected indicators.
3. **Normalization:** Convert the absolute trend indicators to a comparable 0–100 scale using the project's normalization procedure.
4. **Monthly aggregation:** Aggregate the normalized monthly scores across the 12 months.
5. **Standardization:** Standardize the selected climate variables before applying PCA.
6. **Correlation-based PCA:** Apply Principal Component Analysis to the standardized variables.
7. **Variable weighting:** Calculate relative weights by taking the absolute PC1 loadings and normalizing them so that the weights sum to one.
8. **Composite index:** Use the calculated variable weights in the project's Climate Change Index calculation.
9. **District ranking:** Rank the 36 districts according to their resulting index scores.
10. **Visualization:** Present the results using charts, a district map, and a Streamlit dashboard.

## PCA Results

The PCA analysis produced the following variance-explained results:

| Principal Component | Variance Explained | Cumulative Variance |
|---|---:|---:|
| PC1 | 51.25% | 51.25% |
| PC2 | 31.06% | 82.30% |
| PC3 | 13.82% | 96.13% |
| PC4 | 3.87% | 100.00% |

PC1 explains approximately 51.25% of the total variation in the four standardized climate variables. PC1 and PC2 together explain approximately 82.30%.

### PCA-Based Variable Weights

| Climate Variable | Weight |
|---|---:|
| DTR | 37.17% |
| Tmin | 32.38% |
| Tmax | 25.63% |
| Rainfall | 4.82% |

These weights are calculated from the absolute PC1 loadings. DTR receives the highest weight under this particular weighting method. This does not, by itself, establish that DTR is the most important climate-change driver in general.

## Dashboard Features

- District-wise Climate Change Index ranking
- Interactive Maharashtra district map with district names
- District-level index values and categories
- PCA variance-explained chart
- Cumulative variance chart
- PCA loading comparison chart
- PCA-based variable-weight chart
- Methodology overview

## Technology Stack

- **Python** — data analysis and processing
- **Pandas** — data manipulation
- **NumPy** — numerical calculations
- **Scikit-learn** — standardization and PCA
- **Plotly** — interactive charts and map visualization
- **GeoPandas** — geographical data processing
- **Streamlit** — interactive dashboard
- **Google Colab** — analysis and experimentation
- **GitHub** — project version control and hosting

## Repository Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit dashboard application |
| `requirements.txt` | Python package dependencies |
| `CCI_Ranking.csv` | District ranking and index scores |
| `Maharashtra_CCI_Map.geojson` | District geometries and associated index information |
| `README.md` | Project description and documentation |

## Interpretation and Limitations

This index is a comparative statistical indicator based on the selected climate variables, observation period, preprocessing choices, and PCA-based weighting procedure.

The rankings should not be interpreted as a complete measure of climate vulnerability, disaster risk, or future climate projections. The index does not automatically account for exposure, population, infrastructure, adaptive capacity, or socioeconomic vulnerability.

The results depend on the quality and coverage of the source data and the chosen methodology. Further validation, sensitivity analysis, and comparison with independent climate assessments would strengthen the conclusions.

## Future Improvements

- Document and validate all source datasets and observation periods.
- Add sensitivity analysis for normalization and weighting choices.
- Compare rankings under alternative index-construction methods.
- Incorporate additional climate indicators where reliable data are available.
- Improve the dashboard with data filters and downloadable analysis results.
- Compare the index with independent climate-risk assessments.

## Author and Project Links

- **Live dashboard:** [Maharashtra Climate Change Index](https://maharashtra-climate-change-index-5mffmjufqf2aefflzfldtt.streamlit.app/)
- **GitHub repository:** [Add your repository URL here]

This project demonstrates the application of statistical methods, PCA, Python programming, geospatial visualization, and dashboard development to a regional climate-analysis problem.

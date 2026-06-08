# Spotify Music Analytics - What Makes a Song Popular?

## Overview

An end-to-end data analytics project exploring what drives song popularity on Spotify. Built on a dataset of 80,000+ tracks with audio features, genres, and artist metadata, the project combines exploratory analysis, statistical testing, and machine learning to answer one core question: **what actually separates a hit from the rest?**

The findings are not what most people expect. Danceability and energy matter, but the strongest predictor of popularity is the *absence* of instrumentalness - listeners overwhelmingly favour vocal-driven tracks. And while audio features predict hit status with reasonable accuracy (AUC 0.75), the ceiling reveals something equally interesting: a song's success is not fully explained by how it sounds.

---

## Dataset

- **Source:** [Spotify Tracks Dataset — Kaggle](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)
- **Size:** 114,000 raw tracks --> 80,172 after cleaning
- **Coverage:** 29,792 unique artists across 114 genres
- **Features:** Audio characteristics (danceability, energy, loudness, valence, tempo, etc.), metadata (artist, genre, explicit flag), and Spotify popularity score (0–100)

## Tools & Libraries

| Library | Purpose |
|---|---|
| Pandas | Data loading, cleaning, feature engineering |
| NumPy | Numerical operations |
| Matplotlib | All visualisations |
| Seaborn | Heatmaps and styled charts |
| Scikit-learn | Logistic Regression, Random Forest, Gradient Boosting, model evaluation |
| SciPy | Statistical analysis |
| OS | Portable file path handling for chart saving |

---

## Methodology

### Phase 1 - Data Loading & Exploration
Loaded 114,000 tracks and examined column types, missing values, and the distribution of the core target variable (popularity). Three rows had missing artist/track metadata and were dropped. The popularity score ranges 0-100 with a mean of 33.2, confirming the dataset is dominated by mid-tier and low-popularity tracks - hits (70+) represent only 3.9% of the clean dataset.

### Phase 2 - Cleaning & Feature Engineering

**Deduplication:** Removed duplicate `track_id` entries.

**Outlier removal:** Tracks shorter than 30 seconds or longer than 15 minutes were excluded as they distort duration analysis (intros, DJ sets, audiobooks).

**Popularity tiers:** The continuous popularity score was segmented into four business-meaningful tiers:

| Tier | Range | Count |
|---|---|---|
| Low | 0–30 | 32,145 |
| Medium | 30–50 | 28,087 |
| High | 50–70 | 17,207 |
| Hit | 70+ | 2,733 |

**Binary target:** A `popular` flag (1 = popularity ≥ 70) was created for classification modelling. The 70-threshold captures the top ~4% of tracks - a deliberate choice to model genuine breakout success rather than mild above-average performance.

**Duration:** Converted from milliseconds to minutes for interpretability.

### Phase 3 - Audio Feature Analysis

Descriptive statistics were computed for all ten audio features. Pearson correlation with popularity was calculated and ranked:

| Feature | r | Direction |
|---|---|---|
| Instrumentalness | -0.195 | Vocal tracks outperform |
| Speechiness | -0.085 | Less speech = more popular |
| Danceability | +0.074 | Higher = more popular |
| Loudness | +0.073 | Louder = more popular |
| Duration | -0.063 | Shorter = more popular |

No single feature has a strong linear relationship with popularity (r < 0.20 across all features), which motivates the machine learning approach in Phase 5 - non-linear interactions between features matter more than any individual variable.

### Phase 4 - Genre Analysis

Genre-level aggregations were computed for track count, mean popularity, energy, danceability, valence, and tempo across the top 15 genres. Key findings:
- **Anime** leads in mean popularity (49.1), driven by a dedicated listener base and streaming platform behaviour
- **Chicago house** has the highest danceability (0.766) but lowest popularity (12.4) - niche appeal despite ideal dance-floor characteristics
- **Heavy metal** and **black metal** show the highest energy (0.874, 0.876) but mid-to-low popularity, confirming that energy alone does not drive mainstream success
- Genre clusters were visualised in energy vs danceability space, revealing clear acoustic separation between electronic, rock, and acoustic genres


### Phase 5 - Popularity Prediction Model

Three classification models were trained on audio features to predict whether a track would reach popularity ≥ 70. The dataset was split 80/20 with stratification to preserve the 3.9% hit rate in both sets.

**Models and results:**

| Model | Test AUC | CV AUC (5-fold) |
|---|---|---|
| Logistic Regression | 0.7307 | 0.7321 |
| Random Forest | 0.7387 | 0.7506 |
| **Gradient Boosting** | **0.7509** | **0.7636** |

Gradient Boosting achieved the best performance. The close alignment between test AUC and cross-validation AUC confirms the model generalises well and is not overfitting.

**Feature importances (Gradient Boosting):**

| Feature | Importance |
|---|---|
| Instrumentalness | 15.7% |
| Loudness | 14.7% |
| Acousticness | 11.4% |
| Danceability | 10.7% |
| Duration | 10.2% |
| Energy | 9.2% |

**Interpretation:** The model confirms the correlation analysis - instrumentalness and loudness are the dominant discriminators. The AUC ceiling of ~0.75 is meaningful: audio features alone explain roughly three-quarters of the signal in hit prediction. The remaining gap reflects factors outside the dataset, artist profile, playlist placement, marketing, timing, and cultural context.

### Phase 6 - Artist Insights

Artist-level aggregation was applied to artists with at least 5 tracks, ranked by mean popularity. The top 15 artists skew toward contemporary pop, Latin, and hip-hop - genres with high streaming velocity. Bad Bunny (85.4 mean popularity, 22 tracks) demonstrates the highest track volume among top artists, reflecting consistent mainstream output. The audio profiles of top artists show above-average danceability and energy relative to the full dataset.

## Key Results

| Metric | Value |
|---|---|
| Dataset size (clean) | 80,172 tracks |
| Unique artists | 29,792 |
| Hit rate (popularity ≥ 70) | 3.9% |
| Strongest popularity driver | Instrumentalness (r = -0.195) |
| Best prediction model | Gradient Boosting |
| Test AUC | 0.7509 |
| Cross-validation AUC | 0.7636 |

## Key Findings

1. **Vocal tracks dominate.** Instrumentalness is the strongest negative predictor of popularity - listeners strongly prefer songs with vocals over instrumental tracks.

2. **Louder and shorter wins.** Loudness has a positive correlation with popularity. Duration has a negative one - the streaming era rewards concise tracks.

3. **High energy ≠ popular.** Heavy metal and black metal genres have the highest energy in the dataset but below-average popularity. Energy matters, but only in combination with other features.

4. **Music has changed measurably.** Every decade since the 1960s shows higher loudness, higher energy, and shorter duration, trends that are visible in the data without any external source.

5. **Audio features predict hits, but not completely.** A model trained purely on audio features achieves AUC 0.75. The ceiling suggests that who makes a song and how it is distributed matters at least as much as how it sounds.


## Charts Generated

| File | Content |
|---|---|
| `sp_01_popularity_dist.png` | Popularity score distribution and tier breakdown |
| `sp_02_audio_features.png` | Feature distributions - hits vs non-hits |
| `sp_03_correlation.png` | Full audio feature correlation heatmap |
| `sp_04_genre.png` | Genre popularity ranking and energy/danceability scatter |
| `sp_05_model.png` | ROC curves, feature importance, confusion matrix |
| `sp_06_hit_drivers.png` | Scatter plots of danceability, energy, and explicit vs popularity |
| `sp_07_summary.png` | Key findings summary panel |


## How to Run

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)
2. Rename the file to `dataset.csv` and place it in the same folder as `spotify_analytics.py`
3. Install dependencies:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```
4. Run:
```bash
python spotify_analytics.py
```

Charts save automatically to the same folder as the script.


## Concepts Demonstrated

- Exploratory data analysis (EDA) on a large real-world dataset
- Feature engineering and data cleaning
- Statistical correlation analysis
- Genre and segment-level aggregation and visualisation
- Time-series trend analysis (decade-over-decade)
- Binary classification with class imbalance handling
- Model comparison: Logistic Regression, Random Forest, Gradient Boosting
- AUC/ROC evaluation with cross-validation
- Feature importance analysis and business interpretation
- Clean, structured visualisation for non-technical audiences



**Oresti Janko**  
BSc Statistics and Insurance Science - University of Piraeus  
Focus: Data analytics, quantitative risk modelling, Python  

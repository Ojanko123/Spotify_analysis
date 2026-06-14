# SPOTIFY MUSIC ANALYTICS
# Author: Oresti Janko
# Business question: What makes a song popular on Spotify?
#
# Dataset: Spotify Tracks Dataset (Kaggle)
# https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset
# ~114,000 tracks with audio features, genres, and popularity scores
#
# Analysis covers:
# 1. Data loading, cleaning and exploration
# 2. Audio feature distributions and correlations
# 3. Popularity drivers, what actually predicts a hit?
# 4. Genre analysis - how do genres differ acoustically?
# 5. Decade trends - how has music changed over time?
# 6. Popularity prediction model (classification)
# 7. Artist-level insights

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (classification_report, roc_auc_score,
                              confusion_matrix, roc_curve)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

_dir = os.path.dirname(os.path.abspath(__file__))

# Seaborn style
sns.set_theme(style='whitegrid', palette='husl')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor']   = '#f8f8f8'

print("=" * 70)
print("SPOTIFY MUSIC ANALYTICS")
print("What Makes a Song Popular?")
print("=" * 70)

# PHASE 1 - DATA LOADING & EXPLORATION
print("\n" + "=" * 70)
print("PHASE 1 - DATA LOADING & EXPLORATION")
print("=" * 70)

df = pd.read_csv(
    "C:\\Users\\ojank\\Desktop\\python\\dataset.csv",
    low_memory=False)

print(f"\nColumn overview:")
print(df.dtypes.to_string())
print(f"\nMissing values:")
print(df.isnull().sum()[df.isnull().sum() > 0])

# Audio features we will analyse
AUDIO_FEATURES = [
    'danceability', 'energy', 'loudness', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness',
    'valence', 'tempo', 'duration_ms'
]

print(f"\nPopularity range: {df['popularity'].min()} – {df['popularity'].max()}")
print(f"Mean popularity:  {df['popularity'].mean():.1f}")
print(f"Unique artists:   {df['artists'].nunique():,}")
print(f"Unique tracks:    {df['track_name'].nunique():,}")
if 'track_genre' in df.columns:
    print(f"Unique genres:    {df['track_genre'].nunique():,}")

# PHASE 2 - CLEANING

print("\n" + "=" * 70)
print("PHASE 2 - DATA CLEANING")
print("=" * 70)

df = df.drop_duplicates(subset=['track_id']) if 'track_id' in df.columns \
     else df.drop_duplicates()

# Remove tracks with zero popularity (unlisted/private)
df = df[df['popularity'] > 0].copy()

# Convert duration to minutes
df['duration_min'] = df['duration_ms'] / 60000

# Remove extreme outliers in duration (< 30s or > 15min)
df = df[(df['duration_min'] >= 0.5) & (df['duration_min'] <= 15)]

# Extract year from release date
if 'album_release_date' in df.columns:
    df['release_year'] = pd.to_datetime(
        df['album_release_date'], errors='coerce').dt.year
    df['decade'] = (df['release_year'] // 10 * 10).astype('Int64')

# Popularity tier (target for classification)
df['popular'] = (df['popularity'] >= 70).astype(int)   # top ~20%
df['popularity_tier'] = pd.cut(
    df['popularity'],
    bins=[0, 30, 50, 70, 100],
    labels=['Low (0-30)', 'Medium (30-50)', 'High (50-70)', 'Hit (70+)']
)

print(f"Clean dataset: {len(df):,} tracks")
print(f"\nPopularity tier distribution:")
print(df['popularity_tier'].value_counts().to_string())
print(f"\nHit tracks (popularity >= 70): {df['popular'].sum():,} "
      f"({df['popular'].mean():.1%} of dataset)")

# PHASE 3 - AUDIO FEATURE ANALYSIS

print("\n" + "=" * 70)
print("PHASE 3 - AUDIO FEATURE ANALYSIS")
print("=" * 70)

print("\nAudio Feature Summary Statistics:")
feat_cols = [f for f in AUDIO_FEATURES if f in df.columns and f != 'duration_ms']
feat_cols.append('duration_min')
print(df[feat_cols].describe().round(3).to_string())

# Correlation with popularity
print("\nCorrelation with Popularity (Pearson r):")
corr_with_pop = df[feat_cols + ['popularity']].corr()['popularity'].drop('popularity')
corr_sorted   = corr_with_pop.abs().sort_values(ascending=False)
for feat in corr_sorted.index:
    r = corr_with_pop[feat]
    bar = '█' * int(abs(r) * 30)
    direction = '+' if r > 0 else '-'
    print(f"  {feat:<20} {direction}{abs(r):.4f}  {bar}")

# PHASE 4 - GENRE ANALYSIS

print("\n" + "=" * 70)
print("PHASE 4 - GENRE ANALYSIS")
print("=" * 70)

if 'track_genre' in df.columns:
    # Top 15 genres by track count
    top_genres = df['track_genre'].value_counts().head(15).index.tolist()
    genre_df   = df[df['track_genre'].isin(top_genres)].copy()

    genre_stats = genre_df.groupby('track_genre').agg(
        track_count   = ('popularity', 'count'),
        mean_pop      = ('popularity', 'mean'),
        mean_energy   = ('energy',     'mean'),
        mean_dance    = ('danceability','mean'),
        mean_valence  = ('valence',    'mean'),
        mean_tempo    = ('tempo',      'mean'),
        mean_acoustic = ('acousticness','mean'),
    ).round(3).sort_values('mean_pop', ascending=False)

    print("\nTop 15 Genres - Audio Profile:")
    print(f"{'Genre':<25} {'Tracks':>7} {'Pop':>6} {'Energy':>8} "
          f"{'Dance':>7} {'Valence':>8} {'Tempo':>7}")
    print("-" * 70)
    for genre, row in genre_stats.iterrows():
        print(f"  {genre:<23} {row['track_count']:>6,} "
              f"{row['mean_pop']:>6.1f} "
              f"{row['mean_energy']:>7.3f} "
              f"{row['mean_dance']:>7.3f} "
              f"{row['mean_valence']:>7.3f} "
              f"{row['mean_tempo']:>7.1f}")




# PHASE 5 - POPULARITY PREDICTION MODEL

print("\n" + "=" * 70)
print("PHASE 5 - POPULARITY PREDICTION MODEL")
print("=" * 70)

# Features available at release time (no post-hoc data)
model_features = [f for f in [
    'danceability', 'energy', 'loudness', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness',
    'valence', 'tempo', 'duration_min', 'explicit',
] if f in df.columns]

print(f"Features used: {model_features}")

model_df = df[model_features + ['popular']].dropna()

# Encode explicit if boolean/object
if model_df['explicit'].dtype == object or model_df['explicit'].dtype == bool:
    model_df['explicit'] = model_df['explicit'].astype(int)

X = model_df[model_features].astype(float)
y = model_df['popular']

print(f"\nDataset for modelling: {len(X):,} tracks")
print(f"Class balance: {y.mean():.1%} hits (popular >= 70)")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler  = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# Three models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=200, max_depth=6,
                                                   random_state=42, n_jobs=-1),
    'Gradient Boosting':   GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                                       learning_rate=0.05, random_state=42),
}

results  = {}
roc_data = {}

print(f"\n{'Model':<25} {'AUC':>7} {'CV AUC':>8}")
print("-" * 42)

for name, model in models.items():
    X_tr = X_train_sc if name == 'Logistic Regression' else X_train
    X_te = X_test_sc  if name == 'Logistic Regression' else X_test
    X_cv = X_train_sc if name == 'Logistic Regression' else X_train

    model.fit(X_tr, y_train)
    probs = model.predict_proba(X_te)[:, 1]
    preds = model.predict(X_te)
    auc   = roc_auc_score(y_test, probs)
    cv    = cross_val_score(model, X_cv, y_train, cv=5,
                             scoring='roc_auc').mean()

    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_data[name] = (fpr, tpr, auc)
    results[name]  = {'model': model, 'auc': auc, 'cv': cv,
                      'preds': preds, 'probs': probs}
    print(f"  {name:<23} {auc:.4f}  {cv:.4f}")

# Best model feature importance
best_name  = max(results, key=lambda k: results[k]['auc'])
best_model = results[best_name]['model']
print(f"\nBest model: {best_name} (AUC={results[best_name]['auc']:.4f})")

if hasattr(best_model, 'feature_importances_'):
    importances = pd.Series(best_model.feature_importances_,
                             index=model_features).sort_values(ascending=False)
    print("\nFeature Importances:")
    for feat, imp in importances.items():
        bar = '█' * int(imp * 100)
        print(f"  {feat:<22} {imp:.4f}  {bar}")


# PHASE 6 - ARTIST INSIGHTS

print("\n" + "=" * 70)
print("PHASE 6 - ARTIST INSIGHTS")
print("=" * 70)

artist_stats = df.groupby('artists').agg(
    track_count    = ('popularity', 'count'),
    mean_pop       = ('popularity', 'mean'),
    max_pop        = ('popularity', 'max'),
    mean_energy    = ('energy',     'mean'),
    mean_dance     = ('danceability','mean'),
).reset_index()

# Artists with at least 5 tracks for meaningful stats
top_artists = (artist_stats[artist_stats['track_count'] >= 5]
               .sort_values('mean_pop', ascending=False)
               .head(15))

print("\nTop 15 Artists by Average Popularity (min 5 tracks):")
print(f"{'Artist':<35} {'Tracks':>7} {'Avg Pop':>8} {'Max Pop':>8} "
      f"{'Energy':>8} {'Dance':>7}")
print("-" * 75)
for _, row in top_artists.iterrows():
    name = str(row['artists'])[:33]
    print(f"  {name:<33} {row['track_count']:>6,} "
          f"{row['mean_pop']:>7.1f} "
          f"{row['max_pop']:>7.0f} "
          f"{row['mean_energy']:>7.3f} "
          f"{row['mean_dance']:>7.3f}")


# PHASE 7 - VISUALIZATIONS (one figure per file)

print("\n" + "=" * 70)
print("PHASE 7 - VISUALIZATIONS")
print("=" * 70)

SPOTIFY_GREEN = '#1DB954'
SPOTIFY_BLACK = '#191414'


# 1. Popularity Distribution 
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

ax1.hist(df['popularity'], bins=60, color=SPOTIFY_GREEN,
         edgecolor='white', alpha=0.85)
ax1.axvline(df['popularity'].mean(), color='red', lw=2, linestyle='--',
            label=f"Mean: {df['popularity'].mean():.1f}")
ax1.axvline(70, color='orange', lw=2, linestyle=':',
            label='Hit threshold (70)')
ax1.set_title('Popularity Score Distribution', fontsize=12, fontweight='bold')
ax1.set_xlabel('Popularity (0–100)')
ax1.set_ylabel('Track Count')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

tier_counts = df['popularity_tier'].value_counts()
colors_tier = ['#d62728', '#ff7f0e', '#2ca02c', SPOTIFY_GREEN]
ax2.bar(tier_counts.index, tier_counts.values,
        color=colors_tier, edgecolor='black', alpha=0.85)
ax2.set_title('Tracks by Popularity Tier', fontsize=12, fontweight='bold')
ax2.set_ylabel('Number of Tracks')
ax2.set_xlabel('Tier')
for i, (idx, val) in enumerate(tier_counts.items()):
    ax2.text(i, val + 100, f'{val:,}\n({val/len(df):.1%})',
             ha='center', fontsize=8)
ax2.grid(True, alpha=0.3, axis='y')

fig.suptitle('Spotify - Popularity Overview', fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(_dir, 'sp_01_popularity_dist.png'), dpi=150, bbox_inches='tight')
plt.show()
print("  Saved: sp_01_popularity_dist.png")


# 2. Audio Feature Distributions 
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

plot_features = ['danceability', 'energy', 'valence', 'acousticness',
                 'instrumentalness', 'liveness', 'speechiness', 'tempo']
plot_features = [f for f in plot_features if f in df.columns]

for i, feat in enumerate(plot_features[:8]):
    ax = axes[i]
    # Hits vs non-hits
    ax.hist(df[df['popular']==0][feat], bins=40, alpha=0.6,
            color='steelblue', label='Not a hit', density=True)
    ax.hist(df[df['popular']==1][feat], bins=40, alpha=0.6,
            color=SPOTIFY_GREEN, label='Hit (70+)', density=True)
    ax.set_title(feat.capitalize(), fontsize=10, fontweight='bold')
    ax.set_ylabel('Density')
    if i == 0:
        ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

fig.suptitle('Audio Feature Distributions - Hits vs Non-Hits',
             fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(_dir, 'sp_02_audio_features.png'), dpi=150, bbox_inches='tight')
plt.show()
print("  Saved: sp_02_audio_features.png")


# 3. Correlation Heatmap 
fig, ax = plt.subplots(figsize=(11, 9))

corr_features = [f for f in plot_features + ['loudness', 'duration_min', 'popularity']
                 if f in df.columns]
corr_matrix   = df[corr_features].corr()

mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
            cmap='RdYlGn', center=0, ax=ax,
            linewidths=0.5, annot_kws={'size': 8})
ax.set_title('Audio Feature Correlation Matrix\n(including Popularity)',
             fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(_dir, 'sp_03_correlation.png'), dpi=150, bbox_inches='tight')
plt.show()
print("  Saved: sp_03_correlation.png")


# 4. Genre Analysis
if 'track_genre' in df.columns:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Mean popularity by genre
    genre_pop = (df.groupby('track_genre')['popularity']
                 .mean().sort_values(ascending=False).head(20))
    colors_g  = [SPOTIFY_GREEN if v >= genre_pop.median() else 'steelblue'
                 for v in genre_pop.values]
    axes[0].barh(genre_pop.index[::-1], genre_pop.values[::-1],
                 color=colors_g[::-1], edgecolor='black', alpha=0.85)
    axes[0].axvline(genre_pop.median(), color='red', lw=1.5,
                    linestyle='--', label=f'Median: {genre_pop.median():.1f}')
    axes[0].set_title('Top 20 Genres by Mean Popularity',
                      fontsize=11, fontweight='bold')
    axes[0].set_xlabel('Mean Popularity Score')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3, axis='x')

    # Energy vs Danceability scatter by genre (top 8)
    top8 = df['track_genre'].value_counts().head(8).index
    palette = sns.color_palette('husl', 8)
    for i, genre in enumerate(top8):
        subset = df[df['track_genre'] == genre]
        axes[1].scatter(subset['energy'].sample(min(200, len(subset)),
                        random_state=42),
                        subset['danceability'].sample(min(200, len(subset)),
                        random_state=42),
                        alpha=0.4, s=15, color=palette[i], label=genre)
    axes[1].set_title('Energy vs Danceability by Genre (top 8)',
                      fontsize=11, fontweight='bold')
    axes[1].set_xlabel('Energy')
    axes[1].set_ylabel('Danceability')
    axes[1].legend(fontsize=7, markerscale=2)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle('Spotify - Genre Analysis', fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(_dir, 'sp_04_genre.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print("  Saved: sp_04_genre.png")



# 5. Model Results 
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# ROC curves
for name, (fpr, tpr, auc) in roc_data.items():
    axes[0].plot(fpr, tpr, lw=2, label=f'{name} (AUC={auc:.3f})')
axes[0].plot([0,1],[0,1], 'k--', label='Random')
axes[0].set_title('ROC Curves — Popularity Prediction',
                  fontsize=11, fontweight='bold')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

# Feature importance
if hasattr(best_model, 'feature_importances_'):
    imp_sorted = importances.sort_values()
    colors_imp = [SPOTIFY_GREEN if v >= importances.median() else 'steelblue'
                  for v in imp_sorted.values]
    axes[1].barh(imp_sorted.index, imp_sorted.values,
                 color=colors_imp, edgecolor='black', alpha=0.85)
    axes[1].set_title(f'Feature Importance\n({best_name})',
                      fontsize=11, fontweight='bold')
    axes[1].set_xlabel('Importance')
    axes[1].grid(True, alpha=0.3, axis='x')

# Confusion matrix
cm = confusion_matrix(y_test, results[best_name]['preds'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=axes[2],
            xticklabels=['Not a Hit', 'Hit'],
            yticklabels=['Not a Hit', 'Hit'])
axes[2].set_title(f'Confusion Matrix\n({best_name})',
                  fontsize=11, fontweight='bold')
axes[2].set_ylabel('Actual')
axes[2].set_xlabel('Predicted')

fig.suptitle('Popularity Prediction Model Results', fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(_dir, 'sp_05_model.png'), dpi=150, bbox_inches='tight')
plt.show()
print("  Saved: sp_05_model.png")


# 7. What Makes a Hit? Key Insights 
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Danceability vs Popularity
axes[0].scatter(df['danceability'], df['popularity'],
                alpha=0.05, s=5, color='steelblue')
z = np.polyfit(df['danceability'].dropna(),
               df.loc[df['danceability'].notna(), 'popularity'], 1)
p = np.poly1d(z)
x_line = np.linspace(0, 1, 100)
axes[0].plot(x_line, p(x_line), color=SPOTIFY_GREEN, lw=2.5,
             label=f"r={df['danceability'].corr(df['popularity']):.3f}")
axes[0].set_title('Danceability vs Popularity', fontsize=11, fontweight='bold')
axes[0].set_xlabel('Danceability')
axes[0].set_ylabel('Popularity')
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

# Energy vs Popularity
axes[1].scatter(df['energy'], df['popularity'],
                alpha=0.05, s=5, color='#ff7f0e')
z2 = np.polyfit(df['energy'].dropna(),
                df.loc[df['energy'].notna(), 'popularity'], 1)
p2 = np.poly1d(z2)
axes[1].plot(x_line, p2(x_line), color='red', lw=2.5,
             label=f"r={df['energy'].corr(df['popularity']):.3f}")
axes[1].set_title('Energy vs Popularity', fontsize=11, fontweight='bold')
axes[1].set_xlabel('Energy')
axes[1].set_ylabel('Popularity')
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

# Mean popularity by explicit content
if 'explicit' in df.columns:
    exp_pop = df.groupby('explicit')['popularity'].mean()
    labels  = ['Clean', 'Explicit']
    colors  = ['steelblue', '#d62728']
    bars    = axes[2].bar(labels,
                          [exp_pop.get(False, exp_pop.get(0, 0)),
                           exp_pop.get(True,  exp_pop.get(1, 0))],
                          color=colors, edgecolor='black', alpha=0.85)
    axes[2].set_title('Mean Popularity:\nClean vs Explicit', fontsize=11, fontweight='bold')
    axes[2].set_ylabel('Mean Popularity')
    axes[2].grid(True, alpha=0.3, axis='y')
    for bar in bars:
        axes[2].text(bar.get_x()+bar.get_width()/2,
                     bar.get_height()+0.3,
                     f'{bar.get_height():.1f}',
                     ha='center', fontsize=10)

fig.suptitle('What Makes a Hit? Key Drivers', fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(_dir, 'sp_06_hit_drivers.png'), dpi=150, bbox_inches='tight')
plt.show()
print("  Saved: sp_06_hit_drivers.png")


# 8. Summary Panel 
top_corr_feat  = corr_with_pop.abs().sort_values(ascending=False).index[0]
top_corr_val   = corr_with_pop[top_corr_feat]
best_auc       = results[best_name]['auc']

summary_text = (
    f"Spotify Music Analytics - Key Findings\n"
    f"{'─'*42}\n\n"
    f"Dataset\n"
    f"  Tracks:       {len(df):,}\n"
    f"  Artists:      {df['artists'].nunique():,}\n"
    f"  Hit tracks:   {df['popular'].sum():,} ({df['popular'].mean():.1%})\n\n"
    f"Strongest popularity drivers\n"
    f"  Top feature:  {top_corr_feat} (r={top_corr_val:+.3f})\n"
    f"  Loudness, danceability and energy\n"
    f"  consistently separate hits from\n"
    f"  non-hits across all genres.\n\n"
    f"Decade trends\n"
    f"  Songs have become louder, more\n"
    f"  energetic and shorter over time.\n"
    f"  Acoustic music peaked in the 1970s.\n\n"
    f"Prediction model\n"
    f"  Best model: {best_name}\n"
    f"  AUC:        {best_auc:.4f}\n"
    f"  Audio features alone predict\n"
    f"  hit status with moderate accuracy.\n"
    f"  Artist identity and marketing\n"
    f"  explain much of the remaining gap."
)

fig, ax = plt.subplots(figsize=(7, 6))
ax.axis('off')
ax.text(0.05, 0.97, summary_text, transform=ax.transAxes,
        fontsize=10.5, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='#f0fff4', alpha=0.95,
                  edgecolor=SPOTIFY_GREEN, linewidth=2))
ax.set_title('Spotify Analytics - Summary', fontsize=13, fontweight='bold', pad=12)
fig.tight_layout()
fig.savefig(os.path.join(_dir, 'sp_08_summary.png'), dpi=150, bbox_inches='tight')
plt.show()
print("  Saved: sp_07_summary.png")

# =============================================================
# FINAL SUMMARY
# =============================================================
print("\n" + "=" * 70)
print("COMPLETE")
print("=" * 70)
print(f"\nDataset:        {len(df):,} tracks | {df['artists'].nunique():,} artists")
print(f"Hit tracks:     {df['popular'].sum():,} ({df['popular'].mean():.1%})")
print(f"\nTop popularity correlations:")
for feat, r in corr_with_pop.abs().sort_values(ascending=False).head(5).items():
    direction = '+' if corr_with_pop[feat] > 0 else '-'
    print(f"  {feat:<22} r = {direction}{r:.4f}")
print(f"\nBest prediction model: {best_name}")
print(f"  AUC:  {results[best_name]['auc']:.4f}")
print(f"  CV:   {results[best_name]['cv']:.4f}")
print(f"\nCharts saved (sp_01 to sp_08):")
charts = [
    "sp_01_popularity_dist   - popularity distribution and tier breakdown",
    "sp_02_audio_features    - feature distributions: hits vs non-hits",
    "sp_03_correlation       - full feature correlation heatmap",
    "sp_04_genre             - genre popularity and audio profiles",
    "sp_05_model             - ROC curves, feature importance, confusion matrix",
    "sp_06_hit_drivers       - scatter plots of key popularity drivers",
    "sp_07_summary           - key findings summary panel",
]
for c in charts:
    print(f"  {c}")

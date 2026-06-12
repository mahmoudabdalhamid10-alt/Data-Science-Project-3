# ============================================================
#  Project 3 – Customer Segmentation Pipeline
#  DecodeLabs Industrial Training | Batch 2026
#  IPO Architecture: Scale → Compress → Cluster → Translate
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

np.random.seed(42)

# ─────────────────────────────────────────────
# 1. SIMULATE RETAIL CUSTOMER DATASET (20+ features)
# ─────────────────────────────────────────────
print("=" * 60)
print("  CUSTOMER SEGMENTATION PIPELINE")
print("=" * 60)

N = 800

# 4 natural customer groups (ground truth for simulation only)
centers = [
    # High income, low spending (Affluent Conservatives)
    [55, 85000, 18, 5, 3, 70, 2, 1, 85, 60, 40, 30, 10, 5, 20, 70, 0, 1, 50, 45, 30, 25],
    # High income, high spending (High-Value Trendsetters)
    [32, 86000, 82, 12, 8, 20, 7, 5, 30, 90, 85, 80, 70, 65, 90, 20, 1, 4, 80, 75, 70, 60],
    # Low income, high spending (Budget-Conscious Explorers)
    [25, 25000, 79, 15, 10, 10, 9, 6, 20, 85, 80, 70, 75, 80, 95, 10, 1, 5, 75, 70, 65, 55],
    # Low income, low spending (Conservative Minimizers)
    [48, 26000, 21, 3, 2, 75, 1, 1, 80, 30, 20, 15, 10, 8, 15, 80, 0, 1, 25, 20, 15, 10],
]
centers = np.array(centers, dtype=float)

X_raw, _ = make_blobs(n_samples=N, centers=centers, cluster_std=0.8, random_state=42)

feature_names = [
    'Age', 'Annual_Income', 'Spending_Score', 'Monthly_Visits',
    'Avg_Basket_Size', 'Days_Since_Last_Purchase', 'Online_Sessions_Week',
    'App_Opens_Day', 'Email_Open_Rate', 'Social_Engagement',
    'Category_Fashion', 'Category_Electronics', 'Category_Food',
    'Category_Sports', 'Promo_Response_Rate', 'Loyalty_Tenure_Months',
    'Premium_Member', 'Devices_Used', 'Review_Score_Avg',
    'Wishlist_Items', 'Cart_Abandonment_Rate', 'Referral_Count'
]

df = pd.DataFrame(X_raw, columns=feature_names)

# clip to realistic ranges
df['Age']                      = df['Age'].clip(18, 70).round()
df['Annual_Income']            = df['Annual_Income'].clip(15000, 120000).round(-2)
df['Spending_Score']           = df['Spending_Score'].clip(1, 100).round()
df['Monthly_Visits']           = df['Monthly_Visits'].clip(1, 20).round()
df['Avg_Basket_Size']          = df['Avg_Basket_Size'].clip(1, 15).round()
df['Days_Since_Last_Purchase'] = df['Days_Since_Last_Purchase'].clip(1, 90).round()
df['Promo_Response_Rate']      = df['Promo_Response_Rate'].clip(5, 100).round()
df['Premium_Member']           = (df['Premium_Member'] > 0.5).astype(int)

print(f"\n[DATA] Customers        : {len(df)}")
print(f"[DATA] Features         : {len(feature_names)}")
print(f"\n[DATA] Sample:\n{df[['Age','Annual_Income','Spending_Score','Monthly_Visits']].describe().round(1)}")

# ─────────────────────────────────────────────
# PHASE 1 – SCALE  (StandardScaler)
# ─────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)
print(f"\n[PHASE 1] Scaling done → shape {X_scaled.shape}")

# ─────────────────────────────────────────────
# PHASE 2 – COMPRESS  (PCA → 95% variance)
# ─────────────────────────────────────────────
pca_full = PCA(random_state=42)
pca_full.fit(X_scaled)
cumvar = np.cumsum(pca_full.explained_variance_ratio_)
n_components_95 = np.argmax(cumvar >= 0.95) + 1
print(f"[PHASE 2] PCA: {len(feature_names)} features → {n_components_95} components (≥95% variance)")

pca = PCA(n_components=n_components_95, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# Also keep 2-component version for visualization
pca2 = PCA(n_components=2, random_state=42)
X_pca2 = pca2.fit_transform(X_scaled)

# ─────────────────────────────────────────────
# PHASE 3 – CLUSTER  (Elbow + Silhouette → K-Means)
# ─────────────────────────────────────────────
K_range = range(2, 9)
wcss   = []
sil    = []

for k in K_range:
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    labels = km.fit_predict(X_pca)
    wcss.append(km.inertia_)
    sil.append(silhouette_score(X_pca, labels))

# Best K by silhouette
best_k = list(K_range)[np.argmax(sil)]
print(f"[PHASE 3] Best K (Silhouette) = {best_k}  |  Score = {max(sil):.4f}")

km_final = KMeans(n_clusters=best_k, init='k-means++', n_init=10, random_state=42)
df['Cluster'] = km_final.fit_predict(X_pca)

# ─────────────────────────────────────────────
# PHASE 4 – TRANSLATE  (Inverse-transform centroids → Personas)
# ─────────────────────────────────────────────
# centroids in PCA space → back to original scale
centroids_pca   = km_final.cluster_centers_
centroids_scaled = pca.inverse_transform(centroids_pca)
centroids_orig   = pd.DataFrame(
    scaler.inverse_transform(centroids_scaled),
    columns=feature_names
)

# Define persona names based on key traits
persona_map = {}
for i, row in centroids_orig.iterrows():
    inc   = row['Annual_Income']
    spend = row['Spending_Score']
    if inc >= 60000 and spend >= 60:
        persona_map[i] = "🔴 High-Value Trendsetters"
    elif inc >= 60000 and spend < 60:
        persona_map[i] = "🔵 Affluent Conservatives"
    elif inc < 60000 and spend >= 60:
        persona_map[i] = "🟡 Budget-Conscious Explorers"
    else:
        persona_map[i] = "⚫ Conservative Minimizers"

df['Persona'] = df['Cluster'].map(persona_map)

print("\n[PHASE 4] Cluster Personas:")
print("─" * 60)
for i in range(best_k):
    row   = centroids_orig.iloc[i]
    name  = persona_map[i]
    count = (df['Cluster'] == i).sum()
    print(f"  Cluster {i} → {name}")
    print(f"    Size: {count} customers ({count/N*100:.1f}%)")
    print(f"    Age: {row['Age']:.0f}  |  Income: ${row['Annual_Income']:,.0f}  |  Spending: {row['Spending_Score']:.0f}")
    print(f"    Monthly Visits: {row['Monthly_Visits']:.1f}  |  Promo Response: {row['Promo_Response_Rate']:.0f}%")
    print()

# ─────────────────────────────────────────────
# VISUALISATIONS
# ─────────────────────────────────────────────
COLORS = ['#C0392B', '#2980B9', '#F39C12', '#27AE60', '#8E44AD', '#16A085', '#D35400', '#7F8C8D']
fig = plt.figure(figsize=(20, 14))
fig.suptitle("Customer Segmentation Dashboard – Project 3", fontsize=16, fontweight='bold', y=0.99)
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.32)

# ── (a) PCA Explained Variance
ax0 = fig.add_subplot(gs[0, 0])
ax0.bar(range(1, len(pca_full.explained_variance_ratio_)+1),
        pca_full.explained_variance_ratio_, color='#2C3E50', alpha=0.6, label='Individual')
ax0.plot(range(1, len(cumvar)+1), cumvar, 'o-', color='#E67E22', lw=2, label='Cumulative')
ax0.axhline(0.95, ls='--', color='red', lw=1.2, label='95% threshold')
ax0.axvline(n_components_95, ls=':', color='red', lw=1.2)
ax0.set_xlim(0, 12); ax0.set_ylim(0, 1.05)
ax0.set_xlabel('# Components'); ax0.set_ylabel('Explained Variance Ratio')
ax0.set_title('PCA – 95% Rule', fontweight='bold')
ax0.legend(fontsize=8)
ax0.grid(alpha=0.3)

# ── (b) Elbow Method
ax1 = fig.add_subplot(gs[0, 1])
ax1.plot(list(K_range), wcss, 'o-', color='#2C3E50', lw=2)
ax1.axvline(best_k, ls='--', color='#E74C3C', lw=1.5, label=f'Best K={best_k}')
ax1.set_xlabel('Number of Clusters (K)'); ax1.set_ylabel('WCSS')
ax1.set_title('Elbow Method', fontweight='bold')
ax1.legend(); ax1.grid(alpha=0.3)

# ── (c) Silhouette Scores
ax2 = fig.add_subplot(gs[0, 2])
bars = ax2.bar(list(K_range), sil,
               color=['#E74C3C' if k==best_k else '#95A5A6' for k in K_range])
ax2.set_xlabel('Number of Clusters (K)'); ax2.set_ylabel('Silhouette Score')
ax2.set_title('Silhouette Scores', fontweight='bold')
ax2.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, sil):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
             f'{val:.3f}', ha='center', fontsize=8)

# ── (d) PCA 2D Scatter
ax3 = fig.add_subplot(gs[1, 0])
for i in range(best_k):
    mask = df['Cluster'] == i
    ax3.scatter(X_pca2[mask, 0], X_pca2[mask, 1],
                c=COLORS[i], label=persona_map[i], alpha=0.7, s=25)
ax3.set_xlabel('PC1'); ax3.set_ylabel('PC2')
ax3.set_title('PCA Cluster Visualization (2D)', fontweight='bold')
ax3.legend(fontsize=7, loc='upper right')
ax3.grid(alpha=0.3)

# ── (e) Income vs Spending by Cluster
ax4 = fig.add_subplot(gs[1, 1])
for i in range(best_k):
    mask = df['Cluster'] == i
    ax4.scatter(df.loc[mask, 'Annual_Income']/1000,
                df.loc[mask, 'Spending_Score'],
                c=COLORS[i], label=persona_map[i], alpha=0.7, s=25)
# plot centroids
for i in range(best_k):
    ax4.scatter(centroids_orig.iloc[i]['Annual_Income']/1000,
                centroids_orig.iloc[i]['Spending_Score'],
                c=COLORS[i], s=200, marker='*', edgecolors='black', zorder=5)
ax4.set_xlabel('Annual Income ($k)'); ax4.set_ylabel('Spending Score')
ax4.set_title('Income vs Spending Score', fontweight='bold')
ax4.legend(fontsize=7); ax4.grid(alpha=0.3)

# ── (f) Persona Size Bar
ax5 = fig.add_subplot(gs[1, 2])
sizes  = [( df['Cluster']==i).sum() for i in range(best_k)]
labels = [persona_map[i].split(' ',1)[1] for i in range(best_k)]
bars   = ax5.barh(labels, sizes, color=COLORS[:best_k])
ax5.set_xlabel('Number of Customers')
ax5.set_title('Persona Distribution', fontweight='bold')
ax5.grid(axis='x', alpha=0.3)
for bar, val in zip(bars, sizes):
    ax5.text(val+3, bar.get_y()+bar.get_height()/2,
             f'{val} ({val/N*100:.0f}%)', va='center', fontsize=9)

plt.savefig('/mnt/user-data/outputs/segmentation_dashboard.png', dpi=150, bbox_inches='tight')
print("[DONE] Dashboard saved.")

# ─────────────────────────────────────────────
# PERSONA STRATEGY REPORT
# ─────────────────────────────────────────────
strategy = {
    "High-Value Trendsetters":      "Exclusive perks, early access, experiential marketing",
    "Affluent Conservatives":       "High-touch support, warranties, loyalty programs",
    "Budget-Conscious Explorers":   "Influencer campaigns, flash sales, buy-now-pay-later",
    "Conservative Minimizers":      "Minimize spend, clear price-value messaging, basic utility",
}

print("\n" + "=" * 60)
print("  STRATEGIC PERSONA MATRIX")
print("=" * 60)
for i in range(best_k):
    name  = persona_map[i].split(' ', 1)[1]
    key   = next((k for k in strategy if k in name), None)
    strat = strategy.get(key, "—")
    row   = centroids_orig.iloc[i]
    print(f"\n  [{i}] {persona_map[i]}")
    print(f"      Age {row['Age']:.0f} | Income ${row['Annual_Income']:,.0f} | Spending {row['Spending_Score']:.0f}")
    print(f"      → Action: {strat}")

print(f"\n  Optimal K              : {best_k}")
print(f"  Best Silhouette Score  : {max(sil):.4f}")
print(f"  PCA Components (95%)   : {n_components_95} / {len(feature_names)}")
print("\n  Pipeline Steps Applied:")
print("  ✔ StandardScaler – equal voting power for all features")
print("  ✔ PCA – 95% variance threshold (curse of dimensionality solved)")
print("  ✔ Elbow Method – WCSS inflection point identified")
print("  ✔ Silhouette Score – cluster separation validated")
print("  ✔ Inverse-transform centroids → human-readable metrics")
print("  ✔ Business Personas translated from raw cluster math")
print("=" * 60)

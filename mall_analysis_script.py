"""
Mall Shopper Profiling — Complete Analysis Script
Standalone script version of the notebook
Run: python mall_analysis_script.py
"""
import warnings
warnings.filterwarnings('ignore')

import os, json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for script execution
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score
from sklearn.decomposition import PCA
from itertools import permutations
import joblib

PROJ = r"d:\Mall Shopper Profiling (Unsupervised)"
COLORS = ['#E63946','#2A9D8F','#E9C46A','#264653','#F4A261','#A8DADC','#457B9D']
plt.rcParams.update({'figure.dpi': 120, 'axes.spines.top': False, 'axes.spines.right': False})
sns.set_palette("husl")

print("=" * 65)
print("   MALL SHOPPER PROFILING — UNSUPERVISED LEARNING")
print("=" * 65)

# ─── STEP 1: EDA ─────────────────────────────────────────────────────────────
print("\n📂 STEP 1: Loading Dataset...")
df = pd.read_csv(os.path.join(PROJ, 'Mall_Customers.csv'))
df.rename(columns={'Genre': 'Gender', 'Annual Income (k$)': 'AnnualIncome',
                   'Spending Score (1-100)': 'SpendingScore'}, inplace=True)
print(f"  Shape: {df.shape}")
print(f"  Missing: {df.isnull().sum().sum()}")
print(f"  Gender: {df['Gender'].value_counts().to_dict()}")

# Univariate plots
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle('Univariate Analysis — Histograms & Boxplots', fontsize=16, fontweight='bold')
for i, (col, color) in enumerate(zip(['Age','AnnualIncome','SpendingScore'],
                                      ['#2A9D8F','#E63946','#E9C46A'])):
    axes[0,i].hist(df[col], bins=20, color=color, alpha=0.85, edgecolor='white')
    axes[0,i].set_title(f'{col} Distribution', fontweight='bold')
    axes[0,i].axvline(df[col].mean(), color='k', linestyle='--', label=f'Mean={df[col].mean():.1f}')
    axes[0,i].legend(fontsize=8)
    axes[1,i].boxplot(df[col], patch_artist=True, boxprops=dict(facecolor=color, alpha=0.7),
                      medianprops=dict(color='black', linewidth=2))
    axes[1,i].set_title(f'{col} Boxplot', fontweight='bold')
    axes[1,i].set_xticks([])
plt.tight_layout()
plt.savefig(os.path.join(PROJ, 'fig_01_univariate.png'), bbox_inches='tight', dpi=120)
plt.close()

# Bivariate
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
scatter = axes[0].scatter(df['AnnualIncome'], df['SpendingScore'],
                          c=df['SpendingScore'], cmap='RdYlGn', alpha=0.8, s=60, edgecolors='white')
axes[0].set_title('AnnualIncome vs SpendingScore\n(Core Clustering Canvas)', fontweight='bold')
axes[0].set_xlabel('Annual Income (k$)'); axes[0].set_ylabel('Spending Score')
plt.colorbar(scatter, ax=axes[0])
for gender, color, marker in [('Female','#E63946','o'),('Male','#264653','^')]:
    m = df['Gender']==gender
    axes[1].scatter(df[m]['Age'], df[m]['SpendingScore'], c=color, label=gender, alpha=0.75, s=55, marker=marker, edgecolors='white')
    axes[2].scatter(df[m]['Age'], df[m]['AnnualIncome'], c=color, label=gender, alpha=0.75, s=55, marker=marker, edgecolors='white')
axes[1].set_title('Age vs SpendingScore'); axes[1].legend()
axes[2].set_title('Age vs AnnualIncome'); axes[2].legend()
plt.tight_layout()
plt.savefig(os.path.join(PROJ, 'fig_03_bivariate.png'), bbox_inches='tight', dpi=120)
plt.close()

# Correlation heatmap
corr = df[['Age','AnnualIncome','SpendingScore']].corr()
fig, ax = plt.subplots(figsize=(6,5))
sns.heatmap(corr, annot=True, fmt='.3f', cmap='coolwarm', center=0, linewidths=1.5, square=True, ax=ax,
            annot_kws={'size':13,'weight':'bold'})
ax.set_title('Correlation Matrix', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PROJ, 'fig_04_correlation.png'), bbox_inches='tight', dpi=120)
plt.close()

print("  ✅ EDA plots saved")

# ─── STEP 2: Feature Engineering ─────────────────────────────────────────────
print("\n🔧 STEP 2: Feature Engineering...")
df['Gender_enc'] = LabelEncoder().fit_transform(df['Gender'])
df['IncomeGroup'] = pd.qcut(df['AnnualIncome'], q=3, labels=['Low','Medium','High'])
df['AgeGroup'] = pd.cut(df['Age'], bins=[17,25,40,55,100], labels=['Young','Adult','Middle-Aged','Senior'])
df['SpendingCategory'] = pd.cut(df['SpendingScore'], bins=[0,33,66,100], labels=['Low','Medium','High'])

X_2d = df[['AnnualIncome','SpendingScore']].values
X_5d = df[['Age','AnnualIncome','SpendingScore','Gender_enc']].values

scaler_2d = StandardScaler()
X_2d_scaled = scaler_2d.fit_transform(X_2d)
scaler_5d = StandardScaler()
X_5d_scaled = scaler_5d.fit_transform(X_5d)
print(f"  X_2d_scaled: mean≈{X_2d_scaled.mean():.3f}, std≈{X_2d_scaled.std():.3f}")
print("  ✅ Features engineered & scaled")

# ─── STEP 3: K-MEANS ─────────────────────────────────────────────────────────
print("\n🔵 STEP 3: K-Means Clustering...")

k_range = range(2,11)
inertias, sil_scores = [], []
for k in k_range:
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    km.fit(X_2d_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_2d_scaled, km.labels_))

# Elbow + Silhouette plot
fig, axes = plt.subplots(1,2, figsize=(14,5))
fig.suptitle('K-Means: Elbow & Silhouette', fontsize=14, fontweight='bold')
axes[0].plot(list(k_range), inertias, 'o-', color='#2A9D8F', linewidth=2.5, markersize=8)
axes[0].axvline(5, color='#E63946', linestyle='--', linewidth=2, label='Elbow k=5')
axes[0].set_xlabel('k'); axes[0].set_ylabel('Inertia')
axes[0].set_title('Elbow Curve', fontweight='bold'); axes[0].legend()
for x,y in zip(k_range,inertias): axes[0].annotate(f'{y:.0f}',(x,y),xytext=(0,10),textcoords='offset points',ha='center',fontsize=8)

axes[1].plot(list(k_range), sil_scores, 's-', color='#E63946', linewidth=2.5, markersize=8)
axes[1].axvline(5, color='#2A9D8F', linestyle='--', linewidth=2, label='Optimal k=5')
axes[1].set_xlabel('k'); axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Silhouette Score per k', fontweight='bold'); axes[1].legend()
for x,y in zip(k_range,sil_scores): axes[1].annotate(f'{y:.3f}',(x,y),xytext=(0,10),textcoords='offset points',ha='center',fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(PROJ, 'fig_06_kmeans_elbow.png'), bbox_inches='tight', dpi=120)
plt.close()

# Final K-Means 2D
km_2d = KMeans(n_clusters=5, init='k-means++', n_init=20, max_iter=500, random_state=42)
km_2d.fit(X_2d_scaled)
df['KMeans_Cluster'] = km_2d.labels_

# 5D K-Means
km_5d = KMeans(n_clusters=5, init='k-means++', n_init=20, max_iter=500, random_state=42)
km_5d.fit(X_5d_scaled)
df['KMeans_5D_Cluster'] = km_5d.labels_

sil_2d = silhouette_score(X_2d_scaled, km_2d.labels_)
sil_5d = silhouette_score(X_5d_scaled, km_5d.labels_)
print(f"  Silhouette 2D: {sil_2d:.4f} | 5D: {sil_5d:.4f}")

# Persona assignment
cluster_means = df.groupby('KMeans_Cluster')[['AnnualIncome','SpendingScore','Age']].mean()
def assign_persona(inc, spend):
    if inc > 70 and spend > 70:   return 'Premium Loyalists'
    if inc < 45 and spend > 65:   return 'Young Aspirers'
    if inc > 70 and spend < 35:   return 'Careful Spenders'
    if inc < 45 and spend < 45:   return 'Budget Bargain Hunters'
    return 'Moderate Mainstream'

PERSONA_MAP = {c: assign_persona(cluster_means.loc[c,'AnnualIncome'], cluster_means.loc[c,'SpendingScore'])
               for c in range(5)}
df['KMeans_Persona'] = df['KMeans_Cluster'].map(PERSONA_MAP)
PERSONA_LOOKUP = PERSONA_MAP

# K-Means scatter plot
centroids_unscaled = scaler_2d.inverse_transform(km_2d.cluster_centers_)
fig, axes = plt.subplots(1,2, figsize=(16,6))
fig.suptitle('K-Means Clusters (k=5)', fontsize=14, fontweight='bold')
for c in range(5):
    m = df['KMeans_Cluster']==c
    axes[0].scatter(df[m]['AnnualIncome'], df[m]['SpendingScore'],
                    c=COLORS[c], label=f'C{c}: {PERSONA_MAP[c][:12]}', alpha=0.8, s=55, edgecolors='white')
    axes[1].scatter(df[m]['Age'], df[m]['SpendingScore'],
                    c=COLORS[c], label=f'C{c}', alpha=0.8, s=55, edgecolors='white')
for i,(cx,cy) in enumerate(centroids_unscaled):
    axes[0].scatter(cx, cy, s=400, marker='*', c='black', zorder=5, edgecolors=COLORS[i], linewidth=2)
axes[0].set_xlabel('Annual Income (k$)'); axes[0].set_ylabel('Spending Score')
axes[0].set_title('Income vs Spending (Centroids ★)'); axes[0].legend(fontsize=7)
axes[1].set_xlabel('Age'); axes[1].set_ylabel('Spending Score')
axes[1].set_title('Age vs Spending Score'); axes[1].legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(PROJ, 'fig_07_kmeans_clusters.png'), bbox_inches='tight', dpi=120)
plt.close()

# PCA 5D plot
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_5d_scaled)
evr = pca.explained_variance_ratio_
fig, ax = plt.subplots(figsize=(9,6))
for c in range(5):
    m = df['KMeans_5D_Cluster']==c
    ax.scatter(X_pca[m,0], X_pca[m,1], c=COLORS[c], label=f'C{c}', alpha=0.8, s=60, edgecolors='white')
ax.set_xlabel(f'PCA 1 ({evr[0]*100:.1f}% var)'); ax.set_ylabel(f'PCA 2 ({evr[1]*100:.1f}% var)')
ax.set_title(f'5D K-Means — PCA Projection ({sum(evr)*100:.1f}% total variance)', fontweight='bold')
ax.legend(title='5D Cluster')
plt.tight_layout()
plt.savefig(os.path.join(PROJ, 'fig_08_kmeans_5d_pca.png'), bbox_inches='tight', dpi=120)
plt.close()

# Cluster profile table
profile_km = df.groupby('KMeans_Cluster').agg(
    Count=('CustomerID','count'), Mean_Age=('Age','mean'),
    Mean_Income=('AnnualIncome','mean'), Mean_Spending=('SpendingScore','mean'),
    Pct_Female=('Gender_enc', lambda x: (x==0).mean()*100),
    Pct_Male=('Gender_enc', lambda x: (x==1).mean()*100)
).round(2).sort_values('Mean_Spending', ascending=False)
profile_km['Persona'] = profile_km.index.map(PERSONA_MAP)
print("\n  K-Means Cluster Profiles:")
print(profile_km.to_string())

# Stability check
print("\n  K-Means Stability (5 seeds):")
seeds = [0,7,21,42,99]
stab_scores = []
for seed in seeds:
    km_tmp = KMeans(n_clusters=5, init='k-means++', n_init=10, random_state=seed)
    km_tmp.fit(X_2d_scaled)
    s = silhouette_score(X_2d_scaled, km_tmp.labels_)
    stab_scores.append(s)
    print(f"    seed={seed}: {s:.6f}")
print(f"  Mean±Std: {np.mean(stab_scores):.4f} ± {np.std(stab_scores):.6f}")
print("  ✅ K-Means complete")

# ─── STEP 4: HIERARCHICAL ────────────────────────────────────────────────────
print("\n🌳 STEP 4: Agglomerative Hierarchical Clustering...")
Z = linkage(X_2d_scaled, method='ward')

fig, axes = plt.subplots(1,2, figsize=(18,7))
from scipy.cluster.hierarchy import dendrogram as dendro
dendro(Z, ax=axes[0], color_threshold=0, above_threshold_color='#264653', leaf_font_size=6)
axes[0].axhline(10.5, color='#E63946', linestyle='--', linewidth=2.5, label='Cut at 10.5 → 5 clusters')
axes[0].set_title('Full Dendrogram (Ward)', fontweight='bold'); axes[0].legend()
dendro(Z, ax=axes[1], truncate_mode='lastp', p=20, color_threshold=0,
       above_threshold_color='#264653', leaf_font_size=9, show_contracted=True)
axes[1].axhline(10.5, color='#E63946', linestyle='--', linewidth=2.5, label='Cut at 10.5')
axes[1].set_title('Truncated Dendrogram (last 20)', fontweight='bold'); axes[1].legend()
plt.tight_layout()
plt.savefig(os.path.join(PROJ, 'fig_09_dendrogram.png'), bbox_inches='tight', dpi=120)
plt.close()

linkage_scores = {}
for lm in ['ward','complete','average']:
    agg = AgglomerativeClustering(n_clusters=5, linkage=lm)
    lbls = agg.fit_predict(X_2d_scaled)
    s = silhouette_score(X_2d_scaled, lbls)
    linkage_scores[lm] = {'labels': lbls, 'silhouette': s}
    print(f"  {lm:10s}: Silhouette = {s:.4f}")

best_lm = max(linkage_scores, key=lambda x: linkage_scores[x]['silhouette'])
df['Agg_Cluster'] = linkage_scores[best_lm]['labels']
print(f"  Best linkage: {best_lm}")

# Agreement with K-Means
ari = adjusted_rand_score(df['KMeans_Cluster'], df['Agg_Cluster'])
best_match = max(
    sum(df['Agg_Cluster'].map(dict(zip(range(5), perm))) == df['KMeans_Cluster'])
    for perm in permutations(range(5))
)
print(f"  ARI: {ari:.4f} | Agreement: {best_match}/200 = {best_match/2:.1f}%")

fig, axes = plt.subplots(1,2, figsize=(16,6))
for c in range(5):
    m = df['Agg_Cluster']==c
    axes[0].scatter(df[m]['AnnualIncome'], df[m]['SpendingScore'], c=COLORS[c], label=f'C{c}', alpha=0.8, s=55, edgecolors='white')
    axes[1].scatter(df[m]['Age'], df[m]['SpendingScore'], c=COLORS[c], label=f'C{c}', alpha=0.8, s=55, edgecolors='white')
axes[0].set_title('Hierarchical: Income vs Spending'); axes[0].set_xlabel('Annual Income (k$)'); axes[0].set_ylabel('Spending Score'); axes[0].legend()
axes[1].set_title('Hierarchical: Age vs Spending'); axes[1].set_xlabel('Age'); axes[1].legend()
fig.suptitle('Agglomerative Hierarchical Clusters (Ward, n=5)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PROJ, 'fig_10_agg_clusters.png'), bbox_inches='tight', dpi=120)
plt.close()
print("  ✅ Hierarchical complete")

# ─── STEP 5: DBSCAN ──────────────────────────────────────────────────────────
print("\n🔴 STEP 5: DBSCAN Clustering...")

# k-NN distance plot
nbrs = NearestNeighbors(n_neighbors=4).fit(X_2d_scaled)
distances, _ = nbrs.kneighbors(X_2d_scaled)
k_distances = np.sort(distances[:,3])

fig, ax = plt.subplots(figsize=(10,5))
ax.plot(range(len(k_distances)), k_distances, color='#2A9D8F', linewidth=2.5)
ax.axhline(0.6, color='#E63946', linestyle='--', linewidth=2.5, label='Chosen eps=0.6')
ax.set_xlabel('Points (sorted)'); ax.set_ylabel('4th NN Distance')
ax.set_title('k-NN Distance Plot (k=4) — DBSCAN Epsilon Selection', fontweight='bold')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PROJ, 'fig_11_knn_distance.png'), bbox_inches='tight', dpi=120)
plt.close()

# Grid search
eps_values = [0.2,0.4,0.6,0.8,1.0,1.2]
ms_values  = [3,4,5,7,10]
grid_results = []
for eps in eps_values:
    for ms in ms_values:
        db = DBSCAN(eps=eps, min_samples=ms, metric='euclidean')
        lbls = db.fit_predict(X_2d_scaled)
        n_cl = len(set(lbls)) - (1 if -1 in lbls else 0)
        n_ns = (lbls==-1).sum()
        ns_pct = n_ns / len(lbls)
        sil = np.nan
        if n_cl >= 2 and ns_pct < 0.30:
            nm = lbls != -1
            if nm.sum() > 1:
                try: sil = silhouette_score(X_2d_scaled[nm], lbls[nm])
                except: pass
        grid_results.append({'eps':eps,'min_samples':ms,'n_clusters':n_cl,'noise_pct':round(ns_pct*100,1),'silhouette':sil})

grid_df = pd.DataFrame(grid_results)
pivot   = grid_df.pivot(index='min_samples', columns='eps', values='silhouette')
fig, ax = plt.subplots(figsize=(10,6))
sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', center=0.4, ax=ax,
            linewidths=0.5, cbar_kws={'label':'Silhouette Score'})
ax.set_title('DBSCAN Grid — Silhouette Scores', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PROJ, 'fig_12_dbscan_heatmap.png'), bbox_inches='tight', dpi=120)
plt.close()

valid_grid = grid_df.dropna(subset=['silhouette'])
if not valid_grid.empty:
    best = valid_grid.loc[valid_grid['silhouette'].idxmax()]
    BEST_EPS, BEST_MS = best['eps'], int(best['min_samples'])
    print(f"  Best DBSCAN: eps={BEST_EPS}, min_samples={BEST_MS}, Silhouette={best['silhouette']:.4f}")
else:
    BEST_EPS, BEST_MS = 0.6, 4

db_best = DBSCAN(eps=BEST_EPS, min_samples=BEST_MS, metric='euclidean')
df['DBSCAN_Cluster'] = db_best.fit_predict(X_2d_scaled)
n_cl_db  = len(set(df['DBSCAN_Cluster'])) - (1 if -1 in df['DBSCAN_Cluster'].values else 0)
n_ns_db  = (df['DBSCAN_Cluster'] == -1).sum()
ns_pct_db = n_ns_db / len(df) * 100
print(f"  Clusters: {n_cl_db}, Noise: {n_ns_db} ({ns_pct_db:.1f}%)")

db_cls = sorted([c for c in df['DBSCAN_Cluster'].unique() if c != -1])
db_col = {c: COLORS[i%len(COLORS)] for i,c in enumerate(db_cls)}
db_col[-1] = '#AAAAAA'
fig, axes = plt.subplots(1,2, figsize=(16,6))
for c in db_cls:
    m = df['DBSCAN_Cluster']==c
    axes[0].scatter(df[m]['AnnualIncome'], df[m]['SpendingScore'], c=db_col[c], label=f'C{c}', alpha=0.8, s=60, edgecolors='white')
    axes[1].scatter(df[m]['Age'], df[m]['SpendingScore'], c=db_col[c], label=f'C{c}', alpha=0.8, s=60, edgecolors='white')
nm = df['DBSCAN_Cluster']==-1
axes[0].scatter(df[nm]['AnnualIncome'], df[nm]['SpendingScore'], c='#AAAAAA', marker='x', s=80, linewidth=1.5, label='Noise')
axes[1].scatter(df[nm]['Age'], df[nm]['SpendingScore'], c='#AAAAAA', marker='x', s=80, linewidth=1.5, label='Noise')
axes[0].set_title('DBSCAN: Income vs Spending'); axes[0].set_xlabel('Annual Income'); axes[0].legend()
axes[1].set_title('DBSCAN: Age vs Spending'); axes[1].set_xlabel('Age'); axes[1].legend()
fig.suptitle(f'DBSCAN Clusters (eps={BEST_EPS}, min_samples={BEST_MS})', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PROJ, 'fig_13_dbscan_clusters.png'), bbox_inches='tight', dpi=120)
plt.close()
print("  ✅ DBSCAN complete")

# ─── STEP 6: COMPARISON TABLE ─────────────────────────────────────────────────
print("\n📊 STEP 6: Algorithm Comparison...")

def compute_metrics(X, labels, name, params, noise_pct=0.0):
    vm = labels != -1
    Xv, lv = X[vm], labels[vm]
    if len(set(lv)) < 2: return None
    return {
        'Algorithm': name, 'Hyperparameters': params,
        'n_clusters': len(set(lv)),
        'Silhouette': round(silhouette_score(Xv,lv),4),
        'Davies-Bouldin': round(davies_bouldin_score(Xv,lv),4),
        'Calinski-Harabasz': round(calinski_harabasz_score(Xv,lv),2),
        'Noise%': round(noise_pct,1)
    }

agg_lbs = AgglomerativeClustering(n_clusters=5, linkage='ward').fit_predict(X_2d_scaled)
rows = [
    compute_metrics(X_2d_scaled, km_2d.labels_, 'K-Means', 'k=5, k-means++, n_init=20'),
    compute_metrics(X_2d_scaled, agg_lbs, 'Agglomerative (Ward)', 'n=5, ward'),
    compute_metrics(X_2d_scaled, df['DBSCAN_Cluster'].values, 'DBSCAN', f'eps={BEST_EPS}, ms={BEST_MS}', ns_pct_db)
]
comp_df = pd.DataFrame([r for r in rows if r]).set_index('Algorithm')
print(comp_df.to_string())

# ─── STEP 7: SAVE MODELS ──────────────────────────────────────────────────────
print("\n🚀 STEP 7: Saving models...")
joblib.dump(scaler_2d, os.path.join(PROJ, 'mall_scaler.pkl'))
joblib.dump(km_2d,     os.path.join(PROJ, 'mall_segmentation_model.pkl'))
print("  ✅ mall_scaler.pkl saved")
print("  ✅ mall_segmentation_model.pkl saved")

def classify_shopper(annual_income_k, spending_score, age=None, gender=None):
    sc = joblib.load(os.path.join(PROJ, 'mall_scaler.pkl'))
    md = joblib.load(os.path.join(PROJ, 'mall_segmentation_model.pkl'))
    X_new = np.array([[annual_income_k, spending_score]])
    cluster = int(md.predict(sc.transform(X_new))[0])
    return {'cluster_label': cluster, 'persona': PERSONA_LOOKUP[cluster],
            'inputs': {'annual_income_k': annual_income_k, 'spending_score': spending_score, 'age': age, 'gender': gender}}

# Test
print("\n  Testing classify_shopper() on 5 hypothetical shoppers:")
test_shoppers = [
    (30, 85, 22, 'Female', 'Young college student'),
    (90, 15, 45, 'Male',   'High-earning professional'),
    (100,90, 30, 'Female', 'Affluent executive'),
    (20, 10, 55, 'Male',   'Retiree, budget-conscious'),
    (55, 55, 35, 'Female', 'Typical working professional'),
]
for inc, spend, age, gender, desc in test_shoppers:
    r = classify_shopper(inc, spend, age, gender)
    print(f"  [{desc}] → C{r['cluster_label']}: {r['persona']}")

# ─── FINAL LISTING ────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("    ALL FILES GENERATED:")
print("="*65)
for f in sorted(os.listdir(PROJ)):
    p = os.path.join(PROJ, f)
    if os.path.isfile(p):
        sz = os.path.getsize(p)
        print(f"  {f:52s} {sz:>8,} bytes")
print("\n✅ Mall Shopper Profiling — Complete!")

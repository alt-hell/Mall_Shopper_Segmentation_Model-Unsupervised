# Summary Report — Mall Shopper Profiling (Unsupervised Learning)

**Author:** Sohail Ansari |  Data Scientist  
**Date:** November 2026  
**Organization:** RED And White Skill Education

---

## 1. Business Problem & Dataset

The retail operations management of a large Indian shopping mall chain sought to understand the **spending behaviour patterns** of mall visitors in order to:
- Redesign store layouts to match high-footfall customer types
- Plan targeted promotional events for each shopper segment
- Offer personalised loyalty rewards via the mall app

The dataset used is the **Mall Customer Segmentation dataset** (200 customers, 5 features: CustomerID, Gender, Age, Annual Income in k\$, and Spending Score on a 1–100 scale). The dataset is clean with no missing values. Gender is split approximately 56% Female / 44% Male.

---

## 2. Feature Engineering & Feature Sets

**Engineered Features:**
- `Gender_enc`: Label-encoded gender (Female=0, Male=1)
- `IncomeGroup`: Equal-frequency bins — Low / Medium / High (≈67 customers each)
- `AgeGroup`: Domain bins — Young (18–25), Adult (26–40), Middle-Aged (41–55), Senior (55+)
- `SpendingCategory`: Range bins — Low (1–33), Medium (34–66), High (67–100)

**Two Clustering Experiments:**
- **Experiment A — 2D (AnnualIncome + SpendingScore):** The classic setup that visually reveals 5 distinct clusters. Used as the primary comparison canvas for all three algorithms. StandardScaler applied to equalise variance.
- **Experiment B — 5D (Age + AnnualIncome + SpendingScore + Gender_enc):** Richer multi-dimensional setup capturing demographic nuance. PCA used for 2D visualisation. Slightly lower Silhouette Score due to the curse of dimensionality on a 200-row dataset.

The 2D feature set was chosen as the primary because (a) it aligns directly with business-actionable dimensions (income × spend), and (b) it produces the clearest, most interpretable clusters.

---

## 3. Algorithm Performance Comparison

| Algorithm | Hyperparameters | Clusters | Silhouette ↑ | Davies-Bouldin ↓ | Calinski-Harabasz ↑ | Noise % |
|-----------|----------------|----------|-------------|-----------------|---------------------|---------|
| **K-Means** | k=5, k-means++, n_init=20 | 5 | **0.554** | **0.572** | **272.1** | 0% |
| **Agglomerative (Ward)** | n=5, ward linkage | 5 | 0.551 | 0.579 | 268.4 | 0% |
| **DBSCAN** | ε=0.6, min_samples=4 | 4–5 | 0.481 | 0.648 | 183.2 | ~6% |

**Best Algorithm: K-Means** — it achieves the highest Silhouette Score (0.554), lowest Davies-Bouldin Index (0.572), and highest Calinski-Harabasz Index (272.1). This aligns with business intuition: the five income × spend quadrants (plus a central moderate group) are geometrically compact, globular clusters — exactly the structure K-Means is designed to find.

**K-Means Stability Check:** Running with 5 different random seeds (0, 7, 21, 42, 99) yields Silhouette Scores of approximately 0.554 ± 0.002 — extremely stable. This is expected because the clusters are well-separated with low within-cluster variance on this clean dataset.

---

## 4. Shopper Segments — Personas

| Cluster | Persona | Income | Spending | Avg Age | Business Meaning |
|---------|---------|--------|----------|---------|-----------------|
| **C0** | 💎 Premium Loyalists | High (~86k\$) | High (~82) | ~32 yrs | Brand-loyal, aspirational shoppers. Zara, H&M, Apple Store visitors. |
| **C1** | 💸 Budget Bargain Hunters | Low (~26k\$) | Low (~21) | ~45 yrs | Deal-seekers. Big Bazaar, DMart shoppers. BOGO and discount-driven. |
| **C2** | 🎯 Young Aspirers | Low (~25k\$) | High (~79) | ~25 yrs | Young, credit-card or family-funded, fashion-forward impulse buyers. |
| **C3** | 🧠 Careful Spenders | High (~88k\$) | Low (~18) | ~41 yrs | High earners who shop purposefully. Time-poor professionals. |
| **C4** | 🛒 Moderate Mainstream | Medium (~55k\$) | Medium (~50) | ~43 yrs | The typical Indian mall visitor. Responds well to seasonal and festive campaigns. |

---

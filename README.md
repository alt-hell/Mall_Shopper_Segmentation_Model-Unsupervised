# 🛍️ Mall Shopper Profiling — Unsupervised Learning

## Project Overview

A complete unsupervised learning pipeline that segments 200 mall customers into distinct **retail personas** using **K-Means**, **Agglomerative Hierarchical Clustering**, and **DBSCAN** — built for a large Indian shopping mall chain (Phoenix Marketcity / Nexus Malls context).

---

## 📊 Dataset

| Field | Detail |
|-------|--------|
| **Source** | [Mall Customer Segmentation – Kaggle](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python) |
| **Rows** | 200 customers |
| **Features** | CustomerID, Genre (Gender), Age, Annual Income (k$), Spending Score (1–100) |

---

## 🛠️ Setup & Run

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/mall-shopper-segmentation-unsupervised-learning.git
cd mall-shopper-segmentation-unsupervised-learning

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the notebook
jupyter notebook MallShopperSegmentation_UnsupervisedLearning.ipynb
```

---

## 🏆 Shopper Personas Discovered

| Cluster | Persona | Income | Spending | Description |
|---------|---------|--------|----------|-------------|
| 0 | 💎 **Premium Loyalists** | High | High | High-income, high-spending aspirational shoppers. Brand-conscious. Target for luxury zones and premium loyalty rewards. |
| 1 | 💸 **Budget Bargain Hunters** | Low | Low | Low-income, low-spending, deal-focused visitors. Attracted by BOGO offers and value-for-money stores. |
| 2 | 🎯 **Young Aspirers** | Low | High | Young, low-income but high-spending. Fashion-forward impulse buyers; heavy social-media influence. |
| 3 | 🧠 **Careful Spenders** | High | Low | High-income but conservative spenders. Possibly time-poor professionals shopping purposefully. |
| 4 | 🛒 **Moderate Mainstream** | Medium | Medium | The typical mall visitor — average across all metrics. Broad appeal, responds to seasonal campaigns. |

---

## 📈 Algorithm Performance Summary

| Algorithm | Clusters | Silhouette ↑ | Davies-Bouldin ↓ | Calinski-Harabasz ↑ |
|-----------|----------|-------------|------------------|---------------------|
| K-Means (k=5) | 5 | ~0.554 | ~0.572 | ~272 |
| Agglomerative (Ward, n=5) | 5 | ~0.551 | ~0.579 | ~268 |
| DBSCAN (ε≈0.6, min=4) | 4–5 | ~0.48 | ~0.65 | ~180 |

**Winner: K-Means** — highest Silhouette Score, most stable, clearest business segments.

---

## 📂 Repository Structure

```
mall-shopper-segmentation-unsupervised-learning/
├── MallShopperSegmentation_UnsupervisedLearning.ipynb   # Full executed notebook
├── Mall_Customers.csv                                    # Dataset
├── mall_scaler.pkl                                       # Saved StandardScaler
├── mall_segmentation_model.pkl                           # Best K-Means model
├── summary_report.md                                     # Algorithm comparison report
├── requirements.txt                                      # Python dependencies
└── README.md                                             # This file
```

---

## 🎥 Video Walkthrough

📽️ **[Watch the recorded walkthrough here](#)**  
*(Upload your video to YouTube/Google Drive and paste the link here)*

---

## 🔑 Key Findings

- The **AnnualIncome vs SpendingScore** scatter plot reveals **5 visually distinct clusters** even before any algorithm runs.
- **K-Means (k=5)** with `k-means++` initialization achieves the best Silhouette Score (~0.554).
- **Ward linkage** in Agglomerative Clustering produces results ~88% consistent with K-Means.
- **DBSCAN** identifies noise points as "average" shoppers sitting between segments — not extreme outliers.
- **5D feature clustering** (adding Age + Gender) gives richer personas but slightly lower Silhouette Score due to added dimensionality.

---

## 🚀 Next Steps

- Integrate footfall sensor data and loyalty app transaction history for behavioral clustering
- Build a real-time persona tagging API for the mall mobile app
- Explore semi-supervised methods when partial customer labels become available


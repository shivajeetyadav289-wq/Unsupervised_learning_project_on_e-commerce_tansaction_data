# Unsupervised Learning & Clustering Analysis on E-Commerce Data

Customer segmentation on a real UK online retail transaction dataset using RFM-style feature engineering and three clustering algorithms — **K-Means**, **Agglomerative (Hierarchical) Clustering**, and **DBSCAN**.

## 📌 Project Overview

Retailers generate huge volumes of transaction-level data, but treating every customer the same wastes marketing spend and misses opportunities to retain high-value shoppers. This project segments customers into meaningful behavioural groups — **without any predefined labels** — by engineering Recency/Tenure, Frequency, and Monetary (RFM-style) features from raw invoice data, then applying and comparing multiple unsupervised learning algorithms.

## 🎯 Objective

- Clean and preprocess a raw, messy e-commerce transaction log
- Engineer customer-level RFM-style features from line-item invoice data
- Detect and treat outliers before clustering
- Apply and compare K-Means, Hierarchical Clustering, and DBSCAN
- Evaluate cluster quality using Silhouette Score
- Translate statistical clusters into actionable business personas

## 🗂️ Dataset

- **Source:** Online Retail transaction dataset (UK-based gift/homeware retailer)
- **Size:** 541,909 rows × 8 columns (raw)
- **Granularity:** Invoice line-item level (one row per product per order)

| Column | Type | Description |
|---|---|---|
| InvoiceNo | object | Unique invoice number (prefix 'C' = cancellation) |
| StockCode | object | Unique product code |
| Description | object | Product name |
| Quantity | int64 | Units purchased (can be negative for returns) |
| InvoiceDate | object | Date/time of the invoice |
| UnitPrice | float64 | Price per unit |
| CustomerID | float64 | Unique customer identifier |
| Country | object | Customer's country |

## 🛠️ Tech Stack

- **Python 3**
- **pandas**, **NumPy** — data loading, cleaning, aggregation
- **Matplotlib**, **Seaborn** — visualization (boxplots, heatmap)
- **scikit-learn** — `KMeans`, `AgglomerativeClustering`, `DBSCAN`, `MinMaxScaler`, `silhouette_score`
- **yellowbrick** — `KElbowVisualizer` for optimal cluster-count selection

## 🔄 Project Workflow

### 1. Data Cleaning
- Removed 135,080 rows with missing `CustomerID` (~25% of data) — no reliable way to attribute anonymous transactions to a customer
- Removed 8,905 rows with negative `Quantity` (cancellations/returns)
- Final cleaned dataset: **397,924 rows**

### 2. Feature Engineering (RFM-style)
- **Monetary:** `Sales = Quantity × UnitPrice`, summed per customer
- **Frequency:** count of invoice line-items per customer
- **Recency/Tenure ("Last Transaction"):** days between the dataset's most recent invoice and each transaction, aggregated per customer

> **Note:** The tenure feature is computed using `.max()`, which actually captures how long ago a customer's *first* purchase occurred (tenure) rather than their most recent order (true recency) — an important nuance carried through the interpretation of the clusters.

### 3. Outlier Treatment
- Applied the IQR method (1.5×IQR rule) on the `Sales` column
- Removed 424 extreme customers (4,347 → 3,923), avoiding distortion of distance-based clustering

### 4. Feature Scaling
- Applied `MinMaxScaler` to normalize Last Transaction, InvoiceNo, and Sales into a common [0, 1] range

### 5. Clustering
- **Elbow Method** (Yellowbrick `KElbowVisualizer`) identified the optimal cluster count: **k = 3**
- **K-Means** (k=3)
- **Agglomerative Hierarchical Clustering** (k=3)
- **DBSCAN** (`eps=0.2`, `min_samples=4`)

### 6. Model Evaluation — Silhouette Score

| Algorithm | Clusters | Silhouette Score |
|---|---|---|
| K-Means | 3 | 0.774 |
| Agglomerative (Hierarchical) | 3 | **0.896** |
| DBSCAN | density-based (auto) | 0.880 |

All three algorithms scored well above 0.7, confirming genuine, well-separated cluster structure in the engineered features.

## 🧩 Customer Segments Identified

| Segment | Profile | Recommended Action |
|---|---|---|
| **High-Value Loyalists** | Long tenure, high spend | Loyalty programs, early access, personalized retention offers |
| **Long-Tenure, Low-Engagement** | Long tenure, low frequency/spend | Win-back campaigns, targeted discounts |
| **Newer / Low-Activity** | Short tenure, low frequency/spend | Onboarding nudges, first-repeat-purchase incentives |

## 📁 Repository Structure

```
├── main.ipynb                          # Full analysis notebook
├── Ecommerce_Clustering_Report.docx    # Detailed project report
└── README.md
```

## 🚀 Getting Started

```bash
pip install pandas numpy matplotlib seaborn scikit-learn yellowbrick
```

Open `main.ipynb` in Jupyter Notebook / JupyterLab and run all cells sequentially. Place `data.csv` in the same directory as the notebook.

## 📈 Key Insights

- InvoiceNo (frequency) and Sales (monetary value) show the strongest correlation (0.63) — frequent buyers tend to spend more overall
- A small number of extreme-value customers can dominate distance-based clustering if left untreated — outlier handling is essential
- Hierarchical Clustering slightly outperformed K-Means and DBSCAN on silhouette score, suggesting the natural customer segments aren't perfectly spherical/equal-sized

## 🔮 Future Work

- Compute a true recency metric (`.min()` instead of `.max()`) alongside tenure for a complete classical RFM model
- Validate cluster stability by re-running the pipeline on a later transaction window
- Feed cluster labels into a supervised churn-prediction or customer-lifetime-value model

## 📄 License

This project is open-source and available for educational and portfolio purposes.

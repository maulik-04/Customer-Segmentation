# Customer Segmentation — UCI Online Retail Dataset

I built this project to go beyond a typical clustering exercise and create something that actually answers a business question: **who are our customers, what do they buy, and what should we do about it?**

The dataset is a year of real transaction data from a UK-based online retailer. I used RFM analysis, K-Means++ clustering, product category discovery, and sub-clustering to segment 3,916 customers into actionable groups — and validated every output against published benchmarks for this dataset.

---

## Results

| Metric | Value |
|---|---|
| Dataset | 541,909 transactions · Dec 2010 – Dec 2011 |
| Customers analysed | 3,916 UK customers |
| Segments | 4 — Champions, Loyal Customers, At-Risk, Lost/Inactive |
| Silhouette Score | 0.3361 |
| Davies-Bouldin Index | 1.0089 |
| Calinski-Harabasz Index | 3,105 |
| Champions revenue share | 19.8% of customers → 54.8% of revenue |
| Pareto finding | Top 38.4% of customers drive 80% of revenue |

### Segment breakdown

| Segment | Customers | Median Recency | Median Frequency | Median Spend | Revenue | Share |
|---|---|---|---|---|---|---|
| Champions | 745 | 10 days | 8 orders | £2,711 | £2,534,614 | 54.8% |
| Loyal Customers | 1,040 | 61 days | 3 orders | £1,099 | £1,382,676 | 29.9% |
| At-Risk Customers | 692 | 18 days | 2 orders | £399 | £320,258 | 6.9% |
| Lost / Inactive | 1,285 | 178 days | 1 order | £270 | £389,289 | 8.4% |

---

## What I built and why

### Why RFM only for clustering — not TF-IDF or PCA

I tested mixing product description embeddings (TF-IDF) into the clustering space. It pushed the optimal k down to 2, produced a Silhouette of 0.17 and a DBI of 1.86, and gave segments that were impossible to explain to anyone outside data science. Clustering on 3 clean RFM features gives geometrically clean, interpretable clusters. Product features come in *after* clustering — for profiling and sub-segmentation — which is where they actually add value.

### Why k=4 when the silhouette peaks at k=2

The silhouette score peaked at k=2. But two clusters just gives you "good customers" and "bad customers" — you can't build a marketing campaign on that. k=4 maps to the Champions / Loyal / At-Risk / Lost framework used in industry, is backed by the original Chen et al. (2012) paper on this exact dataset, and the elbow method (KneeLocator) pointed to k=5, making k=4 a reasonable and well-justified choice.

### Why log-transform

All three RFM metrics are severely right-skewed:
- Recency skew: 1.14
- Frequency skew: 10.84  
- Monetary skew: 20.45

Without log-transform, one wholesale customer (£259,657 spend) would dominate K-Means distance calculations and collapse the clustering. `np.log1p` compresses the long tail without removing anyone from the analysis.

### Why RobustScaler

StandardScaler uses mean and standard deviation — both pulled heavily by outliers. RobustScaler uses median and IQR, making it resistant to the extreme values that exist in this dataset even after log-transform.

### Product category sub-clustering

I created 8 interpretable product categories (Christmas, Gift, Home Decor, Kitchen, Garden, Toys, Stationery, Other) using keyword matching on product descriptions. Each customer then gets a `pct_gift`, `pct_home_decor` etc. feature showing what percentage of their spend goes to each category. I then ran a second round of K-Means within Champions and Loyal Customers using these category features — producing sub-segments like "Champions — Gift Buyers" vs "Champions — Other Buyers". This is what makes personalised recommendations possible rather than sending every VIP customer the same generic email.

---

## Key findings

**The Pareto principle is stronger than expected here.** The classic 80/20 rule says 20% of customers drive 80% of revenue. In this dataset it's 38/62 — just 38.4% of customers drive 80% of revenue. The top spending decile alone accounts for 41.4% of all revenue. This means retaining Champions is even more critical than standard industry benchmarks suggest.

**Product preferences differ meaningfully across segments.** Champions spend 23% of their budget on Home Decor and 18.7% on Gifts. At-Risk Customers have elevated Christmas spend (9.3%), suggesting they're seasonal buyers who come in once for gifting and don't return. This directly informs campaign timing and product targeting.

**Sub-clustering adds real value.** Within Champions, Gift Buyers (237 customers, £2,915 median spend) have a completely different category mix than Other Buyers (508 customers, £2,617 median spend). Treating them the same wastes campaign budget and relevance.

**Customer growth was strong throughout 2011.** Active customers grew from 814 (Q4 2010) to 2,319 (Q4 2011) — a 185% increase. Revenue grew from £496K to £2.25M — a 353% increase. Both customer acquisition AND spend-per-customer increased, which means this wasn't just volume growth.

---

## Pipeline

```
541,909 raw transactions
    ↓
Data Cleaning
  • Drop missing CustomerID / Description  → 135,080 rows removed
  • Remove cancellations (InvoiceNo = C*)  →   8,905 rows removed
  • Remove Quantity ≤ 0 or UnitPrice ≤ 0  →      40 rows removed
  • Remove non-product StockCodes          →   1,414 rows removed
  • Filter to United Kingdom only          →  42,455 rows removed
  • Remove duplicates                      →   5,113 rows removed
  → 348,902 rows | 3,916 customers
    ↓
Feature Engineering
  • Recency   — days since last purchase
  • Frequency — number of distinct invoices
  • Monetary  — total spend (£)
  • Avg order value, avg basket size, unique products, purchase span
    ↓
RFM Scoring
  • Quartile scoring (1–4) per R, F, M dimension
  • RFM heatmap: customers with F=4 + R=4 spend median £3,098
    ↓
Preprocessing
  • Log-transform (np.log1p) — handles skewness of 1.14 / 10.84 / 20.45
  • RobustScaler — resistant to remaining outliers
  • No PCA — 3 features need no dimensionality reduction
    ↓
Optimal k Selection
  • Elbow method with KneeLocator → knee at k=5
  • Silhouette + DBI curves across k=2 to k=11
  • Selected k=4 for business interpretability
    ↓
K-Means++ Clustering
  • n_init=50, max_iter=500 for thorough initialisation
  • Silhouette=0.3361 | DBI=1.0089 | CHI=3,105
  • Clusters named by RFM score rank
    ↓
Product Category Discovery
  • 8 categories via keyword matching on product descriptions
  • Customer-level pct_gift, pct_home_decor, pct_kitchen etc.
  • Category preference chart per segment
    ↓
Sub-Clustering
  • Second K-Means within Champions + Loyal Customers
  • Uses category spend shares as features
  • Auto-named by dominant category
    ↓
Business Insights
  • Segment profiles with CLV proxy, revenue share, unique products
  • Action plans per segment with KPIs, channels, offers, risk level
```

---

## How to run

### Google Colab
Open the notebook in Colab. The data loads automatically from UCI:

```python
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx'
df_raw = pd.read_excel(url, engine='openpyxl')
```

If the UCI server is unavailable (it goes down occasionally), download the file manually from https://archive.ics.uci.edu/dataset/352/online+retail and upload it when prompted.

### Local
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy kneed openpyxl
jupyter notebook customer_segmentation_FINAL.ipynb
```

---

## Libraries used

| Library | Purpose |
|---|---|
| pandas, numpy | Data manipulation and feature engineering |
| scikit-learn | KMeans, RobustScaler, silhouette/DBI/CHI metrics |
| matplotlib, seaborn | All visualisations |
| scipy | Hierarchical clustering, dendrogram |
| kneed | Automated elbow detection (KneeLocator) |
| openpyxl | Reading the .xlsx file from UCI |

---

## Notebook structure

| Cell | What it does |
|---|---|
| 0 | Imports and colour palette setup |
| 1 | Load data from UCI (541,909 rows) |
| 2 | Google Drive mount (optional, legacy) |
| 3 | EDA — missing values, country distribution, monthly revenue trend |
| 4 | `clean_data()` — 6-step cleaning pipeline with row counts per step |
| 5 | RFM feature engineering — recency, frequency, monetary + behavioural features |
| 6 | `preprocess_rfm()` — log-transform + RobustScaler |
| 7 | Optimal k selection — elbow, silhouette, DBI across k=2 to k=11 |
| 8 | K-Means++ (k=4) — clustering, evaluation metrics, segment naming |
| 9 | Product category discovery — keyword matching, revenue by category |
| 10 | Customer-level category spend shares (pct_gift, pct_home_decor etc.) |
| 11 | Category preferences by segment — grouped bar chart |
| 12 | `sub_cluster_segment()` — second-layer clustering within Champions + Loyal |
| 13 | Sub-segment profiles and category mix visualisation |
| 14 | Final segment summary table with CLV, revenue share, action plans |

---

## References

1. Chen, D., Sain, S.L., & Guo, K. (2012). Data mining for the online retail industry: A case study of RFM model-based customer segmentation. *Journal of Database Marketing and Customer Strategy Management*, 19(3), 197–208. — original paper on this exact dataset, validates 4-segment RFM approach

2. John, J.M., Shobayo, O., & Ogunleye, B. (2023). An Exploration of Clustering Algorithms for Customer Segmentation in the UK Retail Market. *Analytics*, 2(4), 809–823. — benchmarks multiple algorithms on this dataset

3. Hughes, A.M. (1994). Strategic Database Marketing. Probus Publishing. — original RFM framework

4. UCI ML Repository: https://archive.ics.uci.edu/dataset/352/online+retail

---

## Dataset

UCI Online Retail Dataset · 541,909 transactions · UK-based non-store retailer · Dec 2010 – Dec 2011

*All monetary values are in GBP (£).*

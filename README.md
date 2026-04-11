# Customer-Segmentation

In this project, I worked on customer segmentation using a real-world transactional dataset from a UK-based online retail company, covering the period from December 2010 to December 2011. It has 8 features: Invoice Number, Product Code, Description, Quantity, Unit Price, Customer ID, Invoice Date and Country.

The dataset initially contained around 5.4 lakh transaction records. Since the data was at a transaction level, a single customer could appear multiple times — for example, if they purchased multiple products in a single invoice.

The first major challenge was to clean and preprocess this raw data. I removed records with missing Customer IDs because those transactions cannot be linked to any particular customer, filtered out invalid transactions with negative quantities or prices, and removed duplicates. After preprocessing, I was left with around 3.9 lakh valid records.

The next key step was transforming this transaction-level data into a customer-level feature matrix, where each row represents a unique customer. For this, I engineered behavioural features such as Recency, Frequency, and Monetary value (RFM), along with additional features like Average Basket Size and Number of Unique Products purchased.

However, RFM alone captures only how customers buy, not what they buy. To incorporate product preferences, I used TF-IDF vectorisation on product descriptions. This allowed me to generate 25 features representing customer affinity towards different product categories, such as bags, home décor, or gift items.

After combining these features, I applied Standard Scaling to normalise them and then used PCA to reduce dimensionality and remove redundancy, retaining around 90–95% of the variance.

Finally, I applied multiple clustering algorithms — KMeans, DBSCAN, and Hierarchical Clustering — and evaluated them using Silhouette Score and Davies–Bouldin Index. Based on these metrics, DBSCAN performed the best, as it was able to form well-separated clusters while also handling outliers effectively.


I identified five key customer segments: High-Value Loyal Customers, Potential Loyalists, Bargain Hunters, Inactive/Churned Customers, and Wholesale/Bulk Buyers.

Each segment represented distinct behavioural and product-preference patterns. For example, high-value loyal customers showed high frequency and spending along with strong affinity toward certain product categories, while bargain hunters were more price-sensitive with smaller basket sizes.

Based on these insights, I mapped each segment to specific business actions — such as loyalty programs and personalised recommendations for high-value customers, targeted promotions and bundle offers for bargain hunters, reactivation campaigns for inactive users, and bulk pricing strategies for wholesale buyers.

This ensured that the segmentation was not just a technical output, but directly actionable for improving customer retention, increasing revenue, and enabling personalised marketing strategies.

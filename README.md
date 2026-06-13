# E-Commerce Customer Analytics

Analysed a real online retail dataset to understand customer behaviour — who buys often, 
who's slipping away, and how much each customer is likely to spend.

## Dataset
UCI Online Retail Dataset — https://archive.ics.uci.edu/dataset/352/online+retail

Transactions from a UK-based online retailer (2010–2011). After cleaning — removing 
cancellations, missing CustomerIDs and invalid entries — ended up with 473 unique customers 
across 4,733 transactions.

## What I did

### 1. Data Cleaning
- Removed cancelled orders (InvoiceNo starting with 'C')
- Dropped missing CustomerIDs and negative quantities/prices
- Created a Revenue column (Quantity × UnitPrice)

### 2. RFM Feature Engineering
Compressed thousands of transactions into 3 numbers per customer:
- **Recency** — days since last purchase
- **Frequency** — number of unique orders
- **Monetary** — total amount spent

All 3 features were log-transformed (to handle skewness) and standardised with StandardScaler before modelling.

### 3. K-Means Clustering
Used two methods to find the optimal number of clusters:
- **Elbow Curve** — plotted inertia vs k; curve starts flattening after k=4
- **Silhouette Score** — score plateaus at k=4 and k=5 before dropping sharply at k≥6

Chose **k=4** as the best balance between compactness and separation. Customers were 
assigned to 4 segments based on average spend:

| Segment | Behaviour |
|---|---|
| Champions | High spend, frequent buyers, bought recently |
| Loyal | Regular buyers with decent spend |
| At Risk | Used to buy but haven't recently |
| Lost | Low spend, inactive for a long time |

Visualised clusters using **PCA** (2D and interactive 3D with Plotly) to confirm 
the segments are well separated in reduced feature space.

### 4. Classification — Predict Customer Segment
Trained 4 classifiers on RFM features to predict the segment of a new customer:

| Model | F1 Score |
|---|---|
| Logistic Regression | ~0.98 (best) |
| Random Forest | ~0.97 |
| Decision Tree | ~0.96 |
| KNN | ~0.94 |

Used **weighted F1 score** for comparison since segment sizes are imbalanced. 
Evaluated the best model using a classification report and confusion matrix.

### 5. Regression — Predict Customer Lifetime Value (CLV)
Trained 4 regressors to predict how much a customer will spend (log-transformed Monetary):

| Model | RMSE | R² |
|---|---|---|
| Gradient Boosting | ~0.26 (best) | ~0.67 |
| Random Forest | ~0.27 | ~0.65 |
| Decision Tree | ~0.31 | ~0.55 |
| Ridge Regression | ~0.38 | ~0.38 |

Evaluated using RMSE and R². Feature importance showed **Recency and Frequency** 
drive spend more than the segment label itself.

## Key Takeaways
- Recency is the strongest predictor — how recently a customer bought matters more than how much
- Champions are the most valuable group — worth prioritising for retention
- At Risk customers are the biggest opportunity — a targeted campaign could recover revenue
- Lost customers — one re-engagement attempt (discount/email), then deprioritise if no response

## Libraries
pandas, numpy, scikit-learn, matplotlib, seaborn, plotly

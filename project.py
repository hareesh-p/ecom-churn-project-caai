# -------------------------------------------
# 1. Notebook Setup & Imports
# -------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Plotly for interactive visualizations (optional)
import plotly.express as px

# For model evaluation later
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, roc_auc_score, f1_score, 
    confusion_matrix, roc_curve, auc, precision_recall_curve
)

# For warnings
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
sns.set(style='whitegrid', palette='muted')
plt.rcParams['figure.figsize'] = (10, 6)

# For reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("Libraries imported, plotting style and seed set.")

# -------------------------------------------
# 2. Section: Data Loading
# -------------------------------------------

# List all Olist dataset files used in the project
file_customer = "olist_customers_dataset.csv"
file_orders = "olist_orders_dataset.csv"
file_items = "olist_order_items_dataset.csv"
file_products = "olist_products_dataset.csv"
file_payments = "olist_order_payments_dataset.csv"
file_reviews = "olist_order_reviews_dataset.csv"
file_geolocation = "olist_geolocation_dataset.csv"
file_sellers = "olist_sellers_dataset.csv"

# Load all datasets as DataFrames
df_customers = pd.read_csv(file_customer)
df_orders = pd.read_csv(file_orders)
df_items = pd.read_csv(file_items)
df_products = pd.read_csv(file_products)
df_payments = pd.read_csv(file_payments)
df_reviews = pd.read_csv(file_reviews)
df_geolocation = pd.read_csv(file_geolocation)
df_sellers = pd.read_csv(file_sellers)

print("All datasets loaded successfully.")

# -------------------------------------------
# 3. Section: Initial Data Inspection
# -------------------------------------------

print("Customers:", df_customers.shape)
print("Orders:", df_orders.shape)
print("Order Items:", df_items.shape)
print("Products:", df_products.shape)
print("Payments:", df_payments.shape)
print("Reviews:", df_reviews.shape)
print("Geolocation:", df_geolocation.shape)
print("Sellers:", df_sellers.shape)

# Preview the first few rows of each dataset
display(df_customers.head())
display(df_orders.head())
display(df_items.head())
display(df_products.head())
display(df_payments.head())
display(df_reviews.head())

# -------------------------------------------
# 4. Section: Data Overview & EDA Planning
# -------------------------------------------

# Check for missing values in each dataset
for name, df in [
    ("Customers", df_customers),
    ("Orders", df_orders),
    ("Order Items", df_items),
    ("Products", df_products),
    ("Payments", df_payments),
    ("Reviews", df_reviews),
    ("Geolocation", df_geolocation),
    ("Sellers", df_sellers)
]:
    print(f"{name} missing values:\n{df.isnull().sum()}\n")

# List unique customers, orders, products for context
print(f"Unique customers: {df_customers['customer_unique_id'].nunique()}")
print(f"Unique orders: {df_orders['order_id'].nunique()}")
print(f"Unique products: {df_products['product_id'].nunique()}")

# Plan: Next, proceed with detailed EDA –
# - Customer & order distributions
# - Order timeline coverage
# - Geographic spread
# - Initial merges for RFM & churn labeling

# (You can continue adding EDA code in the next section.)

# -------------------------------------------
# Section: 1. Customer & Order Distribution
# -------------------------------------------

# Unique customers
n_unique_customers = df_customers['customer_unique_id'].nunique()
print(f"Unique customers: {n_unique_customers}")

# Merge orders with customers for analysis
orders_customers = pd.merge(df_orders, df_customers, on="customer_id", how="left")

# Orders per unique customer
orders_per_cust = orders_customers.groupby('customer_unique_id')['order_id'].nunique()

print("Orders per customer (describe):")
print(orders_per_cust.describe())

# Histogram: Orders per customer
plt.figure(figsize=(8,5))
sns.histplot(orders_per_cust, bins=20, kde=False)
plt.title("Number of Orders per Customer")
plt.xlabel("Orders per Customer")
plt.ylabel("Number of Customers")
plt.show()

# One-time vs. Repeat Buyers
one_time = (orders_per_cust == 1).sum()
repeat = (orders_per_cust > 1).sum()
plt.pie([one_time, repeat], labels=["One-time Buyers", "Repeat Buyers"], autopct='%.1f%%', startangle=90, colors=['#ff9999','#66b3ff'])
plt.title("One-time vs. Repeat Buyers")
plt.show()

# -------------------------------------------
# Section: 2. Order Timeline & Activity
# -------------------------------------------

# Convert order_purchase_timestamp to datetime
orders_customers['order_purchase_timestamp'] = pd.to_datetime(orders_customers['order_purchase_timestamp'])

# Monthly order volume
orders_customers['order_month'] = orders_customers['order_purchase_timestamp'].dt.to_period('M')
monthly_orders = orders_customers.groupby('order_month')['order_id'].nunique()

monthly_orders.plot(kind='line', marker='o')
plt.title("Monthly Number of Orders")
plt.xlabel("Month")
plt.ylabel("Number of Orders")
plt.xticks(rotation=45)
plt.show()

# New vs. Returning Customers Over Time
first_order = orders_customers.groupby('customer_unique_id')['order_purchase_timestamp'].min().reset_index()
first_order['first_order_month'] = first_order['order_purchase_timestamp'].dt.to_period('M')
customer_first_month = orders_customers.merge(first_order[['customer_unique_id', 'first_order_month']], on='customer_unique_id', how='left')
customer_first_month['is_new_customer'] = customer_first_month['order_month'] == customer_first_month['first_order_month']

monthly_new = customer_first_month[customer_first_month['is_new_customer']].groupby('order_month')['customer_unique_id'].nunique()
monthly_total = customer_first_month.groupby('order_month')['customer_unique_id'].nunique()
monthly_repeat = monthly_total - monthly_new

plt.stackplot(monthly_total.index.astype(str), monthly_new, monthly_repeat, labels=['New Customers', 'Returning Customers'], colors=['#b0c4de', '#4682b4'])
plt.title("Monthly New vs. Returning Customers")
plt.xlabel("Month")
plt.ylabel("Number of Customers")
plt.legend(loc='upper left')
plt.show()

# -------------------------------------------
# Section: 4. Delivery Performance
# -------------------------------------------

# Merge order items and orders to get delivery dates
orders_delivery = pd.merge(df_orders, df_items, on='order_id', how='left')
orders_delivery['order_delivered_customer_date'] = pd.to_datetime(orders_delivery['order_delivered_customer_date'])
orders_delivery['order_estimated_delivery_date'] = pd.to_datetime(orders_delivery['order_estimated_delivery_date'])

# Delivery delay in days
orders_delivery['delivery_delay'] = (orders_delivery['order_delivered_customer_date'] - orders_delivery['order_estimated_delivery_date']).dt.days

# Plot delivery delay distribution
sns.histplot(orders_delivery['delivery_delay'].dropna(), bins=30, color='orange')
plt.title("Delivery Delay Distribution (days)")
plt.xlabel("Days Late (Negative = Early)")
plt.ylabel("Number of Orders")
plt.show()

# Proportion of late deliveries
late_pct = (orders_delivery['delivery_delay'] > 0).mean() * 100
print(f"Proportion of late deliveries: {late_pct:.2f}%")

# -------------------------------------------
# Section: 5. Payment Methods
# -------------------------------------------

# Payment type counts
payment_types = df_payments['payment_type'].value_counts()
sns.barplot(x=payment_types.index, y=payment_types.values, palette='Set2')
plt.title("Payment Types Used")
plt.ylabel("Number of Payments")
plt.xlabel("Payment Type")
plt.show()

# Installments distribution
sns.histplot(df_payments['payment_installments'], bins=20)
plt.title("Distribution of Payment Installments")
plt.xlabel("Number of Installments")
plt.show()

# -------------------------------------------
# Section: 6. Customer Reviews
# -------------------------------------------

# Review score distribution
sns.countplot(x='review_score', data=df_reviews, palette='RdYlGn')
plt.title("Customer Review Score Distribution")
plt.xlabel("Review Score")
plt.ylabel("Number of Reviews")
plt.show()

# Negative (1-2), Neutral (3), Positive (4-5) review shares
review_cat = pd.cut(df_reviews['review_score'], bins=[0,2,3,5], labels=['Negative','Neutral','Positive'], include_lowest=True)
sns.countplot(x=review_cat, palette='coolwarm')
plt.title("Customer Review Categories")
plt.xlabel("Review Sentiment")
plt.ylabel("Number of Reviews")
plt.show()

# -------------------------------------------
# RFM Feature Engineering & Churn Labeling
# -------------------------------------------

from datetime import timedelta

# Step 1: RFM features using latest data
rfm = orders_customers.groupby('customer_unique_id').agg(
    frequency=('order_id', 'nunique'),
    last_purchase=('order_purchase_timestamp', 'max')
).reset_index()
rfm['recency'] = (orders_customers['order_purchase_timestamp'].max() - rfm['last_purchase']).dt.days
# Add monetary, etc.

# Step 2: Churn labeling using cutoff date
cutoff_date = orders_customers['order_purchase_timestamp'].max() - timedelta(days=180)
def churn_label_func(group, cutoff_date):
    # Check if customer purchased after cutoff
    purchased_after = group[group['order_purchase_timestamp'] > cutoff_date]
    if purchased_after.empty:
        return 1  # churned
    else:
        return 0  # retained

churn_labels = (
    orders_customers
    .groupby('customer_unique_id')
    .apply(churn_label_func, cutoff_date)
    .reset_index()
    .rename(columns={0: 'churn_label'})
)
rfm = rfm.merge(churn_labels, on='customer_unique_id', how='left')

print("Sample RFM with churn label:")
display(rfm.head())

# Churn label distribution
churn_counts = rfm['churn_label'].value_counts(dropna=False).sort_index()
print("Churn label distribution (1=churned, 0=retained, NaN=unknown):")
print(churn_counts)
sns.barplot(x=churn_counts.index.astype(str), y=churn_counts.values)
plt.title("Churn Label Distribution")
plt.xlabel("Churn Label")
plt.ylabel("Number of Customers")
plt.show()

# -------------------------------------------
# RFM Segmentation: Quantile Scores & Labels
# -------------------------------------------

# Assign quantile scores (1-5, higher=better) for Recency (inverse), Frequency, and Monetary.
rfm['r_score'] = pd.qcut(rfm['recency'], 5, labels=[5,4,3,2,1]).astype(int)
rfm['f_score'] = pd.qcut(rfm['frequency'].rank(method='first'), 5, labels=[1,2,3,4,5]).astype(int)
rfm['m_score'] = pd.qcut(rfm['monetary'].rank(method='first'), 5, labels=[1,2,3,4,5]).astype(int)

# Combine scores into RFM segment code
rfm['rfm_segment'] = rfm['r_score'].astype(str) + rfm['f_score'].astype(str) + rfm['m_score'].astype(str)
rfm['rfm_score'] = rfm[['r_score', 'f_score', 'm_score']].sum(axis=1)

# Assign simple descriptive segments (example rules, adjust as needed)
def rfm_segment_label(row):
    if row['r_score'] == 5 and row['f_score'] == 5:
        return 'Champion'
    elif row['r_score'] >= 4 and row['f_score'] >= 4:
        return 'Loyal'
    elif row['r_score'] == 5:
        return 'Recent'
    elif row['f_score'] == 5:
        return 'Frequent'
    elif row['m_score'] == 5:
        return 'Big Spender'
    elif row['r_score'] <= 2 and row['f_score'] <= 2:
        return 'At Risk'
    else:
        return 'Others'

rfm['segment'] = rfm.apply(rfm_segment_label, axis=1)

# Visualize segment counts
import seaborn as sns
import matplotlib.pyplot as plt

segment_counts = rfm['segment'].value_counts()
sns.barplot(x=segment_counts.index, y=segment_counts.values, palette='Set2')
plt.title("RFM Segment Distribution")
plt.xlabel("Segment")
plt.ylabel("Number of Customers")
plt.xticks(rotation=45)
plt.show()
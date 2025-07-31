import streamlit as st

st.set_page_config(page_title="🛍️ Shopper Spectrum", layout="centered")

import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Load your models
with open("kmeans_model.pkl", "rb") as f:
    kmeans = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("similarity_matrix.pkl", "rb") as f:
    similarity_df = pickle.load(f)

# App layout and logic follows here...

# Load data and models
@st.cache_data
def load_data():
    df = pd.read_csv("online_retail.csv", encoding='ISO-8859-1')
    df.dropna(subset=['CustomerID'], inplace=True)
    df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    df['CustomerID'] = df['CustomerID'].astype(int)
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    return df

df = load_data()

# Build similarity matrix
@st.cache_data
def build_similarity_matrix(df):
    pivot = df.pivot_table(index='CustomerID', columns='StockCode', values='Quantity', fill_value=0)
    similarity = cosine_similarity(pivot.T)
    similarity_df = pd.DataFrame(similarity, index=pivot.columns, columns=pivot.columns)
    return similarity_df, df

similarity_df, df = build_similarity_matrix(df)

# RFM Cluster Prediction Model (simulate with rules or load model)
def rfm_predict(recency, frequency, monetary):
    if recency <= 30 and frequency >= 10 and monetary >= 500:
        return "High-Value"
    elif frequency >= 5 and monetary >= 100:
        return "Regular"
    elif recency > 90 and frequency < 2:
        return "At-Risk"
    else:
        return "Occasional"

# ----------- Streamlit UI Starts Here -----------

st.title("🛍️ Shopper Spectrum")
st.subheader("Customer Segmentation & Product Recommendation App")

# Tabs
tab1, tab2 = st.tabs(["📦 Product Recommender", "🧑‍💼 Customer Segment Predictor"])

# 1. Product Recommender
with tab1:
    st.markdown("### 🔍 Find Similar Products")
    input_product = st.text_input("Enter Product StockCode (e.g., 85123A):")

    if input_product:
        if input_product in similarity_df.index:
            st.success(f"Top 5 recommendations for Product Code: {input_product}")
            top_products = similarity_df[input_product].sort_values(ascending=False)[1:6]
            for i, (prod, score) in enumerate(top_products.items(), 1):
                st.write(f"{i}. **StockCode**: {prod} — Similarity Score: {score:.2f}")
        else:
            st.error("Product code not found in database.")

# 2. RFM Segment Predictor
with tab2:
    st.markdown("### 🧠 Predict Customer Segment")

    recency = st.number_input("Recency (days since last purchase):", min_value=0, step=1)
    frequency = st.number_input("Frequency (number of purchases):", min_value=0, step=1)
    monetary = st.number_input("Monetary (total spent):", min_value=0.0, step=1.0, format="%.2f")

    if st.button("Predict Segment"):
        result = rfm_predict(recency, frequency, monetary)
        st.success(f"📌 Predicted Segment: **{result}**")

# Footer
st.markdown("---")
st.caption("Built with ❤️ using Streamlit | Shopper Spectrum Project")

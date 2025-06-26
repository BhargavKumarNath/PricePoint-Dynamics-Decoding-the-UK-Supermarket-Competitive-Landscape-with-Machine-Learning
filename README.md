# PricePoint Dynamics: A Deep Dive into UK Supermarket Competition with ML

## Overview

This project is an end-to-end data science analysis of the competitive landscape of the UK supermarket industry. Using daily price data scraped from **Aldi, ASDA, Morrisons, Sainsbury's, and Tesco**, this repository explores the full data science lifecycle: from data engineering and advanced NLP for product matching to predictive modeling and strategic insight generation.

The primary goal is to move beyond simple price comparisons to understand the deep dynamics of the market: Who leads price changes? Who follows? How do pricing strategies differ across categories? And can we predict future price movements?

## Key Features & Highlights

*   **Advanced Product Matching:** Implemented a state-of-the-art canonical product mapping pipeline using **Sentence-BERT embeddings** and **FAISS** for efficient similarity search, enabling true like-for-like product comparisons.
*   **Comprehensive EDA:** Uncovered market positioning, portfolio strategies, and category focus for each of the five major UK supermarkets.
*   **Predictive Modeling:** Built a LightGBM model that predicts daily product prices with a **Mean Absolute Error of just £0.14**.
*   **Explainable AI (XAI):** Used **SHAP** to interpret the "black box" model, revealing the key drivers behind price predictions and quantifying the real-world impact of features like brand and retailer choice.
*   **Dynamic Analysis:** Modeled market dynamics through **time-series decomposition, cross-correlation analysis** to identify price leaders, and **price dispersion analysis** to measure market competitiveness over time.
*   **Unsupervised Learning:** Deployed an **Isolation Forest** model to automatically detect anomalous pricing events, useful for identifying promotions or data errors.

---

## Key Findings & Insights

The analysis revealed a stable, two-tiered market with highly predictable dynamics.

1.  **Aldi is the Undisputed Price Leader:** Aldi consistently sets the price benchmark, with the "Big Four" following its lead with a median lag of 4-7 days.
2.  **The "Big Four" are Fast Followers:** Among the larger chains, the competitive environment is highly reactive, with price changes being matched within 0-2 days. ASDA and Morrisons emerge as the fastest price-setters in this group.
3.  **Market in Stable Equilibrium:** Despite dynamic daily price changes, the overall price dispersion across the market has remained stable since February 2024, indicating retailers are maintaining their strategic price gaps rather than engaging in a price war.
4.  **Pricing is Highly Predictable:** The final regression model demonstrates that future prices are overwhelmingly a function of their recent history and competitive context, allowing for highly accurate forecasting.

---

## Tech Stack & Libraries

*   **Data Manipulation & Analysis:** Pandas, NumPy
*   **Machine Learning:** Scikit-learn, LightGBM
*   **Natural Language Processing:** Sentence-Transformers (`intfloat/e5-large`)
*   **Vector Search:** FAISS (Facebook AI Similarity Search)
*   **Statistical Analysis:** Statsmodels
*   **Model Interpretability:** SHAP
*   **Data Visualization:** Matplotlib, Seaborn

---

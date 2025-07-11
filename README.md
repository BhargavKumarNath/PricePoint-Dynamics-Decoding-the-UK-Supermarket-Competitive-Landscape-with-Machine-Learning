# PricePoint Dynamics: A Deep Dive into UK Supermarket Competition with Machine Learning

## Overview

This project is an end-to-end data science analysis of the competitive landscape of the UK supermarket industry. Using daily price data scraped from **Aldi, ASDA, Morrisons, Sainsbury's, and Tesco**, this repository explores the full data science lifecycle: from data engineering and advanced NLP for product matching to predictive modeling and strategic insight generation.

The primary goal is to move beyond simple price comparisons to understand the deep dynamics of the market: Who leads price changes? Who follows? How do pricing strategies differ across categories? And can we predict future price movements with high accuracy?

## Key Features & Highlights

*   **Advanced Product Matching:** Implemented a state-of-the-art canonical product mapping pipeline using **Sentence-BERT embeddings** and **FAISS** for efficient similarity search, enabling true like-for-like product comparisons.
*   **Comprehensive EDA:** Uncovered market positioning, portfolio strategies, and category focus for each of the five major UK supermarkets.
*   **Predictive Modeling:** Built a LightGBM model that predicts daily product prices with a **Mean Absolute Error of just £0.14**.
*   **Explainable AI (XAI):** Used **SHAP** to interpret the "black box" model, revealing the key drivers behind price predictions and quantifying the real-world impact of features like brand and retailer choice.
*   **Dynamic Market Analysis:** Modeled market dynamics through **time-series decomposition, cross-correlation analysis** to identify price leaders, and **price dispersion analysis** to measure market competitiveness over time.
*   **Unsupervised Learning:** Deployed an **Isolation Forest** model to automatically detect anomalous pricing events, useful for identifying promotions or data errors.
* **Interactive Dashboard:** A user friendly Streamlit application to visualise the findings and interact with the predictive models, making complex data accessible to all stakeholders.

---

## Key Findings & Insights

The analysis revealed a stable, two-tiered market with highly predictable dynamics.

1.  **Distinct Market Positioning:** The analysis reveals clear, data-driven evidence of different strategic positions among the UK's top supermarkets:
    * **Aldi** stands out as the go-to budget option, consistently offering the lowest prices on a carefully selected range of everyday essentials.
    * **Tesco & Sainsbury** are in close competition for the mainstream shopper. Tesco tends to have slightly better deals on branded products, while Sainsbury’s offers a broader selection—especially in niche and specialty categories.
    * **Morrisons** has positioned itself as a value-focused alternative within the "Big Four." Surprisingly, it even beats Aldi in certain categories like ‘Healthy Choice’ and ‘Free From’ baskets.

2. **Price Leadership Dynamics:** The data unveils a clear "leader-follower" dynamic in the market:
    * **Tesco** frequently acts as the price leader. Cross-correlation analysis shows that its price changes are often the first to occur.
    * Other supermarkets, notably **Sainsbury** tend to follow Tesco's lead, adjusting their prices around two weeks later.

3. **Highly Accurate Price Prediction:** The developed LightGBM model can predict product prices with a Mean Absolute Error (MAE) of just £0.14, making it a reliable tool for forecasting future pricing trends.

4. **Key Drivers of Price Changes:** The model's feature importance, revealed through SHAP analysis, shows that:
    * A product's own price history (such as its 7-day minimum price) is the single most powerful predictor of its future price.
    * The supermarket brand itself is a major systematic driver of price level, confirming the distinct pricing strategies of each retailer.
    * Market-relative metrics, like a product's price compared to the market average, are also highly influential.

5. **Stable Competitive Equilibrium:** Despite daily price fluctuations, the overall price dispersion across the market has remained remarkably stable since February 2024. This indicates that retailers are maintaining their strategic price gaps rather than engaging in a destructive price war.

6.**Methodological Innovation for Fair Comparison:** A significant challenge in this analysis was ensuring like-for-like product comparisons. This was overcome by implementing an advanced product matching pipeline using Sentence-BERT embeddings and FAISS, which allowed for the creation of standardized shopping baskets for accurate cross-retailer analysis.

---

## Tech Stack & Libraries

*   **Data Manipulation & Analysis:** Pandas, NumPy
*   **Machine Learning:** Scikit-learn, LightGBM
*   **Natural Language Processing:** Sentence-Transformers (`intfloat/e5-large`)
*   **Vector Search:** FAISS (Facebook AI Similarity Search)
*   **Statistical Analysis:** Statsmodels
*   **Model Interpretability:** SHAP
*   **Data Visualization:** Matplotlib, Seaborn, Plotly
*   **Dashboard:** Streamlit
---

## Interactive Dashboard
The project culminates in a powerful Streamlit dashboard designed for users to explore the complex world of supermarket pricing.
* **Market Overview**: Presents a consolidated view of product coverage and average pricing across major UK supermarkets, providing a high-level understanding of market composition.

* **Basket Analysis:** Facilitates direct, like-for-like price comparisons using standardized shopping baskets (e.g., “Essentials,” “Healthy Choice”), allowing for fair cross-retailer evaluation.

* **Price Predictor:** Offers an interactive forecasting tool powered by a trained machine learning model, enabling users to estimate future prices for any product across different retailers.

* **Model Insights:** Delivers interpretability through SHAP-based visualizations, highlighting the most influential factors behind model predictions and ensuring transparency in decision logic.

* **Market Dynamics:** Visualizes broader trends such as price leadership shifts and market-wide price dispersion over time, helping users understand underlying competitive behaviors.
---


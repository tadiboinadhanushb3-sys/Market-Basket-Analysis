# RetailIntelligence: Market Basket Pro 🛒

**Industry-Level Market Basket Analysis using FP-Growth**

This project is a production-style implementation of association rule mining, designed for large-scale retail datasets. Unlike the traditional Apriori algorithm, this system utilizes **FP-Growth (Frequent Pattern Growth)** for superior scalability and memory efficiency.

## 🚀 Key Features
- **Scalable Engine**: Uses FP-Growth to handle thousands of transactions without the exponential complexity of Apriori.
- **Smart Data Generation**: Simulates 10,000+ transactions with built-in product affinities (e.g., Bread -> Butter).
- **Executive Dashboard**: Real-time Streamlit UI with metric-driven KPIs.
- **Advanced Visualizations**: Support heatmaps, scatter plots, and support distribution charts.
- **Recommendation Engine**: Suggests top-selling product pairs based on the Lift metric.
- **Business Insight Generator**: Automatically converts mathematical rules into readable strategy recommendations.

## 📊 Technical Metrics
- **Support**: How frequently the itemset appears in the dataset.
- **Confidence**: How often the consequent follows the antecedent.
- **Lift**: The strength of an association (Lift > 1 implies a positive relationship).
- **Leverage**: Difference between the observed frequency of A and B appearing together and the frequency that would be expected if A and B were independent.

## 🛠️ Installation & Setup

1. **Clone/Navigate to the directory**:
   ```bash
   cd market_basket_pro
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Dashboard**:
   ```bash
   streamlit run app.py
   ```

## 📂 Project Structure
```
market_basket_pro/
├── app.py                  # Main Streamlit Application
├── src/
│   ├── data_generator.py   # Synthetic Transaction Engine
│   └── mba_engine.py       # Core MBA / ML Logic
├── data/
│   └── transactions.csv    # Generated data (Auto-created)
├── requirements.txt        # Project dependencies
└── README.md               # Documentation
```

## 🧠 Scalability Approach: Why FP-Growth?
Traditional algorithms like **Apriori** use a "Join and Prune" strategy, which requires multiple passes over the dataset and generates a massive number of candidate itemsets. This becomes a bottleneck for industry-scale data.

**FP-Growth** addresses this by:
1. **Compressing the database**: It builds a Trie-based structure called an **FP-Tree**.
2. **Dividing and Conquering**: It eliminates the need for candidate generation by extracting frequent itemsets directly from the FP-Tree.
3. **Memory Efficiency**: The tree structure is highly compact, making it significantly faster for datasets with 10k+ rows.

---
*Created for AI/Data Science Professional Portfolios.*

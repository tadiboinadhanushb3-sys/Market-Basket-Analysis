import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from src.data_generator import generate_synthetic_data
from src.mba_engine import MBAEngine
import os

# --- Page Config ---
st.set_page_config(page_title="RetailIntelligence | Market Basket Pro", layout="wide")

# --- Custom Styling ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { border-left: 5px solid #007bff; padding-left: 10px; }
    .reportview-container .main .block-container{ padding-top: 1rem; }
    h1 { color: #1e3a8a; font-weight: 800; }
    h3 { color: #1e40af; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- Sidebar Controls ---
st.sidebar.title("📊 Control Panel")
min_support = st.sidebar.slider("Minimum Support", 0.001, 0.1, 0.01, help="Frequency of itemsets in transactions")
min_confidence = st.sidebar.slider("Minimum Confidence", 0.1, 1.0, 0.2, help="Probability of consequent item being bought with antecedent")

# --- Data Loading ---
@st.cache_data
def load_data():
    if not os.path.exists('data/transactions.csv'):
        generate_synthetic_data()
    return pd.read_csv('data/transactions.csv')

df = load_data()
categories = ['All'] + sorted(df['Category'].unique().tolist())
selected_cat = st.sidebar.selectbox("Filter by Category", categories)

# --- Initialization ---
engine = MBAEngine(df)

# --- Main Dashboard ---
st.title("🛒 Market Basket Analysis Pro")
st.markdown("### Industry-Level Association Rule Mining & Recommendation Engine")
st.write("Leveraging **FP-Growth** for highly scalable market insights.")

# Process Data
with st.spinner("Analyzing transaction patterns..."):
    basket = engine.prep_basket(category=selected_cat)
    frequent_itemsets = engine.run_fpgrowth(min_support=min_support)
    rules = engine.generate_rules(min_threshold=min_confidence)

# --- KPI Section ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Transactions", f"{df['Transaction_ID'].nunique():,}")
col2.metric("Unique Products", f"{df['Product_ID'].nunique()}")
col3.metric("Itemsets Found", f"{len(frequent_itemsets)}")
col4.metric("Rules Generated", f"{len(rules)}")

# --- Visualizations ---
tab1, tab2, tab3 = st.tabs(["📌 Association Rules", "📊 Visual Analytics", "🤖 Recommendation Engine"])

with tab1:
    if not rules.empty:
        st.subheader("Top Association Rules (Sorted by Lift)")
        
        # Style the dataframe
        styled_rules = rules[['antecedents_str', 'consequents_str', 'support', 'confidence', 'lift', 'leverage']].head(20)
        st.dataframe(styled_rules.style.background_gradient(subset=['lift'], cmap='Blues'), use_container_width=True)
        
        # Business Insights Generator
        st.markdown("### 🔍 Automated Business Insights")
        selected_rule_idx = st.selectbox("Select a rule to expand:", rules.index[:10])
        insight = engine.generate_insights(rules.loc[selected_rule_idx])
        st.info(insight)
        
        # Download
        csv = rules.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Full Rules CSV", data=csv, file_name='market_basket_rules.csv', mime='text/csv')
    else:
        st.warning("No associations found with the current thresholds. Try lowering Support or Confidence.")

with tab2:
    if not rules.empty:
        col_v1, col_v2 = st.columns(2)
        
        with col_v1:
            st.markdown("#### Top 20 Frequent Itemsets")
            top_items = frequent_itemsets.sort_values('support', ascending=False).head(20)
            top_items['itemsets_str'] = top_items['itemsets'].apply(lambda x: ', '.join(list(x)))
            fig_bar = px.bar(top_items, x='support', y='itemsets_str', orientation='h', color='support', 
                             title="Support Distribution", color_continuous_scale='Blues')
            st.plotly_chart(fig_bar, use_container_width=True)
            
        with col_v2:
            st.markdown("#### Support vs Confidence Scatter")
            fig_scatter = px.scatter(rules, x="support", y="confidence", color="lift", size="lift",
                                     hover_data=['antecedents_str', 'consequents_str'], title="Rule Strength Distribution")
            st.plotly_chart(fig_scatter, use_container_width=True)

        st.markdown("#### 🌡️ Rule Heatmap (Top 15)")
        top_15_rules = rules.head(15)
        pivot = top_15_rules.pivot(index='antecedents_str', columns='consequents_str', values='lift')
        fig_heat = px.imshow(pivot, labels=dict(x="Consequent", y="Antecedent", color="Lift"), 
                             title="Association Strength Heatmap", color_continuous_scale='RdBu_r')
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("Insufficient data for visualization.")

with tab3:
    st.markdown("### 🛍️ Smart Product Recommender")
    all_products = sorted(df['Product_Name'].unique())
    target_product = st.selectbox("Select a product to see recommendations for:", all_products)
    
    recs = engine.get_recommendations(target_product)
    
    if isinstance(recs, pd.DataFrame) and not recs.empty:
        st.success(f"Top products associated with **{target_product}**:")
        for _, row in recs.iterrows():
            st.markdown(f"- **{row['consequents_str']}** (Lift: {row['lift']:.2f}, Confidence: {row['confidence']:.1%})")
            
        # Recommendation Chart
        fig_rec = px.bar(recs, x='lift', y='consequents_str', orientation='h', title=f"Recommendations for {target_product}")
        st.plotly_chart(fig_rec, use_container_width=True)
    else:
        st.info("No strong associations found for this product with current settings.")

# --- Footer ---
st.markdown("---")
st.caption("Built for AI/Data Science Final Year Project | Scalable FP-Growth Implementation")

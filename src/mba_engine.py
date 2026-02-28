import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules
import numpy as np

class MBAEngine:
    """
    Advanced MBA Engine with Adaptive Thresholding and Graph Analytics.
    """
    def __init__(self, df):
        self.df = df
        self.basket = None
        self.frequent_itemsets = None
        self.rules = None

    # --- 1. Adaptive Basket Prep ---
    def prep_basket(self, category='All'):
        """
        Transforms transaction data into a binary basket format with Category Filtering.
        """
        filtered_df = self.df if category == 'All' else self.df[self.df['Category'] == category]
        
        # Binary encoding
        basket = (filtered_df.groupby(['Transaction_ID', 'Product_Name'])['Quantity']
                  .sum().unstack().reset_index().fillna(0)
                  .set_index('Transaction_ID'))
        
        # Convert to boolean for MLxtend
        self.basket = basket.applymap(lambda x: True if x > 0 else False)
        return self.basket

    # --- 2. Scaling FP-Growth ---
    def run_fpgrowth(self, min_support=0.01):
        """
        Finds frequent itemsets using FP-Growth (Scalable).
        """
        if self.basket is None or self.basket.empty:
            return pd.DataFrame()
            
        self.frequent_itemsets = fpgrowth(self.basket, min_support=min_support, use_colnames=True)
        return self.frequent_itemsets

    # --- 3. Adaptive Rule Tuning ---
    def generate_rules(self, metric="lift", min_threshold=1.0):
        """
        Generates rules with dynamic thresholding logic.
        """
        if self.frequent_itemsets is None or self.frequent_itemsets.empty:
            # Re-run with lower support if zero results (Adaptive Tuning)
            self.frequent_itemsets = self.run_fpgrowth(min_support=0.005)
            
        if self.frequent_itemsets.empty:
            return pd.DataFrame()
            
        rules = association_rules(self.frequent_itemsets, metric=metric, min_threshold=min_threshold)
        
        if not rules.empty:
            # Add descriptive strings
            rules['antecedents_str'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
            rules['consequents_str'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
            
            # Additional Metric: Conviction (How much consequent depends on antecedent)
            rules['conviction'] = (1 - rules['consequent support']) / (1 - rules['confidence'])
            # Sort by strength index (Combine Lift + Confidence weight)
            rules['weight'] = (rules['lift'] * rules['confidence']).round(3)
            
            self.rules = rules.sort_values('weight', ascending=False)
        else:
            self.rules = pd.DataFrame()
            
        return self.rules

    # --- 4. Network Graph Preparation ---
    def get_network_data(self, top_n=30):
        """
        Prepares Source-Target-Weight mapping for professional network viz.
        """
        if self.rules is None or self.rules.empty:
            return pd.DataFrame()
            
        # Select top rules
        top_rules = self.rules.head(top_n)
        
        nodes = []
        # Extract individual nodes from rules
        for idx, row in top_rules.iterrows():
            nodes.append({'source': row['antecedents_str'], 'target': row['consequents_str'], 'value': row['lift']})
            
        return pd.DataFrame(nodes)

    @staticmethod
    def get_business_strategy(rule):
        """
        Generate actionable retail strategy based on rule metrics.
        """
        ant = rule['antecedents_str']
        con = rule['consequents_str']
        lift = round(rule['lift'], 2)
        
        if lift > 5:
            return f"🔥 **Hard Bundle Strategy:** Merge **{ant}** and **{con}** into a single SKU package. Extremely high affinity!"
        elif rule['confidence'] > 0.8:
            return f"🛒 **Inventory Anchor:** **{ant}** acts as a driver for **{con}**. Ensure **{con}** is always in stock near **{ant}**."
        else:
            return f"📦 **Cross-Promotion:** Place **{con}** at the checkout or end-aisle for customers who picked **{ant}**."

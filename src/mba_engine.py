import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules
import streamlit as st

class MBAEngine:
    def __init__(self, df):
        self.df = df
        self.basket = None
        self.frequent_itemsets = None
        self.rules = None

    def prep_basket(self, category=None):
        """
        Transforms transaction data into a one-hot encoded basket format.
        """
        df_filtered = self.df
        if category and category != 'All':
            df_filtered = self.df[self.df['Category'] == category]
            
        # Group by Transaction _ID and Product_Name
        basket = (df_filtered.groupby(['Transaction_ID', 'Product_Name'])['Product_ID']
                  .count().unstack().reset_index().fillna(0)
                  .set_index('Transaction_ID'))
        
        # Convert counts to binary (1 or 0)
        def encode_units(x):
            if x <= 0: return False
            if x >= 1: return True
        
        self.basket = basket.applymap(encode_units)
        return self.basket

    def run_fpgrowth(self, min_support=0.01):
        """
        Runs the FP-Growth algorithm for better scalability.
        """
        self.frequent_itemsets = fpgrowth(self.basket, min_support=min_support, use_colnames=True)
        return self.frequent_itemsets

    def generate_rules(self, metric="lift", min_threshold=1.0):
        """
        Generates association rules from frequent itemsets.
        """
        if self.frequent_itemsets is None or self.frequent_itemsets.empty:
            self.rules = pd.DataFrame(columns=['antecedents', 'consequents', 'support', 'confidence', 'lift', 'leverage'])
            return self.rules
            
        rules = association_rules(self.frequent_itemsets, metric=metric, min_threshold=min_threshold)
        
        # Post-processing: Filter only lift > 1 as per user requirement
        if not rules.empty:
            rules = rules[rules['lift'] > 1.0].sort_values('lift', ascending=False)
            
            # Convert frozensets to strings for UI display
            rules['antecedents_str'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
            rules['consequents_str'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
        else:
            # Ensure columns exist even if empty
            rules = pd.DataFrame(columns=['antecedents', 'consequents', 'support', 'confidence', 'lift', 'leverage', 'antecedents_str', 'consequents_str'])
            
        self.rules = rules
        return rules

    def get_recommendations(self, product_name, top_n=5):
        """
        Get top associated products for a specific item.
        """
        if self.rules is None or self.rules.empty:
            return pd.DataFrame()
        
        # Filter rules where the product is in the antecedent
        recs = self.rules[self.rules['antecedents'].apply(lambda x: product_name in x)]
        
        if recs.empty:
            return pd.DataFrame()
            
        return recs.sort_values('lift', ascending=False).head(top_n)

    @staticmethod
    def generate_insights(rule_row):
        """
        Generates a human-readable business insight.
        """
        ant = rule_row['antecedents_str']
        con = rule_row['consequents_str']
        lift = round(rule_row['lift'], 2)
        conf = round(rule_row['confidence'] * 100, 1)
        
        return f"💡 **Strategic Insight:** Customers buying **{ant}** are **{lift}x** more likely to also purchase **{con}**. This pair has a **{conf}%** confidence level, suggesting a strong cross-selling opportunity."

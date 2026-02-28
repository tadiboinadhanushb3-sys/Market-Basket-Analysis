import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import os

class AnalyticsEngine:
    """
    Predictive Decision Support & Customer Intelligence Engine.
    Handles Customer Segmentation, Sales Forecasting, and Profit Simulation.
    """
    def __init__(self, df):
        self.df = df
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.segmentation_data = None
        self.forecasts = None

    # --- 1. Customer Segmentation (KMeans) ---
    def segment_customers(self, n_clusters=4):
        """
        Segment customers based on RFM (Recency, Frequency, Monetary) and Basket Size.
        """
        # Calculate RFM metrics
        last_date = self.df['Date'].max()
        
        rfm = self.df.groupby('Customer_ID').agg({
            'Date': lambda x: (last_date - x.max()).days, # Recency
            'Transaction_ID': 'nunique',                   # Frequency
            'Total_Price': 'sum',                          # Monetary (Monetary Value)
            'Quantity': 'mean'                             # Avg Basket Size
        })
        
        rfm.columns = ['Recency', 'Frequency', 'Monetary', 'Avg_Basket_Size']
        
        # Scale features
        scaler = StandardScaler()
        rfm_scaled = scaler.fit_transform(rfm)
        
        # Apply KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        rfm['Segment'] = kmeans.fit_predict(rfm_scaled)
        
        # Human-Readable Segment Map
        segment_map = {
            0: 'Top Spenders',
            1: 'At Risk',
            2: 'Loyal Customers',
            3: 'New/Occasional'
        }
        rfm['Segment_Label'] = rfm['Segment'].map(segment_map)
        
        self.segmentation_data = rfm
        return rfm

    # --- 2. Sales Forecasting (Linear Regression per Category) ---
    def forecast_sales(self, target_category='All'):
        """
        Predict next 3 months of sales using Linear Regression trend analysis.
        """
        if target_category == 'All':
            agg_df = self.df.resample('ME', on='Date')['Total_Price'].sum().reset_index()
        else:
            agg_df = self.df[self.df['Category'] == target_category].resample('ME', on='Date')['Total_Price'].sum().reset_index()

        # Prepare X (Time Index) and y (Sales)
        agg_df['Month_Index'] = np.arange(len(agg_df))
        X = agg_df[['Month_Index']]
        y = agg_df['Total_Price']
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Predict next 3 months
        future_indices = np.array([[len(agg_df)], [len(agg_df)+1], [len(agg_df)+2]])
        predictions = model.predict(future_indices)
        
        forecast_df = pd.DataFrame({
            'Month': [agg_df['Date'].max() + pd.DateOffset(months=i+1) for i in range(3)],
            'Predicted_Sales': predictions
        })
        
        return agg_df, forecast_df

    # --- 3. Profit Optimization Simulator ---
    def simulate_profit_impact(self, discount=0.1, category='All'):
        """
        Simulate impact of discounts on revenue and potential rule lift.
        Estimation heuristic: 10% discount -> 15% increase in volume, but lower margin.
        """
        filtered_df = self.df if category == 'All' else self.df[self.df['Category'] == category]
        
        # Base Scenario
        base_revenue = filtered_df['Total_Price'].sum()
        base_volume = filtered_df['Quantity'].sum()
        
        # Scenario: Discounted
        # Elasticity Assumption: % change in Q / % change in P = -1.5 (Standard Retail)
        # 10% discount -> 15% volume increase
        volume_multi = 1 + (abs(discount) * 1.5)
        new_volume = base_volume * volume_multi
        
        avg_price = (filtered_df['Total_Price'] / filtered_df['Quantity']).mean()
        new_price = avg_price * (1 - discount)
        
        sim_revenue = new_volume * new_price
        margin_impact = sim_revenue - base_revenue
        
        return {
            'Base_Rev': round(base_revenue, 2),
            'Sim_Rev': round(sim_revenue, 2),
            'Impact': round(margin_impact, 2),
            'Volume_Gain': round((volume_multi - 1) * 100, 1)
        }

    # --- 4. Hybrid Recommendation (Segment-Aware) ---
    def get_hybrid_recommendations(self, customer_id, top_n=5):
        """
        Combine Association Rules (Item Affinity) + Customer Segment behavior.
        """
        # 1. Get Customer's last purchased category
        cust_history = self.df[self.df['Customer_ID'] == customer_id].sort_values('Date', ascending=False)
        if cust_history.empty:
            return "New Customer - No History"
        
        fav_cat = cust_history['Category'].mode()[0]
        recent_items = cust_history['Product_ID'].head(3).tolist()
        
        # Logic: Recommend top products in their segment's high-affinity categories
        segment = self.segmentation_data.loc[customer_id, 'Segment_Label']
        
        return f"Based on segment: **{segment}** and interest in **{fav_cat}**..."

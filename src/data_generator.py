import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

def generate_synthetic_data(num_transactions=15000, num_customers=1000, num_products=200):
    """
    Generates a high-fidelity synthetic retail dataset for advanced AI analytics.
    Includes: Customer IDs, Quantities, Unit Prices, and Transaction Dates.
    """
    print(f"🚀 Generating AI-Ready Dataset: {num_transactions} transactions, {num_customers} customers...")

    # 1. Product Catalog
    categories = {
        'Bakery': (2.0, 10.0),
        'Dairy': (3.0, 15.0),
        'Produce': (1.0, 8.0),
        'Meat': (10.0, 60.0),
        'Beverages': (1.5, 25.0),
        'Canned Goods': (2.0, 12.0),
        'Frozen': (5.0, 30.0),
        'Snacks': (1.0, 15.0),
        'Electronics': (50.0, 500.0),
        'Home & Kitchen': (10.0, 150.0)
    }
    
    products = []
    pid_to_cat = {}
    for i in range(num_products):
        cat = random.choice(list(categories.keys()))
        min_p, max_p = categories[cat]
        pid = f'P{str(i+1).zfill(3)}'
        products.append({
            'Product_ID': pid,
            'Product_Name': f'{cat} Item {i+1}',
            'Category': cat,
            'Unit_Price': round(random.uniform(min_p, max_p), 2)
        })
        pid_to_cat[pid] = cat
    
    products_df = pd.DataFrame(products)
    all_pids = products_df['Product_ID'].tolist()

    # 2. Customer Base
    customer_ids = [f'CUST-{str(i+1).zfill(4)}' for i in range(num_customers)]
    # Assign customer weights to simulate "High-Value" vs "Occasional" buyers
    cust_weights = np.random.dirichlet(np.ones(num_customers), size=1)[0]

    # 3. Time Series Data (Past 540 days)
    start_date = datetime(2024, 1, 1)
    
    # 4. Realistic Rules (Item Affinity)
    affinities = [
        ('P001', 'P005', 0.8), # Bread & Butter (High)
        ('P010', 'P012', 0.7), # Milk & Cereal
        ('P020', 'P021', 0.6), # Beer & Chips
        ('P030', 'P031', 0.9), # Coffee & Sugar
        ('P045', 'P046', 0.75), # Pasta & Sauce
    ]

    transactions = []
    
    for t_id in range(1, num_transactions + 1):
        cust_id = np.random.choice(customer_ids, p=cust_weights)
        date = start_date + timedelta(days=random.randint(0, 540))
        
        # Basket Size (Poisson distribution)
        basket_size = np.random.poisson(lam=4) + 1
        basket_size = min(max(basket_size, 1), 15)
        
        basket = set()
        
        # Apply Affinity Rules
        for a, b, prob in affinities:
            if random.random() < prob * 0.2: # Trigger rule with fractional probability
                basket.add(a)
                basket.add(b)
        
        # Fill remaining
        while len(basket) < basket_size:
            basket.add(random.choice(all_pids))
            
        for pid in basket:
            qty = random.randint(1, 5)
            row = products_df[products_df['Product_ID'] == pid].iloc[0]
            
            transactions.append({
                'Transaction_ID': f'T{str(t_id).zfill(6)}',
                'Customer_ID': cust_id,
                'Date': date.strftime('%Y-%m-%d'),
                'Product_ID': pid,
                'Product_Name': row['Product_Name'],
                'Category': row['Category'],
                'Quantity': qty,
                'Unit_Price': row['Unit_Price'],
                'Total_Price': round(qty * row['Unit_Price'], 2)
            })
            
    df = pd.DataFrame(transactions)
    
    # Save processed data
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/transactions.csv', index=False)
    products_df.to_csv('data/products.csv', index=False)
    
    print(f"✅ Data Generation Complete. Total Rows: {len(df)}")
    return df

if __name__ == "__main__":
    generate_synthetic_data()

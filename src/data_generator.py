import pandas as pd
import numpy as np
import random
import os

def generate_synthetic_data(num_transactions=10000, num_products=150):
    """
    Generates a synthetic retail dataset with realistic buying behavior.
    """
    # 1. Generate Products
    categories = ['Bakery', 'Dairy', 'Produce', 'Meat', 'Beverages', 'Canned Goods', 'Frozen', 'Snacks']
    products = []
    for i in range(num_products):
        cat = random.choice(categories)
        products.append({
            'Product_ID': f'P{str(i+1).zfill(3)}',
            'Product_Name': f'{cat} Item {i+1}',
            'Category': cat,
            'Price': round(random.uniform(1.0, 50.0), 2)
        })
    
    products_df = pd.DataFrame(products)
    
    # 2. Define Realistic Rules (Item Affinity)
    # Pairs often bought together: (Product_ID, Product_ID)
    affinities = [
        ('P001', 'P005'), # Bread & Butter
        ('P010', 'P012'), # Milk & Cereal
        ('P020', 'P021'), # Beer & Chips
        ('P030', 'P031'), # Coffee & Sugar
        ('P045', 'P046'), # Pasta & Sauce
    ]
    
    # 3. Generate Transactions
    transactions = []
    all_pids = products_df['Product_ID'].tolist()
    
    for t_id in range(1, num_transactions + 1):
        # Determine number of items in this basket
        basket_size = np.random.geometric(p=0.3) + 1 # Geometric distribution for realistic basket sizes
        basket_size = min(basket_size, 15) # Cap it
        
        basket = set()
        
        # Add affinity items with probability
        for a, b in affinities:
            if random.random() < 0.15: # 15% chance to trigger a rule
                basket.add(a)
                basket.add(b)
        
        # Fill the rest with random items
        while len(basket) < basket_size:
            basket.add(random.choice(all_pids))
            
        for pid in list(basket):
            transactions.append({
                'Transaction_ID': f'T{str(t_id).zfill(5)}',
                'Product_ID': pid
            })
            
    transactions_df = pd.DataFrame(transactions)
    
    # 4. Merge Data
    final_df = transactions_df.merge(products_df, on='Product_ID')
    
    # Save files
    os.makedirs('data', exist_ok=True)
    final_df.to_csv('data/transactions.csv', index=False)
    products_df.to_csv('data/products.csv', index=False)
    
    print(f"Dataset generated: {num_transactions} transactions, {len(final_df)} line items.")
    return final_df

if __name__ == "__main__":
    generate_synthetic_data()

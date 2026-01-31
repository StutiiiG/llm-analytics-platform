import pandas as pd
import numpy as np
import os

os.makedirs('sample_data', exist_ok=True)

# Generate sample data
np.random.seed(42)
dates = pd.date_range('2023-01-01', '2024-12-31', freq='D')

df = pd.DataFrame({
    'date': dates,
    'sales': np.random.normal(5000, 1000, len(dates)),
    'category': np.random.choice(['Electronics', 'Furniture', 'Clothing'], len(dates)),
    'region': np.random.choice(['North', 'South', 'East', 'West'], len(dates)),
    'units_sold': np.random.randint(10, 100, len(dates))
})

# Save to CSV (no space in path!)
df.to_csv('/Users/stuti/Desktop/LLM-Analytics-Assistant/sample data/sales_data.csv', index=False)
print("Sample data created at: sample_data/sales_data.csv")
print(f"Generated {len(df):,} rows")
print(f" Date range: {df['date'].min()} to {df['date'].max()}")
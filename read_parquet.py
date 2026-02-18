import pandas as pd

# Read the parquet files
nodes_df = pd.read_parquet('nodes.parquet')
relationships_df = pd.read_parquet('relationships.parquet')

# Display basic info about the dataframes
print("Nodes DataFrame:")
print(f"Shape: {nodes_df.shape}")
print(f"Columns: {list(nodes_df.columns)}")
print("\nFirst 5 rows:")
print(nodes_df.head())

print("\n" + "="*50 + "\n")

print("Relationships DataFrame:")
print(f"Shape: {relationships_df.shape}")
print(f"Columns: {list(relationships_df.columns)}")
print("\nFirst 5 rows:")
print(relationships_df.head())
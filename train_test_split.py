from sklearn.model_selection import train_test_split
import pandas as pd

# Load the dataset
df = pd.read_csv(r'D:\skin\HAM10000_metadata.csv')

# Split the data
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['dx'])

# Save the split data
train_df.to_csv('train.csv', index=False)
test_df.to_csv('test.csv', index=False)


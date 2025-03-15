from sklearn.model_selection import train_test_split
import pandas as pd

# Load the dataset
df = pd.read_csv(r'D:\skin\HAM10000_metadata.csv')

# Split the data
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['dx'])

test_df,val_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['dx'])

# Save the split data
train_df.to_csv('train.csv', index=False)
test_df.to_csv('test.csv', index=False)
val_df.to_csv('val.csv', index=False)



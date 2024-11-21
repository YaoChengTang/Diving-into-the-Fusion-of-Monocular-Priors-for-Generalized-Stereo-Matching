import os
import sys
import pandas as pd

LOG_ROOT = os.getenv('LOG_ROOT', default="logs")

# Read the Excel file
excel_file = os.path.join(LOG_ROOT, "eval.xlsx")
xls = pd.ExcelFile(excel_file)

# Initialize the merged DataFrame
merged_df = pd.DataFrame()

# Define the key field for merging
key_column = 'Model'

# Iterate through each sheet in the Excel file
for sheet_name in xls.sheet_names:
    df = pd.read_excel(excel_file, sheet_name=sheet_name)
    # If it's the first DataFrame, assign it directly to merged_df
    if merged_df.empty:
        merged_df = df
    else:
        # Merge the DataFrame based on the key field into the same row
        merged_df = pd.merge(merged_df, df, on=key_column, how='outer')

# Move the key column to the first position
merged_df = merged_df[[key_column] + [col for col in merged_df.columns if col != key_column]]

# Write the merged DataFrame to a new Excel file
merged_excel_file = os.path.join(LOG_ROOT, 'merged_eval.xlsx')
merged_df.to_excel(merged_excel_file, index=False)

print("Sheets have been merged successfully!")

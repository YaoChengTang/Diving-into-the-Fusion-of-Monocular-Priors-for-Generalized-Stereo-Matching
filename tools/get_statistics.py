import pandas as pd
import argparse

def calculate_metrics(file_path, start_row, end_row):
    # Read the Excel file
    df = pd.read_excel(file_path)
    
    # Select the rows within the given range (note: start_row and end_row are 1-based row numbers)
    data = df.iloc[start_row-1:end_row]
    
    # Calculate mean and standard deviation
    epe_mean = data['middleburyH-epe'].mean()
    epe_std = data['middleburyH-epe'].std()
    d1_mean = data['middleburyH-d1'].mean()
    d1_std = data['middleburyH-d1'].std()
    
    # Print the results
    print(f"middleburyH-epe Mean: {epe_mean:.2f}, Standard Deviation: {epe_std:.2f}")
    print(f"middleburyH-d1 Mean: {d1_mean:.2f}, Standard Deviation: {d1_std:.2f}")
    print(f"{epe_mean:.2f}+-{epe_std:.2f}    {d1_mean:.2f}+-{d1_std:.2f}")

if __name__ == "__main__":
    # Set up command-line arguments
    parser = argparse.ArgumentParser(description="Calculate metrics for specified rows in an Excel file.")
    parser.add_argument("file_path", type=str, help="Path to the Excel file")
    parser.add_argument("start_row", type=int, help="Start row (1-based index)")
    parser.add_argument("end_row", type=int, help="End row (1-based index)")
    
    args = parser.parse_args()
    
    # Call the function and pass the parameters
    calculate_metrics(args.file_path, args.start_row, args.end_row)

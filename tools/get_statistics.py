import pandas as pd
import argparse

def calculate_metrics(file_path, start_row, end_row):
    # 读取Excel文件
    df = pd.read_excel(file_path)
    
    # 选择给定范围的行（注意start_row和end_row是基于1的行号）
    data = df.iloc[start_row-1:end_row]
    
    # 计算平均值和标准差
    epe_mean = data['middleburyH-epe'].mean()
    epe_std = data['middleburyH-epe'].std()
    d1_mean = data['middleburyH-d1'].mean()
    d1_std = data['middleburyH-d1'].std()
    
    # 输出结果
    print(f"middleburyH-epe Mean: {epe_mean:.2f}, Standard Deviation: {epe_std:.2f}")
    print(f"middleburyH-d1 Mean: {d1_mean:.2f}, Standard Deviation: {d1_std:.2f}")
    print(f"{epe_mean:.2f}+-{epe_std:.2f}    {d1_mean:.2f}+-{d1_std:.2f}")

if __name__ == "__main__":
    # 设置命令行参数
    parser = argparse.ArgumentParser(description="Calculate metrics for specified rows in an Excel file.")
    parser.add_argument("file_path", type=str, help="Path to the Excel file")
    parser.add_argument("start_row", type=int, help="Start row (1-based index)")
    parser.add_argument("end_row", type=int, help="End row (1-based index)")
    
    args = parser.parse_args()
    
    # 调用函数并传递参数
    calculate_metrics(args.file_path, args.start_row, args.end_row)


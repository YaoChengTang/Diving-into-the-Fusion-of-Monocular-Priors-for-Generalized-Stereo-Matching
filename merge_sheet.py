import os
import sys
import pandas as pd


LOG_ROOT  = os.getenv('LOG_ROOT', default="logs")


# 读取 Excel 文件
excel_file = os.path.join(LOG_ROOT, "eval.xlsx")
xls = pd.ExcelFile(excel_file)

# 初始化合并后的 DataFrame
merged_df = pd.DataFrame()

# 定义要合并的关键字段
key_column = 'Model'

# 逐个读取每个 sheet
for sheet_name in xls.sheet_names:
    df = pd.read_excel(excel_file, sheet_name=sheet_name)
    # 如果是第一个 DataFrame，则直接赋值给 merged_df
    if merged_df.empty:
        merged_df = df
    else:
        # 使用 merge() 函数根据关键字段合并到同一行
        merged_df = pd.merge(merged_df, df, on=key_column, how='outer')

# 将关键字移动到第一列
merged_df = merged_df[[key_column] + [col for col in merged_df.columns if col != key_column]]

# 将合并后的 DataFrame 写入新的 Excel 文件
merged_excel_file = os.path.join(LOG_ROOT, 'merged_eval.xlsx')
merged_df.to_excel(merged_excel_file, index=False)

print("Sheets have been merged successfully!")
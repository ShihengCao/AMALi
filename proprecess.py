import pandas as pd
import os, sys

# Project Name format is project_name
Project_name = sys.argv[1]
workspace_path = os.path.join("..", "GCoM_outputs")
outputs_dir = os.path.join(workspace_path, Project_name)
output_file = os.path.join(workspace_path, Project_name + ".csv")
ground_truth_dir = os.path.join('..', "HW")
ground_truth_file = os.path.join(ground_truth_dir, Project_name + "-bf16_10_m_processed.csv")

files = os.listdir(outputs_dir)

def get_idx(name):
    name_token = name.split('_')
    idx = int(name_token[0])
    return idx

def cmp_file_name(a):
    idxa = get_idx(a)
    return idxa

files.sort(key=cmp_file_name)

# Initialize the lists to store data
columns = []
data_lists = {}

# Read the first file to determine the order of keys
with open(os.path.join(outputs_dir, files[0]), 'r') as f:
    line = f.readline()
    columns = line[:-1].split('!')

    # Initialize data_lists with empty lists for each column
    for col in columns:
        data_lists[col] = []

# Now read all files and fill in the data
for file in files:
    with open(os.path.join(outputs_dir, file), 'r') as f:
        line = f.readline()  # Skip the header line since we already determined it
        line = f.readline()

        while line:
            tokens = line[:-1].split('!')
            reduction_dict = dict(zip(columns, tokens))
            
            # Append values or None if key is missing
            for col in columns:
                if col in reduction_dict:
                    try:
                        data_lists[col].append(float(reduction_dict[col]))
                    except ValueError:
                        data_lists[col].append(reduction_dict[col])
                else:
                    data_lists[col].append(None)

            line = f.readline()

# Ensure that all lists have the same length
length = len(next(iter(data_lists.values())))
assert all(len(lst) == length for lst in data_lists.values()), "Lists are not of the same length"

# Create DataFrame from collected data
df = pd.DataFrame(data_lists, columns=columns)

def add_average_to_df(df, ground_truth_file):
    # 读取ground_truth文件
    ground_truth_df = pd.read_csv(ground_truth_file)
    
    # 确保kernel_id列存在且为整数类型
    if 'kernel_id' not in df.columns:
        raise ValueError("DataFrame中缺少kernel_id列")
    df['kernel_id'] = df['kernel_id'].astype(int)
    
    # 确保Average列存在于ground_truth_file中
    if 'Average' not in ground_truth_df.columns:
        raise ValueError("ground_truth_file中缺少Average列")
    
    # 创建一个新的Series,其index是ground_truth_df的行号(从1开始)
    average_series = pd.Series(ground_truth_df['Average'].values, index=range(1, len(ground_truth_df) + 1))
    
    # 使用kernel_id映射到average_series并添加新列
    df['Ground_Truth_Average'] = df['kernel_id'].map(average_series)
    
    return df

def save_df_to_csv(df, output_file):
    df.to_csv(output_file, index=False)
    print(f"更新后的文件已保存为: {output_file}")

df = add_average_to_df(df, ground_truth_file)
df.to_csv(output_file, index=False, sep=',')

# Calculate and print the sums
print(sum(df['GCoM']),
      sum(df['simulation_time_memory'] + df['simulation_time_compute'] + df['simulation_time_parse']),)

# 读取两个 CSV 文件
file_10_m = ground_truth_file
file_fp16 = output_file

df_10_m = pd.read_csv(file_10_m)
df_fp16 = pd.read_csv(file_fp16)

# 创建一个空的 DataFrame，用于存储最终的匹配数据
merged_df = df_10_m.copy()

# 定义要添加的列
columns_to_add = ['GCoM']

# 初始化这些列为 NaN
for col in columns_to_add:
    merged_df[col] = pd.NA

# 遍历 df_10_m 中的每一行
for idx, row in df_10_m.iterrows():
    kernel_name = row['Kernel Name']
    avg_value = row['Average']
    #print(row["similar_kernel_ids"])
    #print(df_fp16['kernel_id'])
    closest_row_data = df_fp16[df_fp16['kernel_id'] == int(row["similar_kernel_ids"])]
    #print(closest_row_data)
    for col in columns_to_add:
        if not closest_row_data[col].empty:
            merged_df.at[idx, col] = float(closest_row_data[col].iloc[0])
        #print(closest_row_data[col])
        # merged_df.at[idx, col] = float(closest_row_data[col])        


# 保存合并后的文件
merged_df.to_csv(os.path.join(workspace_path, Project_name + "_extracted.csv"), index=False)

# 按照 'similar_kernel_ids' 列分组，计算每组 'Average' 的总和
grouped_sum = df_10_m.groupby('similar_kernel_ids')['Average'].sum().reset_index()

# 按照 'Average' 的总和进行降序排序
sorted_groups = grouped_sum.sort_values(by='Average', ascending=False)

# 计算总的 Average 和
total_average_sum = sorted_groups['Average'].sum()

# 累加到总和的90%
cumulative_sum = 0
threshold = total_average_sum * 0.9
selected_ids = []

for idx, row in sorted_groups.iterrows():
    cumulative_sum += row['Average']
    selected_ids.append(int(row['similar_kernel_ids']))  # 转换为整数
    if cumulative_sum >= threshold:
        break

# 筛选 df_fp16 中的行，kernel_id 转为整数进行比较
filtered_df = df_fp16[df_fp16['kernel_id'].isin(map(int, selected_ids))]

# 保存到 CSV 文件
output_path = os.path.join(workspace_path, Project_name + "_selected.csv")
filtered_df.to_csv(output_path, index=False)

print(f"Selected rows saved to {output_path}")

'''
kernel 映射
'''
import os

def select_kernels_and_save(df_10_m, df_fp16, workspace_path, Project_name, threshold = 0.9):
    # 按照 'similar_kernel_ids' 列分组，计算每组 'Average' 的总和
    grouped_sum = df_10_m.groupby('similar_kernel_ids')['Average'].sum().reset_index()

    # 按照 'Average' 的总和进行降序排序
    sorted_groups = grouped_sum.sort_values(by='Average', ascending=False)

    # 计算总的 Average 和
    total_average_sum = sorted_groups['Average'].sum()

    # 累加到总和的90%
    cumulative_sum = 0
    threshold = total_average_sum * threshold
    selected_ids = []

    for idx, row in sorted_groups.iterrows():
        cumulative_sum += row['Average']
        selected_ids.append(int(row['similar_kernel_ids']))  # 转换为整数
        if cumulative_sum >= threshold:
            break

    # 筛选 df_fp16 中的行，kernel_id 转为整数进行比较
    filtered_df = df_fp16[df_fp16['kernel_id'].isin(map(int, selected_ids))]

    # 保存到 CSV 文件
    output_path = os.path.join(workspace_path, f"{Project_name}_selected.csv")
    filtered_df.to_csv(output_path, index=False)
    print(f"Selected kernels saved to: {output_path}")
    return filtered_df

# 使用这个函数时，你需要提供正确的参数：
# select_kernels_and_save(df_10_m, df_fp16, '/path/to/workspace', 'YourProjectName')
def split_dataframe(df, row_index):
    """
    将给定的 DataFrame 在指定行索引处分割成两个 DataFrame。
    
    参数:
        df (pd.DataFrame): 要分割的原始 DataFrame。
        row_index (int): 分割点的行索引（从0开始）。
        
    返回:
        tuple: 包含两个 DataFrame 的元组，第一个是从开始到分割点（包括），第二个是从分割点后一个位置到最后。
    """
    # 确保行索引在有效范围内
    if row_index < 0 or row_index >= len(df):
        raise ValueError("row_index 必须在 DataFrame 的有效行索引范围内。")

    # 分割 DataFrame
    df_first = df.iloc[:row_index+1]  # 包括第 row_index 行
    df_second = df.iloc[row_index+1:]  # 从第 row_index+1 行开始
    
    return df_first, df_second

# 假设 df_10_m 是你的原始 DataFrame
# 使用函数来分割 DataFrame
argmax_indices = df_10_m.index[df_10_m['Kernel Name'].str.contains('ArgMax', na=False)].tolist()[0] + 8
df_10_m_first, df_10_m_second = split_dataframe(df_10_m, argmax_indices)  # 注意：因为是基于0的索引，所以这里使用1359代表第1360行
# df_10_m_first, df_10_m_second = split_dataframe(df_10_m, 1359)  # 注意：因为是基于0的索引，所以这里使用1359代表第1360行

filtered_df_first = select_kernels_and_save(df_10_m_first, df_fp16, workspace_path, Project_name+"_first")
filtered_df_second = select_kernels_and_save(df_10_m_second, df_fp16, workspace_path, Project_name+"_second", threshold=0.8)

'''
    计算MAPE
'''

import numpy as np
# 计算 MAPE 的函数

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-5))) * 100

# 获取 Ground_Truth_Average 列作为真实值
y_true_first = filtered_df_first['Ground_Truth_Average']
y_true_second = filtered_df_second['Ground_Truth_Average']

# 计算每一列与 Ground_Truth_Average 之间的 MAPE for first part
mape_gcom_first = mean_absolute_percentage_error(y_true_first, filtered_df_first['GCoM'])
# 打印结果 for first part
print(f'First Part - MAPE of GCoM: {mape_gcom_first:.2f}%')

# 计算每一列与 Ground_Truth_Average 之间的 MAPE for second part
mape_gcom_second = mean_absolute_percentage_error(y_true_second, filtered_df_second['GCoM'])

# 打印结果 for second part
print(f'Second Part - MAPE of GCoM: {mape_gcom_second:.2f}%')

# 拼接两个 DataFrame
combined_df = pd.concat([filtered_df_first, filtered_df_second])

# 获取拼接后 DataFrame 中的 Ground_Truth_Average 列作为真实值
y_true_combined = combined_df['Ground_Truth_Average']

# 计算每一列与 Ground_Truth_Average 之间的 MAPE for the combined DataFrame
mape_gcom_combined = mean_absolute_percentage_error(y_true_combined, combined_df['GCoM'])

# 打印结果 for the combined DataFrame
print(f'Combined - MAPE of GCoM: {mape_gcom_combined:.2f}%')
import pandas as pd
import os, sys

# Project Name format is project_name
Project_name = sys.argv[1]
workspace_path = os.path.join("..", "GCoM_outputs")
outputs_dir = os.path.join(workspace_path, Project_name)
output_file = os.path.join(workspace_path, Project_name + ".csv")
ground_truth_dir = os.path.join('.', "HW")
ground_truth_file = os.path.join(ground_truth_dir, Project_name + "_10_m.csv")

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
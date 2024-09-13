import pandas as pd
import os, sys

# Project Name format is project_name
Project_name = sys.argv[1]
outputs_dir = os.path.join("../outputs", Project_name)
output_file = os.path.join("../outputs", Project_name + ".csv")

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
df.to_csv(output_file, index=False, sep=',')

# Calculate and print the sums
print(sum(df['GCoM+KLL+ID']),
      sum(df['simulation_time_memory'] + df['simulation_time_compute'] + df['simulation_time_parse']),
      "{:.4f}".format(sum(df['GCoM']) / sum(df['GCoM+KLL+ID'])))
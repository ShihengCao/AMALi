import pandas as pd
import sys, ast

def parse_tuple_to_product(s):
    """
    change string like "(128, 1, 1)" into 128*1*1 = 128
    """
    try:
        tup = ast.literal_eval(s)
        if isinstance(tup, tuple):
            prod = 1
            for v in tup:
                prod *= int(v)
            return prod
        else:
            return int(s)
    except Exception:
        return s
    
def process_csv(input_csv, output_csv, group_size=10):
    # 先读前两行，检查第二行是不是单位行
    with open(input_csv, "r") as f:
        first_line = f.readline()
        second_line = f.readline()

    skiprows = [1] if "cycle" in second_line.lower() else []

    # 读入 CSV，跳过单位行
    df = pd.read_csv(input_csv, skiprows=skiprows)

    # 保留需要的列
    df = df[["Block Size", "Grid Size", "gpc__cycles_elapsed.avg",
             "gpc__cycles_elapsed.min", "gpc__cycles_elapsed.max"]]
    # 转换 Block Size / Grid Size
    df["Block Size"] = df["Block Size"].apply(parse_tuple_to_product)
    df["Grid Size"] = df["Grid Size"].apply(parse_tuple_to_product)
    results = []

    for i in range(0, len(df), group_size):
        group = df.iloc[i:i+group_size]

        if len(group) < group_size:
            # 最后一组不足 group_size 就跳过
            continue

        block_size = group["Block Size"].iloc[0]
        grid_size = group["Grid Size"].iloc[0]

        avg_val = group["gpc__cycles_elapsed.avg"].mean()
        min_val = group["gpc__cycles_elapsed.min"].mean()
        max_val = group["gpc__cycles_elapsed.max"].mean()

        results.append([block_size, grid_size, avg_val, min_val, max_val])

    out_df = pd.DataFrame(results, columns=["Block Size", "Grid Size", "avg", "min", "max"])
    out_df.to_csv(output_csv, index=False)

    print(f"Processed {len(results)} groups, saved to {output_csv}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python process_ncu_csv.py input.csv output.csv")
        sys.exit(1)

    process_csv(sys.argv[1], sys.argv[2])

import re
import os
import glob
import csv

def parse_log(app_name):
    log_dir = os.path.join("logs", app_name)
    # print(os.getcwd())
    if not os.path.isdir(log_dir):
        print(f"No log directory found for application: {app_name}", log_dir)
        return

    tag_line_re = re.compile(r'^\s*###\s*(.*?)\s*###\s*$')
    info_name_re = re.compile(r'^(?P<kernel>.+?)_info\.log$')
    all_logs = glob.glob(os.path.join(log_dir, "*_info.log"))
    for log_path in all_logs:
        m = info_name_re.match(os.path.basename(log_path))
        if not m:
            continue
        kernel_id = m.group("kernel")

        try:
            with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.read().splitlines(keepends=True)
        except FileNotFoundError:
            continue

        remaining = []
        capturing = False
        current_tag = None
        buf = []

        def flush_buf(tag, data):
            if not data:
                return
            content = "".join(data)
            if content and not content.endswith("\n"):
                content += "\n"
            safe_tag = re.sub(r"\s+", "_", tag.strip()) if tag else "untagged"
            out_path = os.path.join(log_dir, f"{kernel_id}_{safe_tag}.log")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "a", encoding="utf-8") as out_f:
                out_f.write(content)

        for line in lines:
            tm = tag_line_re.match(line)
            if tm:
                tag_name = tm.group(1)
                if not capturing:
                    capturing = True
                    current_tag = tag_name
                    buf = []
                else:
                    if tag_name == current_tag:
                        flush_buf(current_tag, buf)
                        capturing = False
                        current_tag = None
                        buf = []
                    else:
                        buf.append(line)
                continue

            if capturing:
                buf.append(line)
            else:
                remaining.append(line)

        if capturing:
            remaining.append(f"### {current_tag} ###\n")
            remaining.extend(buf)

        with open(log_path, "w", encoding="utf-8") as f:
            f.writelines(remaining)

def parse_out(app_name, front_keys=None):
    """
    Parse kernel output files and create a CSV file while preserving the original key order,
    but move a subset of keys to the front.

    To customize the front keys, pass them as a list of strings.
    
    Args:
        app_name: Application name
        front_keys: List of keys to move to the front (optional). Defaults to ["kernel_id","kernel_name","AMALi (GCoM+TCM+KLL+ID)","selected","wait","drain","long_scoreboard","short_scoreboard","C_idle_ij_orig","C_idle_ij_ID","math_pipe_throttle","tex_throttle","lg_throttle","S_MSHR_i","S_NoC_i","S_Dram_i","C_idle_i_orig","C_idle_i_ID","no_instructions_and_imc_miss","GCoM+TCM","GCoM+TCM+KLL"] if not provided.
    """
    out_dir = os.path.join("outputs", app_name)
    if not os.path.isdir(out_dir):
        print(f"No output directory found for application: {app_name}")
        return
    
    all_data = []
    column_order = []
    seen_columns = set()
    
    for out_path in sorted(glob.glob(os.path.join(out_dir, "*_all_info.out"))):
        try:
            with open(out_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read().strip()
            lines = content.split('\n')
            if len(lines) < 2:
                continue
                
            keys = lines[0].split('!')
            values = lines[1].split('!')
            
            if len(keys) != len(values):
                continue

            # Preserve original key order as first encountered across files
            for k in keys:
                if k not in seen_columns:
                    seen_columns.add(k)
                    column_order.append(k)
                
            processed_values = []
            for value in values:
                value = value.strip()
                try:
                    if '.' in value or 'e' in value.lower():
                        processed_values.append(float(value))
                    else:
                        processed_values.append(int(value))
                except ValueError:
                    processed_values.append(value)
            
            row_data = dict(zip(keys, processed_values))
            all_data.append(row_data)
            
        except (FileNotFoundError, UnicodeDecodeError):
            continue
    
    if not all_data:
        return

    # If front_keys is not provided, use default values
    if front_keys is None:
        front_keys = ["kernel_id","kernel_name","AMALi (GCoM+TCM+KLL+ID)","selected","wait","drain","long_scoreboard","short_scoreboard","C_idle_ij_orig","C_idle_ij_ID","math_pipe_throttle","tex_throttle","lg_throttle","S_MSHR_i","S_NoC_i","S_Dram_i","C_idle_i_orig","C_idle_i_ID","no_instructions_and_imc_miss","GCoM+TCM","GCoM+TCM+KLL"]

    # Reorder columns: front keys first (in given order, if present), then the rest
    ordered_columns = [k for k in front_keys if k in column_order] + [k for k in column_order if k not in front_keys]
    
    csv_path = os.path.join(out_dir, f"{app_name}_parsed.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=ordered_columns)
        writer.writeheader()
        writer.writerows(all_data)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python parse.py <app_name>")
        sys.exit(1)
    
    app_name = sys.argv[1]
    parse_log(app_name)
    parse_out(app_name)
    print(f"Parsing completed for application: {app_name}")

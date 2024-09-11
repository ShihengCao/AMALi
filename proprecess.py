import pandas as pd
import os, sys

# Project Name format is project_name
Project_name = sys.argv[1]
outputs_dir = os.path.join("../outputs",Project_name)
# if outputs_dir not in os.listdir():
#     os.mkdir(outputs_dir)

output_file = Project_name + ".csv"

files= os.listdir(outputs_dir)
# files.remove("all_kernels_all_info.out")
def get_idx(name):
    name_token = name.split('_')
    idx = int(name_token[0])
    return idx
def cmp_file_name(a):
    idxa = get_idx(a)
    return idxa
files.sort(key = cmp_file_name)
# f = open(os.path.join(outputs_dir,filename))               # 返回一个文件对象 
# line = f.readline()               # 调用文件的 readline()方法 
kernel_name_GCoM = []
GCoM_KLL_ID = []
GCoM_KLL = []
GCoM_ID = []
GCoM = []
Kernel_id = []
time_GCoM = []

for file in files:
    # if get_idx(file) != 40:
    #     continue
    f = open(os.path.join(outputs_dir,file))  
    line = f.readline() 
    keys = line[:-1].split('!')
    line = f.readline()
    while line:                   # 后面跟 ',' 将忽略换行符 
        #print(line, end = '')　      # 在 Python 3 中使用 
        
        tokens = line[:-1].split('!')
        reduction_dict = {}
        for k,v in zip(keys, tokens):
            reduction_dict[k] = v
        kernel_name_GCoM.append(reduction_dict["kernel_name"])
        Kernel_id.append(reduction_dict["kernel_id"])
        GCoM_KLL_ID.append(
            float(reduction_dict["GCoM+KLL+ID"]) 
        )
        GCoM_KLL.append(
            float(reduction_dict["GCoM+KLL"]) 
        )
        GCoM_ID.append(
            float(reduction_dict["GCoM+ID"])
        )
        GCoM.append(
            float(reduction_dict["GCoM"]) 
        )

        time_GCoM.append(float(reduction_dict["simulation_time_memory"]) + float(reduction_dict["simulation_time_compute"])+ float(reduction_dict["simulation_time_parse"]))
        line = f.readline()
    f.close()  

dataframe = pd.DataFrame({'kernel_id':Kernel_id,
                          'kernel_name_GCoM':kernel_name_GCoM,
                          'GCoM':GCoM,                          
                          'GCoM+KLL':GCoM_KLL,
                          'GCoM+KLL+ID':GCoM_KLL_ID,
                          'GCoM+ID':GCoM_ID,
                          'time_GCoM':time_GCoM,
                          })

dataframe.to_csv(output_file,index=False,sep=',')
print(sum(GCoM_KLL_ID),
    #   sum(insts_GCoM),
      sum(time_GCoM),
      "{:.4f}".format(sum(GCoM)/sum(GCoM_KLL_ID)))
import os, sys, math
import numpy as np
from sklearn.cluster import KMeans
from scipy import special as sp

# functional_units_list = ["iALU", "fALU", "hALU", "dALU", 
#                          "SFU", "dSFU", 
#                          "LDST",
#                         #  "iTCU", "hTCU", "fTCU", "dTCU", 
#                          "BRA", "EXIT",]

# uniform_insts_list = ["R2UR","REDUX", "S2UR","UBMSK","UBREV","UCLEA","UF2FP",  "UFLO" ,"UIADD3" ,"UIADD3.64" ,"UIMAD" ,
# "UISETP", "ULDC", "ULEA","ULOP","ULOP3","ULOP32I","UP2UR","UMOV","UP2UR","UPLOP3","UPOPC","UPRMT","UPSETP","UR2UP",
# "USEL","USGXT","USHF","USHL","USHR","VOTEU",] 

# def get_unit_idx(unit):
#     result = -1
#     for i in range(len(functional_units_list)):
#         if functional_units_list[i] == unit:
#             result = i
#     return result

def sm_id_str_to_int(sm_id_str):
    return int(sm_id_str.split('#')[0])

def rptv_warp_select(kmeans_features):
    '''
        use kmeans algorithm to find the represetative warp.

        Args:
            kmeans_features (list): a list

        Returns:
            all_center_warp_list: a list of warps idx closest to the kmeans cluster center 
            represetative_index: int, index of the represetative warp
    '''
    # kmeans_features = np.array(kmeans_features)
    # calculate the avg of kmeans_features in every column
    # kmeans_features_avg = np.mean(kmeans_features, axis = 0)
    # divide corresponding feature avg in every column
    # for feature in kmeans_features:
    #     for i in range(len(feature)):
    #         if feature[i] == 0:
    #             continue
    #         feature[i] /= kmeans_features_avg[i]
    # do kmeans algorithm
    n_clusters = min(4, len(kmeans_features))
    kmeans = KMeans(n_clusters = n_clusters,
                    random_state = 0, n_init = 10).fit(kmeans_features)
    # print_jpg(kmeans, n_clusters, kmeans_features)
    # count the number of each cluster
    count_cluster = {}
    for i in kmeans.labels_:
        if i not in count_cluster:
            count_cluster[i] = 1
        else:
            count_cluster[i] += 1
    # find the largest cluster
    max_index = max(count_cluster, key=count_cluster.get)
    max_value = count_cluster[max_index]
    # calculate the representive warp which is the nearest warp to the largest cluster center
    min_dist_v = 1e9
    rptv_index = 0
    for i in range(len(kmeans.labels_)):
        if kmeans.labels_[i] == max_index:
            dist = np.linalg.norm(kmeans_features[i] - kmeans.cluster_centers_[kmeans.labels_[i]])
            if dist < min_dist_v:
                min_dist_v = dist
                rptv_index = i
    # represetative_index = -1
    all_center_warp_index = [-1] * n_clusters
    min_dist = [1e9] * n_clusters
    #print(len(min_dist),kmeans.labels_)
    for i in range(len(kmeans.labels_)):
        label_i = kmeans.labels_[i]
        dist = np.linalg.norm(kmeans_features[i] - kmeans.cluster_centers_[label_i])
        if dist < min_dist[label_i]:
            min_dist[label_i] = dist
            all_center_warp_index[label_i] = i
    # print(count_cluster)
    return all_center_warp_index, rptv_index

def dump_output(pred_out):

    kernel_prefix = str(pred_out["kernel_id"])+"_"+pred_out["ISA"] +"_g"+pred_out["granularity"]
    output_path = os.path.join(pred_out["app_path"],"output")
    if not os.path.exists(output_path):  
        os.makedirs(output_path)
    outF = open(os.path.join(output_path, "kernel_"+kernel_prefix+".out"), "w+")

def print_config_error(config_name, flag=0):
	if flag == 1:
		print("\n[Error]\nGPU Compute Capabilty \"" +config_name+"\" is not supported")
		sys.exit(1)
	elif flag == 2:
		print("\n[Error]\n\""+config_name+"\" is not defined in the hardware compute capability file")
		sys.exit(1)
	else:
		print("\n[Error]\n\""+config_name+"\" config is not defined in the hardware configuration file")
		sys.exit(1)


def print_warning(arg1, arg2, flag=False):
	if flag:
		print("\n[Warning]\n\"" + arg1 + "\" is not defined in the config file "+\
		"assuming L1 cache is "+ arg2 + "\n")
	else:
		print("\n[Warning]\n\"" + arg1 + "\" can't be more than " + arg2\
		 	+" registers\n assuming \"" + arg1 + "\" = " + arg2 + "\n")


def ceil(x, s):
	return s * math.ceil(float(x)/s)

def qfunc(arg):
    return 0.5-0.5*sp.erf(arg/1.41421)

def floor(x, s):
    return s * math.floor(float(x)/s)

def print_output_info(pred_out, rptv_warp_GCoM_output):
    print("| kernel id:",pred_out["kernel_id"])
    print("| kernel name", pred_out["kernel_name"][:min(len(pred_out["kernel_name"]),20)])
    print(
        "| simulation time:{:.4f}s\n| AMAT:{:.4f}\n| ACPAO:{:.4f}\n| cpi:{:.8f}".format(pred_out["simulation_time_memory"] + pred_out["simulation_time_compute"] + pred_out["simulation_time_parse"], 
        pred_out["AMAT"], 
        pred_out["ACPAO"], 
        pred_out["cpi"]) 		
    )
    print("| Analytical model output:",rptv_warp_GCoM_output)
    print('+'+'-'*30)

def write_to_file(pred_out):
    # mkdir if target dir is not exist
    output_dir = os.path.join(".","outputs")
    app_name = pred_out["app_name"]
    app_output_dir = os.path.join(output_dir,app_name)
    # check output_dir is exist or not
    if not os.path.exists(app_output_dir):
        os.makedirs(app_output_dir, exist_ok=True)
    # write outputs to files
    with open(os.path.join(app_output_dir,str(pred_out["kernel_id"]) + "_all_info.out"),'a+') as f:
        f.write('!'.join([str(i) for i in list(pred_out.keys())])+'\n')  
        f.write('!'.join([str(i) for i in list(pred_out.values())])+'\n')  
class Logger():
    def __init__(self, pred_out, is_active=True):
        self.pred_out = pred_out
        self.is_active = is_active
        # mkdir if target dir is not exist
        output_dir = os.path.join(".","logs")
        app_name = pred_out["app_name"]
        app_output_dir = os.path.join(output_dir,app_name)
        # check output_dir is exist or not
        if not os.path.exists(app_output_dir):
            os.makedirs(app_output_dir, exist_ok=True)
        self.f = open(os.path.join(app_output_dir,str(pred_out["kernel_id"]) + "_info.log"),'a+')
    
    # 析构时关闭文件
    def __del__(self):
        self.f.close()
    def write(self, *args, **kwargs):
        if not self.is_active:
            return
        # 使用 args 和 kwargs 构建格式化字符串
        output = ' '.join(map(str, args))
        if kwargs.get('end') is not None:
            del kwargs['end']  # 移除 end，避免在字符串格式化中出现

        if kwargs:
            output += ' ' + ' '.join(f'{k}: {v}' for k, v in kwargs.items())

        # 输出到文件
        print(output.strip(), end=kwargs.get('end', '\n'), file=self.f)

import os
import sys
import pandas as pd
import numpy as np
import PATH
import torch
import argparse
import seaborn as sns
# 
from models import reader
from models import train_val
from models.popen import Auto_popen
from models import max_activation_patch as MAP
# 
from sklearn.linear_model import Lasso, Ridge, ElasticNet, LassoCV, RidgeCV, ElasticNetCV
import warnings
from scipy.cluster import hierarchy
from matplotlib.backends.backend_pdf import PdfPages
warnings.filterwarnings('ignore')


#
parser = argparse.ArgumentParser('the script to evlauate the effect of ')
parser.add_argument("-c", "--config", type=str, required=True, help='the model config file: xxx.ini')
parser.add_argument("-s", "--set", type=int, default=2, help='train - 0 ,val - 1, test - 2 ')
parser.add_argument("-p", "--n_max_act", type=int, default=500, help='the number of seq')
parser.add_argument("-k", "--kfold_index", type=int, default=None,
                    help='k-fold index; omit to disable CV')
parser.add_argument("-d", "--device", type=str, default='cpu', help='the device to use to extract featmap, digit or cpu')
parser.add_argument("--csv", type=str, default="/home/xli263/xli/utr_design/DRAKES/data_and_model/utr_seq/utr_seq.csv", help='Override csv_path with a custom dataset')
args = parser.parse_args()

# args = parser.parse_args(["-c","/ssd/users/wergillius/Project/MTtrans/log/Backbone/RL_hard_share/3M/small_repective_filed_strides1113.ini",
#     "-s","2",
#     "-d", "1"])


def center_trim(seq, target_len):
    if len(seq) <= target_len:
        return seq
    offset = max(0, (len(seq) - target_len) // 2)
    return seq[offset:offset+target_len]


def extract_motif_records(map_task, metadata, task, group, seq_col, trim_len=7):
    records = []
    for seq_idx, feature_pos, delta in metadata:
        utr_seq = map_task.df.iloc[seq_idx][seq_col]
        start, end = map_task.retrieve_input_site(feature_pos)
        start = max(0, min(start, len(utr_seq)))
        end = max(start, min(end, len(utr_seq)))
        window = utr_seq[start:end]
        if len(window) == 0:
            continue
        records.append({
            "task": task,
            "group": group,
            "sequence_index": int(seq_idx),
            "feature_position": int(feature_pos),
            "start_nt": int(start),
            "end_nt": int(end),
            "motif_9nt": window,
            "motif_7nt": center_trim(window, trim_len),
            "delta": delta,
            "abs_delta": abs(delta)
        })
    return records

config_path = args.config
save_path = config_path.replace(".ini", "_coef")
config = Auto_popen(config_path)
config.batch_size = 256
config.shuffle = False
if args.kfold_index is None:
    config.kfold_cv = False
elif config.kfold_cv is False:
    config.kfold_cv = True
all_task = config.cycle_set

task_channel_effect = {}
task_performance = {}

# path check
if os.path.exists(config_path) and not os.path.exists(save_path):
    os.mkdir(save_path)

saved_pdf =os.path.join(save_path, 'changepoint_actmap.pdf')
pp = PdfPages(saved_pdf)
channel_cluster_task = {}
negative_motif_records = []
positive_motif_records = []

if args.csv:
    config.csv_path = args.csv
    config.cycle_set = ['MPA_H']
    config.split_like = None
    config.seq_col = 'seq'
    config.aux_task_columns = ['te']

for task in all_task:

    # .... format featmap as data ....
    print(f"\n\nevaluating for task: {task}")
    print(f"using csv: {config.csv_path}")
    # re-instance the map for each task
    map_task = MAP.Maximum_activation_patch(popen=config, which_layer=4,
                                      n_patch=args.n_max_act,
                                      kfold_index=args.kfold_index,
                                      device_string=args.device)
    print(f"device={map_task.popen.cuda_id}, total_stride={np.product(map_task.strides)}")
    # extract feature map and rl decision chain
    featmap = map_task.extract_feature_map(task=task, which_set=args.set)
    cum_rl_trend = map_task.cumulative_rl_decision(task=task, which_set=args.set)

    # truncate the featmap and rl trend according to sequence length
    seq_len_ls = map_task.df[config.seq_col].apply(len)
    total_stride = np.product(map_task.strides)
    to_stay = seq_len_ls//total_stride - 4
    trunc_start = featmap.shape[2] - to_stay

    # if max_seq_len == 50:
    #     trunc_start = -12


    # find the low rl sequences
    rl_pred = cum_rl_trend[:,-1]
    threshold = np.quantile(rl_pred, [0.05, 0.95])

    low_rl = rl_pred < threshold[0]
    high_rl = rl_pred > threshold[1]

    # clamp the detect region
    feat_len = featmap.shape[2]
    max_start = max(0, feat_len - 2)
    clamped_regions = [slice(max(0, min(max_start, int(start))), None) for start in trunc_start]
    sample_indices = np.arange(len(trunc_start))

    # subset find low rl change point feature
    lowrl_rl_chain = cum_rl_trend[low_rl]
    lowrl_ft = featmap[low_rl]
    low_indices = sample_indices[low_rl]
    lowrl_detect_region = [clamped_regions[idx] for idx in low_indices]
    changepoint_map, neg_metadata = map_task.retrieve_featmap_at_changepoint(
        lowrl_ft, lowrl_rl_chain,
        threshold=-1, direction='less',
        detect_region=lowrl_detect_region,
        source_indices=low_indices,
        return_metadata=True
    )
    print(changepoint_map.shape)
    negative_motif_records.extend(
        extract_motif_records(map_task, neg_metadata, task, 'negative', config.seq_col)
    )

    # sample high rl feature
    highrl_rl_chain = cum_rl_trend[high_rl]
    highrl_ft = featmap[high_rl]
    high_indices = sample_indices[high_rl]
    highrl_detect_region = [clamped_regions[idx] for idx in high_indices]
    background_map, pos_metadata = map_task.retrieve_featmap_at_changepoint(
        highrl_ft, highrl_rl_chain,
        threshold=0.5, direction='greater',
        detect_region=highrl_detect_region,
        source_indices=high_indices,
        return_metadata=True
    )
    positive_motif_records.extend(
        extract_motif_records(map_task, pos_metadata, task, 'positive', config.seq_col)
    )
    
    # and then subsample
    n_background = background_map.shape[0]
    n_foreground = changepoint_map.shape[0]
    if n_background > n_foreground:
        downsample_seed = np.random.choice(np.arange(0,n_background), size=n_foreground)
        background_map = background_map[downsample_seed]
        n_background = background_map.shape[0]
    
    # concate and order two group
    row_colors = ["#E09832"]*n_background + ["#192D48"]*n_foreground
    all_act=np.concatenate([background_map, changepoint_map], axis=0)
    feat_norm_act = (all_act - all_act.min(axis=0)) / (all_act.max(axis=0) - all_act.min(axis=0))


    # Visualization
    g=sns.clustermap(feat_norm_act, row_colors=row_colors);

    # 'array', 'axis', 'calculate_dendrogram', 'calculated_linkage', 'data', 
    # 'dendrogram', 'dependent_coord', 'independent_coord', 'label', 'linkage', 
    # 'method', 'metric', 'plot', 'reordered_ind', 'rotate', 'shape', 'xlabel', 
    # 'xticklabels', 'xticks', 'ylabel', 'yticklabels', 'yticks'

    
    Z_col=g.dendrogram_col.linkage
    thres = 0.8*max(Z_col[:,2])
    R = hierarchy.dendrogram(Z_col ,color_threshold=thres, truncate_mode=None,
        above_threshold_color='#AAAAAA', p=10, orientation='top',ax=g.ax_col_dendrogram);
    # R['leaves']
    # R['ivl']

    Z_row=g.dendrogram_row.linkage
    thres = 0.8*max(Z_row[:,2])
    R2 = hierarchy.dendrogram(Z_row ,color_threshold=thres, truncate_mode=None, orientation='left',
        above_threshold_color='#AAAAAA', p=10, ax=g.ax_row_dendrogram);
    g.ax_row_dendrogram.invert_yaxis()
    g.figure.suptitle(task)

    pp.savefig(g.figure)

    # channel_cluster_task[task] = pd.DataFrame(dict(zip(R['leaves'],R['leaves_color_list'])))
    channel_cluster_task[task] = pd.DataFrame({
    "leaf": R['leaves'],
    "color": R['leaves_color_list']
    })
    print("No..")
pp.close()
if negative_motif_records:
    negative_df = pd.DataFrame(negative_motif_records)
    negative_df.to_csv(os.path.join(save_path, "negative_motifs.csv"), index=False)
if positive_motif_records:
    positive_df = pd.DataFrame(positive_motif_records)
    positive_df.to_csv(os.path.join(save_path, "positive_motifs.csv"), index=False)
# channel_cluster_df = pd.DataFrame(channel_cluster_task)
channel_cluster_df = pd.concat(channel_cluster_task, axis=1)
saved_csv = os.path.join(save_path, "changepoint_channel_cluster.csv")
channel_cluster_df.to_csv(saved_csv, index=False)
print("==DONE==")
print(f"result save to {saved_pdf}\n \t\t {saved_csv}")

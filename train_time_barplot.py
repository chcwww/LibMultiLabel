"""
python train_time_barplot.py log_dir

Export normalized training time barplots of experiments.
"""
import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

prefix = "."
if len(sys.argv) > 1:
    prefix = sys.argv[1]
base_dir = os.path.join(prefix, 'para_log')

datasets = ['rcv1', 'EUR-Lex', 'Wiki10-31K', 'AmazonCat-13K']
linear_techs = ['1vsrest', 'tree']
branches = ['master', 'no_parallel', 'ovr_thread', 'sep_ovr_thread']

bname = ['liblr-multi', 'liblr-single', 'thread', 'no-prob-thread']
bname_map = {k: v for k, v in zip(branches, bname)}

full_time = pd.read_csv(os.path.join(base_dir, 'extract_time.csv')).set_index('path')

dataname = []
techs = []
branch = [] 
for i in full_time.index:
    d, t, b = i.split('--')
    dataname.append(d)
    techs.append(t)
    branch.append(b.split('.')[0])
full_time['dataname'] = dataname
full_time['techs'] = techs
full_time['branch'] = branch

for tech, df in full_time.groupby('techs'):
    key = " training_time" if tech == '1vsrest' else " training" # consider kmeans overhead or not
    
    pivot_df = df.pivot(index='dataname', columns='branch', values=key)
    pivot_df_normalized = pivot_df.div(pivot_df['master'], axis=0)
    df_normalized = pivot_df_normalized.reset_index().melt(id_vars='dataname', var_name='branch', value_name=key)
    df_normalized["dataname"] = pd.Categorical(df_normalized["dataname"], 
                                           categories=datasets, 
                                           ordered=True)
    
    ax = sns.barplot(x='dataname', y=key, hue='branch', data=df_normalized)
    
    cnt = 0
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f"{height:.2f}", (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom', fontsize=10, color='black')
        cnt += 1
        if cnt == 16:
            break
        
    plt.ylabel("Normalized Training Time (liblr-multi = 1)")
    plt.title(f"Normalized {tech} Training Time")
    handles, labels = plt.gca().get_legend_handles_labels()
    new_labels = [bname_map.get(label, label) for label in labels] 
    plt.legend(handles, new_labels, loc='center left', bbox_to_anchor=(1, 0.5), title="Branch")
    plt.tight_layout()
    plt.savefig(os.path.join(prefix, 'aggr_fig', tech, f'barplot_{tech}.png'), dpi=600)
    plt.close()

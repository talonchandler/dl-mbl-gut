import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


model_name = '3d_asym_avl_new_augs_noisecorrectsaveepoch3onward_model_epoch_13'
modeldir = f'/mnt/efs/dlmbl/G-bs/AvL/pred/{model_name}/'

lw = 3

#assemble the dataframe
df = pd.read_csv(modeldir+'predHD.csv', index_col = 0)
df = df[df.threshold==0.5].reset_index(drop=True)


fig ,ax = plt.subplots(1,1,figsize=(5, 6))
sns.set_palette('RdBu')
boxprops = dict(linewidth=lw)
whiskerprops = dict(linewidth=lw)
medianprops = dict(linewidth=lw)
capprops = dict(linewidth = lw)
sns.boxplot(data = df, x='dataset', y = 'HD', hue = 'dataset',ax = ax,
            boxprops=boxprops,
            whiskerprops=whiskerprops,
            medianprops=medianprops,
            capprops=capprops)
ax.set_xlabel('')
ax.set_ylabel('Hausdorff Distance', fontsize = 24)
ax.tick_params(axis='y', labelsize=14)
tick_positions = ax.get_xticks()
ax.set_xticks(tick_positions)
xlabelz = ['Test', 'Training','Validation']
ax.set_xticklabels(xlabelz)
ax.tick_params(axis='x', labelsize=18)
plt.tight_layout()
plt.savefig(modeldir+'HD_05.png')
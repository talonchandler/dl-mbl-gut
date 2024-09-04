import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


model_name = '3d_asym_avl_new_augs_noisecorrectsaveepoch3onward_model_epoch_13'
modeldir = f'/mnt/efs/dlmbl/G-bs/AvL/pred/{model_name}/'

#assemble the dataframe
df = pd.read_csv(modeldir+'predHD.csv', index_col = 0)
print(df.HD)
fig ,ax = plt.subplots(1,1,figsize=(10, 6))

sns.set_palette('Accent')
sns.boxplot(data = df, x='dataset', y = 'HD', hue = 'threshold', ax = ax)
# ax.set_xlabel(ax.get_xlabel(), fontsize = 36)
# ax.set_ylabel(ax.get_ylabel(), fontsize = 36)
# ax.set_xticks([1,2,3,4,5])
# ax.set_xticklabels(ax.get_xticklabels(), fontsize = 17)
# ax.set_yticklabels(ax.get_yticklabels(), fontsize = 17)
plt.tight_layout()
plt.savefig(modeldir+'HD scan.png')
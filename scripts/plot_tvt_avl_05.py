import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



model_name = '3d_asym_avl_new_augs_noisecorrectsaveepoch3onward_model_epoch_13'
modeldir = f'/mnt/efs/dlmbl/G-bs/AvL/pred/{model_name}/'
#get all of the prediction folders
foldlist = [x for x in os.listdir(modeldir) if '.' not in x]


lw = 3

#assemble the dataframe
dflist = []
for f in foldlist:
    dflist.append(pd.read_csv(modeldir+f+'/'+'prediction_metrics.csv', index_col = 0))
df = pd.concat(dflist).reset_index(drop = True)
df = df[df.dice_threshold == 0.5]


#### plot the plot
fig ,ax = plt.subplots(1,1,figsize=(5, 6))
sns.set_palette('Accent')
boxprops = dict(linewidth=lw)
whiskerprops = dict(linewidth=lw)
medianprops = dict(linewidth=lw)
capprops = dict(linewidth = lw)
sns.boxplot(data = df, x='dataset', y = 'dice_value', hue = 'dataset',
            boxprops=boxprops,
            whiskerprops=whiskerprops,
            medianprops=medianprops,
            capprops=capprops)
ax.set_xlabel('')
ax.set_ylabel('Dice Coefficient', fontsize = 24)
ax.tick_params(axis='y', labelsize=14)
tick_positions = ax.get_xticks()
ax.set_xticks(tick_positions)
xlabelz = ['Test', 'Validation','Training']
ax.set_xticklabels(xlabelz)
ax.tick_params(axis='x', labelsize=18)



plt.tight_layout()
plt.savefig(modeldir+'dice validation 05.png')
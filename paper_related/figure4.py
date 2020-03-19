import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

layer = 13      # 3, 8 or 13
files = {"tinyimagenet.txt": "Unconstrained",
         "tinyimagenet_channelwise_norm.txt": "Channel-wise unit norm",
         "tinyimagenet_layerwise_norm.txt": "Layer-wise unit norm",
         "tinyimagenet_layerwise_norm_nomomentum.txt": "Layer-wise unit norm, no momentum",
         }

# Setup
fig, ax = plt.subplots()

for file in files:
    f = open(file, "r")
    cond = []
    layers = {3: 1, 8: 2, 13: 3}

    for x in f:
        x = x.split()

        if (x[0] == 'Epoch:'):
            i = x.index('cond')
            l = layers[layer]
            cond.append(float(x[i+l]))

    # Option 1
    # sns.set()
    # sns.set_style("white")
    # sns.despine(bottom=False, top=True, right=True, left=True, trim=False)
    # sns_plot = sns.kdeplot(np.asarray(cond), shade=True, label=file, legend=True)
    # sns_plot.set(yticklabels=[])

    # Option 2
    for i in range(1, len(cond) - 1):
        # Moving average
        cond[i] = (cond[i - 1] + cond[i] + cond[i + 1]) / 3
    ax.plot(cond)

ax.legend(files.values())
ax.set_ylabel('Conditioning number', fontsize=14)
ax.set_xlabel('Epochs', fontsize=16)
ax.set_title('Layer ' + str(layer))

fig.savefig('conditioning_layer' + str(layer) + '.pdf', format='pdf')
plt.show()


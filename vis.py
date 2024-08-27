import scipy.sparse as sparse  # for sparse weight matrix
import seaborn as sns
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

class ErrorVis:    
    @staticmethod 
    def vis(errordf, filename, low=0.3, high=0.6, title=None):
        sns.set_context(rc={'font.size': 26, 'axes.titlesize': 26, 'axes.labelsize': 26})
        fig, ax = plt.subplots(1, 1, figsize=(1.75 * 4, 10))
        my_pal = {'stest': 'blue', 
                  'strain': 'darkorange', 
                  'rtest': 'teal', 
                  'rtrain': 'blue',
                  'sstest': 'gold', 
                  'standard': 'blue', 
                  '2 subres.': 'darkorange', 
                  '4 subres.': 'gold',
                  '8 subres.': 'yellow',
                  '1': 'blue', 
                  '2': 'darkorange', 
                  '4': 'gold',
                  '8': 'yellow',
                  'restricted': 'darkorange',
                  'ctrain': 'white',
                  'gtrain': 'white',
                  'ctest': 'gold',
                  'gtest': 'lightgreen'}

        sns.boxplot(data=errordf, ax=ax, notch=True, width=0.6, linewidth=1.5, fliersize=3, palette=my_pal)
        # ax.set_ylabel('NRMSE')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_ylim(low, high)
        fig.suptitle(title)
        plt.close('all')
        # plt.show()
        fig.savefig(filename, format="png")
#!/bin/python
# Copyright 2024 Yogesh Kalakoti (WallnerLab), Linkoping University, Sweden
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

## Import necessary dataframes and define plotting functions
import glob, os, re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import wilcoxon
from scipy import integrate
import collections
from sklearn.metrics import mean_absolute_error as mae
from tqdm import tqdm
import warnings
import pickle

warnings.filterwarnings("ignore")

class BestModelsPlotter:
    """
    Processes a DataFrame with the following expected structure:
    
    Columns:
    - uniprotid (str): Unique protein identifier
    - AFvanilla (float)
    - MSAsubsample (float)
    - AFcluster (float)
    - SPEACH_AF (float)
    - AFsample (float)
    - AFsample2 (float)

    Example row:
    A2RJ53, 0.87205, 0.87878, 0.68717, 0.93350, 0.87654, 0.93906

    :param df: pandas DataFrame with the structure described above
    :return: Processed DataFrame
    """

    def __init__(self, best_tms_o_path, best_tms_c_path, dataset):
        self.best_tms_o = pd.read_csv(best_tms_o_path, index_col=0, compression='gzip')
        self.best_tms_c = pd.read_csv(best_tms_c_path, index_col=0, compression='gzip')
        self.colors = sns.color_palette()
        self.dataset = dataset
        if self.dataset=='OC23':
            self.s1, self.s2 = 'open', 'closed'
        elif self.dataset=='TP16':
            self.s1, self.s2 = 'inward', 'outward'

    def plot_bestmodels_scatter(self):
        fig, axes = plt.subplots(1,6, figsize=(11,2.2), sharex=True, sharey=True)
        p = sns.color_palette()
        colors = [p[0], p[2], p[3], p[4], p[6], p[1]]

        methods = self.best_tms_o.columns[1:]

        best_tms_merged = self.best_tms_o.merge(self.best_tms_c, on='uniprotid', suffixes=('_o', '_c'))

        for method, ax, color in zip(methods, axes, colors):
            df = best_tms_merged[[f'{method}_o', f'{method}_c']]
            selected_df = df[(df[f'{method}_o']>0.8) & (df[f'{method}_c']>0.8)]
            ax.scatter(selected_df[f'{method}_o'], selected_df[f'{method}_c'], c=[color]*len(selected_df), edgecolor='black', linewidth=0.2)

            selected_df = df[(df[f'{method}_o']<0.8) | (df[f'{method}_c']<0.8)]
            ax.scatter(selected_df[f'{method}_o'], selected_df[f'{method}_c'], color=['lightgray']*len(selected_df), edgecolor='black', linewidth=0.2)
            ax.set_title(method)

        for ax in axes:
            ax.set_xlabel(f'Best {self.s1}')
            ax.set_ylabel(f'Best {self.s2}')
            ax.axvline(x=0.8, linewidth=1, linestyle='-.', c='gray')
            ax.axhline(y=0.8, linewidth=1, linestyle='-.', c='gray')
            
            ax.set_xlim(0.5, 1)
            ax.set_ylim(0.5, 1)
        plt.tight_layout()
        plt.show()
        return None

    def plot_bestmodels_boxplot(self):
        p = sns.color_palette()
        colors = [p[0], p[2], p[3], p[4], p[6], p[1]]

        sigs_o = []
        for i, method in enumerate(self.best_tms_o.columns[1:-1]):
            p = wilcoxon(self.best_tms_o['AFsample2'].astype(float)-self.best_tms_o[method].astype(float)).pvalue
            sigs_o.append((i, 5, round(p, 5), method))

        sigs_c = []
        for i, method in enumerate(self.best_tms_o.columns[1:-1]): 
            p = wilcoxon(self.best_tms_c['AFsample2'].astype(float)-self.best_tms_c[method].astype(float)).pvalue
            sigs_c.append((i, 5, round(p, 5),method))

        sigs = [sigs_o]+[sigs_c]

        fig, ax = plt.subplots(1, 2, figsize=(6.5, 4))
        #colors = sns.color_palette('Paired', n_colors=20)[-9:]
        cols = self.best_tms_o.columns[1:]

        # OPEN
        # Explicitly map positions for both boxplot and stripplot
        positions = range(len(cols))  # 0-based for consistency
        # Boxplot: Pass both positions and data
        box_o = ax[0].boxplot([self.best_tms_o[col] for col in cols], patch_artist=True, showfliers=False,
                            positions=positions,  # Use the same 0-based positions
                            meanprops={'marker': '_', 'markerfacecolor': 'black', 'markeredgecolor': 'black'}, boxprops=dict(alpha=.7))
        # Stripplot: Pass consistent positions
        for i, col in enumerate(cols):
            sns.stripplot(x=np.full(len(self.best_tms_o), i), y=self.best_tms_o[col], color=colors[i], 
                        alpha=0.7, size=3, jitter=True, ax=ax[0])
        for patch, color in zip(box_o['boxes'], colors):
            patch.set_facecolor('white')
            patch.set_edgecolor('gray')
        for median in box_o['medians']:
            median.set(color='black', linewidth=1.5)


        # CLOSED
        # Explicitly map positions for both boxplot and stripplot
        positions = range(len(cols))  # 0-based for consistency
        # Boxplot: Pass both positions and data
        box_c = ax[1].boxplot([self.best_tms_c[col] for col in cols], patch_artist=True, showfliers=False,
                            positions=positions,  # Use the same 0-based positions
                            meanprops={'marker': '_', 'markerfacecolor': 'black', 'markeredgecolor': 'black'}, boxprops=dict(alpha=.7))
        # Stripplot: Pass consistent positions
        for i, col in enumerate(cols):
            sns.stripplot(x=np.full(len(self.best_tms_c), i), y=self.best_tms_c[col], color=colors[i], 
                        alpha=0.7, size=3, jitter=True, ax=ax[1])
        for patch, color in zip(box_c['boxes'], colors):
            patch.set_facecolor('white')
            patch.set_edgecolor('gray')
        for median in box_c['medians']:
            median.set(color='black', linewidth=1.5)

        # Set labels
        ax[0].set_xlabel('Methods')
        ax[1].set_xlabel('Methods')
        ax[0].set_ylabel(f'TM-score to {self.s1}')
        ax[1].set_ylabel(f'TM-score to {self.s2}')
        ax[0].grid(linestyle='-.')
        ax[1].grid(linestyle='-.')
        ax[0].set_title(f'Best models ({self.s1} state)')
        ax[1].set_title(f'Best models ({self.s2} state)')
        ax[0].axhline(y=1, linewidth=1, c='black')
        ax[1].axhline(y=1, linewidth=1, c='black')

        # Set xticks and xticklabels
        ax[0].set_xticks(range(0, 6))
        ax[0].set_xticklabels(self.best_tms_o.columns[1:], rotation=45, rotation_mode='anchor', ha='right')
        ax[1].set_xticks(range(0, 6))
        ax[1].set_xticklabels(self.best_tms_o.columns[1:], rotation=45, rotation_mode='anchor', ha='right')

        for x, sig in zip(ax, sigs):
            bottom, top = 0.08, 1
            y_range = top - bottom

            for i, a in enumerate(sig):
                x1, x2, p, _ = a

                if p < 0.001:
                    sig_symbol = '***'
                elif p < 0.01:
                    sig_symbol = '**'
                elif p < 0.05:
                    sig_symbol = '*'
                else:
                    sig_symbol = '-'

                level = len(sig) - i
                bar_height = (y_range * 0.08 * level) + top - 0.03
                bar_tips = bar_height - (y_range * 0.02)
                if sig_symbol=='-':
                    lincolor='gray'
                else:
                    lincolor='k'
                
                x.plot(
                    [x1, x1, 5, 5],
                    [bar_tips, bar_height, bar_height, bar_tips], lw=1, c=lincolor
                )

                text_height = bar_height - (y_range * 0.02)
                x.text((x1 + x2) * 0.50, text_height, sig_symbol, ha='center', va='bottom', c=lincolor)

        plt.tight_layout()
        plt.show()
        return None

    # Min-max scaling function
    def min_max_scaling(self, numbers):
        min_val = min(numbers)
        max_val = max(numbers)
        scaled_numbers = [(x - min_val) / (max_val - min_val) for x in numbers]
        return scaled_numbers

    def plot_auc(self):
        p = sns.color_palette()
        colors = [p[0], p[2], p[3], p[4], p[6], p[1]]

        fig, ax = plt.subplots(1, figsize=(4, 4))
        for mode, color in zip(self.best_tms_o.columns[1:], colors):
            roc = []
            for threshold in np.arange(0.01, 1.01, 0.01):
                roc.append(self.best_tms_o[(self.best_tms_o[f'{mode}'] > threshold) & 
                                    (self.best_tms_c[f'{mode}'] > threshold)].shape[0])
            
            ax.plot(np.arange(0.01, 1.01, 0.01), 
                    self.min_max_scaling(roc), 
                    label=f'{mode} (AUC: {np.round(integrate.trapezoid(self.min_max_scaling(roc), np.arange(0.01, 1.01, 0.01)), 2)})',
                    color=color)

        ax.set_ylabel('Fraction of successful targets')
        ax.set_xlabel('TM-score threshold')
        ax.grid(color='gray', linestyle='--', linewidth=1)
        ax.legend(fontsize=9, title='AUC')  # Improve legend readability
        ax.set_xlim(0,1)
        ax.set_ylim(0,1.01)
        ax.set_title(self.dataset) 

        plt.show()

    def plot_scatter_comparision(self):
        fig, axes = plt.subplots(1,5, figsize=(9,2.3), sharex=True, sharey=True)
        i=0
        tolerance = 0.05
        for method1 in self.best_tms_o.columns[1:-1]:
            for method2 in ['AFsample2']:
                mins_method1 = np.minimum(self.best_tms_o[f'{method1}'].values, self.best_tms_c[f'{method1}'].values)
                mins_method2 = np.minimum(self.best_tms_o[f'{method2}'].values, self.best_tms_c[f'{method2}'].values)

                # Grey out points within the diagonal band
                colors = [
                    'gray' if abs(x - y) <= tolerance else ('red' if x > y else 'green') 
                    for x, y in zip(mins_method1, mins_method2)
                ]

                #colors = ['red' if x > y else 'green' for x, y in zip(mins_method1, mins_method2)]
                axes.flatten()[i].scatter(mins_method1, mins_method2, c=colors, s=20, alpha=0.5)

                axes.flatten()[i].set_xlabel(method1)
                axes.flatten()[i].set_title(f'{collections.Counter(colors)["green"]}/{len(mins_method1)} improved')
                # axes.flatten()[i].set_title(f'{collections.Counter(mins_method1<mins_method2)[True]}/{len(mins_method1)} improved')
                #print(method2, imp_open, imp_close)

                # Add tolerance lines
                x_vals = np.linspace(0.4, 1, 100)
                axes.flatten()[i].plot(x_vals, x_vals + tolerance, linestyle='-', color='gray', alpha=0.7, linewidth=1, label=f'+{tolerance}')
                axes.flatten()[i].plot(x_vals, x_vals - tolerance, linestyle='-', color='gray', alpha=0.7, linewidth=1, label=f'-{tolerance}')
                
                pval = wilcoxon(mins_method2, mins_method1, alternative='greater').pvalue
                print(method1, method2, f"{pval:.2e}")
                axes.flatten()[i].text(0.65,0.43,f"p<{pval:.1e}")

                i+=1
                # print(method1, method2, wilcoxon(mins_method2, mins_method1, alternative='greater'))
                # print(method1, method2, wilcoxon(mins_method2, mins_method1))
                # print()
                        
        axes.flatten()[0].set_ylabel(method2)
        for x in axes.flatten():
            x.set_xlim(0.4,1)
            x.set_ylim(0.4,1)	
            x.axline((0, 0), slope=1, linestyle=':', color='black')

        plt.tight_layout()
        return None

    def plot_bar_comparision(self):
        cs = ['blue', 'green']
        fig, ax = plt.subplots(2,1, figsize=(5,3.2), sharey=True, sharex=True)
        d = self.best_tms_o['AFsample2']-self.best_tms_o['AFvanilla']
        self.best_tms_o[f'diff_{self.s1}'] = d
        self.best_tms_o = self.best_tms_o.sort_values(by=f'diff_{self.s1}', ascending=False)

        diff_s1 = self.best_tms_o[f'diff_{self.s1}']
        xlabels = np.array([*range(len(diff_s1))])
        mask1 = diff_s1 < 0
        mask2 = diff_s1 >= 0
        ax[0].bar(xlabels[mask1], diff_s1[mask1], color=cs[0])
        ax[0].bar(xlabels[mask2], diff_s1[mask2], color=cs[1])
        #ax[0].axvline(x=11.5, linestyle='-.', color='blue', linewidth=1)
        #ax[0].axvspan(-0.5, 11.5, alpha=0.1, color='red')
        ax[0].set_xticks(range(len(self.best_tms_o)))
        ax[0].set_xticklabels(self.best_tms_o['uniprotid'], rotation=90)
        ax[0].grid(color='gray', linestyle='--', linewidth=0.3)
        ax[0].set_title(f'{self.s1} conformation')

        diff_s2= self.best_tms_c['AFsample2']-self.best_tms_c['AFvanilla']
        self.best_tms_c[f'diff_{self.s1}'] = d
        self.best_tms_c[f'diff_{self.s2}'] = diff_s2
        self.best_tms_c = self.best_tms_c.sort_values(by=f'diff_{self.s1}')
        diff_s2 = self.best_tms_c[f'diff_{self.s2}']

        xlabels = np.array([*range(len(diff_s2))])
        mask1 = diff_s2 < 0
        mask2 = diff_s2 >= 0
        ax[1].bar(xlabels[mask1], diff_s2[mask1], color=cs[0])
        ax[1].bar(xlabels[mask2], diff_s2[mask2], color=cs[1])
        ax[1].set_ylabel(r'$\Delta$Best TMscore' +'\n'+'AFsample2-AFVanilla')
        ax[1].set_xticks(range(len(self.best_tms_o)))
        ax[1].set_xticklabels(self.best_tms_o['uniprotid'], rotation=45, rotation_mode='anchor', ha='right', fontsize=8)
        ax[1].grid(color='gray', linestyle='-.', linewidth=0.3)
        ax[1].set_title('Closed conformation')

        plt.tight_layout()
        return None
    
    def main(self):
        # Figure 3a
        self.plot_bestmodels_scatter()
        # Figure 3b
        self.plot_bestmodels_boxplot()
        # Figure 3c
        self.plot_auc()
        # Figure 3d
        self.plot_scatter_comparision()
        # Figure 3e
        self.plot_bar_comparision()

class ProteinDataAnalyzer:
    '''
    Expected DataFrame structure:
    Columns:
    - 'model_path' (str): File path to the model results
    - 'confidence' (float): Confidence score of the model
    - 'model_name' (str): Model name identifier (e.g., "model_1_ptm")
    - 'model_pdb' (str): Filename of the predicted PDB structure
    - 'state' (str): Model state (e.g., "TM_open")
    - 'tmscore' (float): TM-score for model quality assessment
    - 'uniprotid' (str): Unique protein identifier
    - 'rand' (str): Randomization category (e.g., "vanilla")

    Example:

        model_path                                  | confidence | model_name  | model_pdb                                     | state   | tmscore | uniprotid | rand  
        --------------------------------------------------------------------------------------------------------------------------
        data/result_model_1_ptm_pred_80_vanilla.pkl | 89.91       | model_1_ptm  | unrelaxed_model_1_ptm_pred_80_dropout.pdb | TM_open | 0.53115 | A2RJ53   | vanilla
        data/result_model_1_pred_9_vanilla.pkl      | 90.85       | model_1      | unrelaxed_model_1_pred_9_dropout.pdb      | TM_open | 0.57069 | A2RJ53   | 15
        data/result_model_2_ptm_pred_53_vanilla.pkl | 89.64       | model_2_ptm  | unrelaxed_model_2_ptm_pred_53_dropout.pdb | TM_open | 0.56201 | A2RJ53   | 30

    '''

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.rands = ['vanilla', '05', '10', '15', '20', '25', '30', '35', '40', '50']
        self.master_df = self.load_data()
        self.processed_dfs = self.process_data()
    
    def load_data(self) -> pd.DataFrame:
        """Loads the master CSV file."""
        return pd.read_csv(self.file_path, dtype={"rand": str}, compression='gzip')
    
    def process_data(self):
        """Processes data into required formats for plotting."""
        df1, df2, df3, df5 = [pd.DataFrame(columns=self.rands, index=self.master_df['uniprotid'].unique()) for _ in range(4)]
        
        gk_model = self.master_df.groupby(by='rand')
        for abl in self.rands:
            subset = gk_model.get_group(abl).groupby(by='uniprotid')
            for protein in subset.groups:
                protein_subset = subset.get_group(protein)
                
                df1.at[protein, abl] = protein_subset[protein_subset['state'] == 'TM_open']['tmscore'].max()
                df2.at[protein, abl] = protein_subset[protein_subset['state'] == 'TM_close']['tmscore'].max()
                df3.at[protein, abl] = protein_subset[protein_subset['state'] == 'TM_open']['tmscore'].mean()
                df5.at[protein, abl] = protein_subset['confidence'].mean()
        
        return {
            "best_tm_open": df1.apply(pd.to_numeric, errors='coerce'),
            "best_tm_close": df2.apply(pd.to_numeric, errors='coerce'),
            "mean_tm_open": df3.apply(pd.to_numeric, errors='coerce'),
            "confidence": df5.apply(pd.to_numeric, errors='coerce')
        }
    
    def plot_tm_scores(self):
        """Generates strip plots for TM scores."""
        df1, df2, df5 = self.processed_dfs['best_tm_open'], self.processed_dfs['best_tm_close'], self.processed_dfs['confidence']
        cp = sns.color_palette()
        fig, ax = plt.subplots(1, 2, figsize=(7, 3.5))
        
        for df, color, label, i in zip([df1, df2], [cp[0], cp[1]], ['Best open', 'Best close'], [0, 0]):
            sns.stripplot(df, color=color, alpha=0.2, size=3, jitter=True, ax=ax[i], zorder=4)
            ax[i].errorbar(self.rands, df.mean(axis=0), yerr=df.std(axis=0)/np.sqrt(23), linestyle='-', alpha=0.8, zorder=2, color=color, capsize=2, label=label)
            ax[i].scatter(self.rands, df.mean(axis=0), marker='o', edgecolor='black', alpha=0.8, s=60, color=color, zorder=5)
        
        ax[0].set_xlabel('MSA randomization (%)')
        ax[0].set_ylabel('TM-score of best model\n(Averaged over 23 proteins)')
        ax[0].grid(color='lightgray', linestyle='--', linewidth=0.3, zorder=-1)
        ax[0].set_title('Best TM-scores')
        
        ax[1].errorbar(self.rands, df5.mean(axis=0), yerr=df5.std(axis=0)/np.sqrt(23), alpha=0.8, label='Aggregate (1000 models)', color='black')
        sns.stripplot(df5, alpha=0.2, size=3, jitter=True, ax=ax[1], color='black', zorder=4)
        ax[1].scatter(self.rands, df5.mean(axis=0), marker='o', edgecolor='black', alpha=1, color='white', zorder=5)
        ax[1].set_xlabel('MSA randomization (%)')
        ax[1].set_ylabel('Model confidence')
        ax[1].grid(color='lightgray', linestyle='--', linewidth=0.3, zorder=-1)
        ax[1].set_title('Model confidence')
        
        for x in ax:
            x.spines['right'].set_visible(False)
            x.spines['top'].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def plot_heatmaps(self):
        """Generates heatmaps for best TM scores."""
        df1, df2 = self.processed_dfs['best_tm_open'], self.processed_dfs['best_tm_close']
        fig, ax = plt.subplots(1, 2, sharey=True)
        cmap, linewidths, linecolor, vmin = 'jet_r', 0.7, 'white', 0.5
        sns.heatmap(df1, cmap=cmap, square=True, linewidths=linewidths, ax=ax[0], linecolor=linecolor, vmin=vmin)
        ax[0].set_title('Best Open Models', fontsize=10)
        sns.heatmap(df2, cmap=cmap, square=True, linewidths=linewidths, ax=ax[1], linecolor=linecolor, vmin=vmin)
        ax[1].set_title('Best Closed Models', fontsize=10)
        plt.tight_layout()
        plt.show()
    
    def run_sampling_experiment(self, iterations=100):
        rands = ['vanilla', '05', '10', '15', '20', '25', '30', '35', '40', '50']
        sampling_o, sampling_c = [], []
        gk_model = self.master_df.groupby(by='rand')
        print(rands)
        for abl in tqdm(rands):
            subset = gk_model.get_group(abl)
            gk_subset = subset.groupby(by='uniprotid')
            tmbest_means_o, tmbest_means_c = [], []
            for protein in subset['uniprotid'].unique():
                protein_subset = gk_subset.get_group(protein)
                tm_open = protein_subset[protein_subset['state']=='TM_open']
                tm_close = protein_subset[protein_subset['state']=='TM_close']
                maxes_o, maxes_c = [], []
                
                for i in np.arange(50,1001,50):
                    m_o, m_c = [], []
                    for iter in range(iterations):
                        if i>len(tm_open):
                            i=len(tm_open)
                        if i>len(tm_close):
                            i=len(tm_close)
                        m_o.append(tm_open.sample(n=i)['tmscore'].max())
                        m_c.append(tm_close.sample(n=i)['tmscore'].max())
                    maxes_o.append(np.mean(m_o))
                    maxes_c.append(np.mean(m_c))
            
                tmbest_means_o.append(maxes_o)
                tmbest_means_c.append(maxes_c)
            sampling_o.append(tmbest_means_o)
            sampling_c.append(tmbest_means_c)  

        sampling_o = np.array(sampling_o)
        sampling_c = np.array(sampling_c)
        return sampling_o, sampling_c
    
    def plot_sampling_exp(self, iterations):
        sampling_o, sampling_c = self.run_sampling_experiment(iterations)
        rands = ['vanilla', '05', '10', '15', '20', '25', '30', '35', '40', '50']
        #rands == ['vanilla', 'dropout', '20']
        fig, ax = plt.subplots(1,2, figsize=(6,3), sharey=True)
        for sampling, a, x in zip([np.array(sampling_o), np.array(sampling_c)], ['open', 'close'], ax):
            for rand , sample in zip(rands, sampling):
                if rand in ['vanilla', 'dropout']:
                    marker = 'x'
                    linestyle='-.'
                    linewidth=2
                    label='00'
                else:
                    marker = '.'
                    linestyle='-'
                    linewidth=1
                    label=rand

                sample = np.array(sample).mean(axis=0)
                x.plot(np.arange(50,1001,50), sample, label=label, marker=marker, linewidth=linewidth, linestyle=linestyle)
                x.set_xticks(np.arange(50,1001,50))
                x.set_xticklabels(np.arange(50,1001,50), rotation=90, fontsize=8)
                x.grid(color='gray', linestyle='-.', linewidth=0.04)
                x.set_xlabel('Number of samples')
                x.set_ylabel(f'TM score ({a} conformation)')
                x.grid(color='gray', linestyle='-.', linewidth=0.01)

        handles, labels = x.get_legend_handles_labels()
        fig.legend(handles, labels, bbox_to_anchor=(1.09, 0.9), fontsize=8, title='%')
        plt.tight_layout()
        plt.show()
    
    def plot_intermediates(self, intermediatespath):
        pdbhits_df = pd.read_csv(intermediatespath, compression='gzip')
        pdbhits_df_f = pdbhits_df[pdbhits_df['tmscore_to_model']>0.8]

        afs2_tmdf = self.master_df
        grp_afs2_tmdf = afs2_tmdf.groupby("uniprotid")

        fig, axes = plt.subplots(2, 4, figsize=(8.5, 4.5), sharex=True, sharey=True, constrained_layout=True)
        vmin = pdbhits_df_f.tmscore_to_model.min()
        vmax = pdbhits_df_f.tmscore_to_model.max()

        for uniprotid, ax in zip(pdbhits_df.protein.unique(), axes.flatten()):
            tm_df = grp_afs2_tmdf.get_group(uniprotid)
            tm_df = tm_df[tm_df['rand'] == '15']
            ax.scatter(tm_df[tm_df['state'] == 'TM_open'].tmscore, 
                    tm_df[tm_df['state'] == 'TM_close'].tmscore, 
                    color='lightgray', s=6, zorder=2)
            pdbhits_df_ = pdbhits_df[pdbhits_df['protein'] == uniprotid]
            c = ax.scatter(pdbhits_df_.max_row_tmo, 
                        pdbhits_df_.max_row_tmc, 
                        c=pdbhits_df_.tmscore_to_model, 
                        vmin=vmin, vmax=vmax, cmap='viridis_r', s=15, zorder=3)
            ax.set_title(uniprotid)

        for ax in axes[1]:
            ax.set_xlabel('TMscore (to open)')
            ax.text(
                    0.28, 
                    0.08, 
                    'No exp.\n intermediates', 
                    color="black", 
                    fontsize=8, 
                    rotation=0, 
                    ha="center", 
                    va="center",
                    transform=ax.transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7)
                    )

        for ax in axes[0]:
            ax.text(
                    0.28, 
                    0.08, 
                    'Mapped exp.\n intermediates', 
                    color="black", 
                    fontsize=8, 
                    rotation=0, 
                    ha="center", 
                    va="center",
                    transform=ax.transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7)
                    )
            
        for ax in axes[:, 0]:
            ax.set_ylabel('TMscore (to closed)')
        for ax in axes.flatten():
            ax.grid(linestyle='--', zorder=1)  # Lower zorder for the gridlines

        cbar = fig.colorbar(c, ax=axes, orientation='vertical', shrink=1, aspect=20)
        cbar.set_label('TMscore of model to PDB')
        plt.show()
        return None
        
    def analyze(self):
        """Runs the full analysis pipeline."""
        self.plot_tm_scores()
        self.plot_heatmaps()
        self.plot_sampling_exp()

class ClusterReader:
    """Handles reading cluster files and extracting relevant data."""
    @staticmethod
    def read_clusterfile(clusterfile: str) -> pd.DataFrame:
        data = []
        with open(clusterfile) as f:
            for line in f:
                match = re.match(r'protocols.cluster: \(0\)\s+\d+\s+(\S+)\s+(\d+)\s+\d+', line)
                if match:
                    data.append({"model": match.group(1), "cluster": int(match.group(2))})
        return pd.DataFrame(data)

class SampleClassifier:
    """Classifies two samples into states based on TM scores."""
    @staticmethod
    def classify(state1, state2):
        tm_s1_o, tm_s1_c, _ = state1
        tm_s2_o, tm_s2_c, _ = state2
        return (1, 2) if tm_s1_o + tm_s2_c < tm_s1_c + tm_s2_o else (2, 1)

class ReferenceFreeSelection:
    """Performs reference-free selection of best models based on confidence thresholds."""
    def __init__(self, good_examples: dict):
        self.good_examples = good_examples
    
    def run(self, rand_dfs):
        maes_o, maes_c = [], []
        best_result, lowest_error, num_models_mean = None, float('inf'), []
        
        for conf_threshold in np.arange(0, 101, 5):
            results = pd.DataFrame(columns=['uniprotid', 'sel_tm_o', 'sel_tm_c', 'best_tm_o', 'best_tm_c'])
            num_models = []
            
            for i, uniprotid in enumerate(self.good_examples.keys()):      
                rand_df = rand_dfs[rand_dfs['uniprotid'] == uniprotid]      
                max_confidence = rand_df['confidence'].max()
                conf_filtered = rand_df[rand_df['confidence'] >= max_confidence * (conf_threshold * 0.01)]
                num_models.append(len(conf_filtered))
                
                if conf_filtered.empty:
                    results.loc[i] = [uniprotid, 0, 0, rand_df['TM_open'].max(), rand_df['TM_close'].max()]
                    continue
                
                idx1, idx2 = conf_filtered['tm_bestAF'].idxmax(), conf_filtered['tm_bestAF'].idxmin()
                state1, state2 = conf_filtered.loc[idx1], conf_filtered.loc[idx2]
                b1, b2 = state1[['TM_open', 'TM_close', 'tm_bestAF']].values.flatten(), state2[['TM_open', 'TM_close', 'tm_bestAF']].values.flatten()
                state_check = SampleClassifier.classify(b1, b2)
                
                sel_tm_o, sel_tm_c = (b2[0], b1[1]) if state_check == (1, 2) else (b1[0], b2[1])
                best_tm_o, best_tm_c = rand_df['TM_open'].max(), rand_df['TM_close'].max()
                results.loc[i] = [uniprotid, sel_tm_o, sel_tm_c, best_tm_o, best_tm_c]
            
            mae_o, mae_c = mae(results['best_tm_o'], results['sel_tm_o']), mae(results['best_tm_c'], results['sel_tm_c'])
            maes_o.append(mae_o)
            maes_c.append(mae_c)
            num_models_mean.append(np.mean(num_models))
            
            if mae_o + mae_c < lowest_error:
                best_result = results
                lowest_error = mae_o + mae_c
                print(f'Best conf_threshold: {max_confidence * (conf_threshold * 0.01)}, {conf_threshold}, losses: {lowest_error}')
        
        return best_result, self.rolling_average(maes_o), self.rolling_average(maes_c)
    
    @staticmethod
    def rolling_average(errors):
        return [(errors[i - 1] + errors[i + 1] + errors[i]) / 3 for i in range(1, len(errors) - 1)]

    @staticmethod
    def plot_results(best_result, maes_o_rolling, maes_c_rolling, title):
        fig, ax = plt.subplots(1, 3, figsize=(8.5, 3.5))
        
        ax[0].plot(range(len(maes_o_rolling)), maes_o_rolling, label='Open', c='red', marker='o', markersize=4)
        ax[0].plot(range(len(maes_c_rolling)), maes_c_rolling, label='Closed', marker='o', markersize=4)
        ax[0].set_xticks(np.arange(0, 20, step=2))
        ax[0].set_xticklabels([5, 15, 25, 35, 45, 55, 65, 75, 85, 95], rotation=90)
        ax[0].set_title('MAE')
        ax[0].set_xlabel('Confidence threshold (% of highest confidence)')
        ax[0].set_ylabel('MAE (TMscore): Selected & best')
        ax[0].legend()
        
        for j, label, color, best_tm, sel_tm in zip([1, 2], ['Open', 'Closed'], ['red', 'blue'], ['best_tm_o', 'best_tm_c'], ['sel_tm_o', 'sel_tm_c']):
            ax[j].scatter(best_result[best_tm], best_result[sel_tm], edgecolor='black', linewidth=0.1, c=color)
            ax[j].set_title(f'{label}\nConfidence=85% of max')
            ax[j].set_xlabel('TMscore (Best model)')
            ax[j].set_ylabel('TMscore (Selected model)')
            ax[j].axline((0, 0), slope=1, color='black', linestyle=':')
            ax[j].grid(linestyle='-.', linewidth=0.5)
            ax[j].set_ylim(0, 1)
            ax[j].set_xlim(0, 1)
        
        plt.suptitle(title, fontweight='bold')
        plt.tight_layout()
        plt.show()

# Other functions

def plot_fillratio(fillratio2_df, dataset):
    # (A)
    fig, axes = plt.subplots(1,6, figsize=(13,3))
    tolerance = 0
    for method, ax in zip(fillratio2_df.columns[:-1], axes.flatten()[1:]):
        colors = [
            'gray' if abs(x - y) <= tolerance else ('black' if x > y else 'black') 
            for x, y in zip(fillratio2_df[method], fillratio2_df['AFsample2'])
        ]
        
        ax.scatter(fillratio2_df[method], fillratio2_df['AFsample2'], c=colors, s=20, edgecolor='gray', alpha=0.7, linewidth=0.5)

        ax.axline((0, 0), slope=1, linestyle='--', color='gray')
        ax.set_xlim(-0.05, 0.9)
        ax.set_ylim(-0.05, 0.9)
        
        ax.set_ylabel('AFsample2')
        ax.set_xlabel(f'{method}')

        # Add tolerance lines
        x_vals = np.linspace(-0.05, 0.9, 100)
        pval = wilcoxon( fillratio2_df['AFsample2'], fillratio2_df[method], alternative='greater').pvalue
        ax.text(0.4,0.02,f"p<{pval:.1e}")

    p = sns.color_palette()
    colors = [p[0], p[2], p[3], p[4], p[6], p[1]]

    cols = fillratio2_df.columns
    positions = range(len(cols))  # 0-based for consistency
    box_c = axes.flatten()[0].boxplot([fillratio2_df[col] for col in cols], patch_artist=True, showfliers=False,
                        positions=positions,  # Use the same 0-based positions
                        meanprops={'marker': '_', 'markerfacecolor': 'black', 'markeredgecolor': 'black'}, boxprops=dict(alpha=.7))
    for patch, color in zip(box_c['boxes'], colors):
        patch.set_facecolor('white')
        patch.set_edgecolor('gray')
    for median in box_c['medians']:
        median.set(color='black', linewidth=1.5)
    for i, col in enumerate(cols):
        sns.stripplot(x=np.full(len(fillratio2_df), i), y=fillratio2_df[col], color=colors[i], 
                    alpha=0.7, size=3, jitter=True, ax=axes.flatten()[0])

    rands = fillratio2_df.columns
    axes.flatten()[0].set_xticklabels(rands,rotation=45, rotation_mode='anchor', ha='right')
    axes.flatten()[0].set_ylabel('Fill ratio')

    plt.suptitle(f'Fill ratio ({dataset})')
    plt.tight_layout()
    plt.show()

    # (B)
    fig, ax = plt.subplots(figsize=(12,3.5))
    sns.heatmap(fillratio2_df.T, annot=True,  ax=ax, cmap='viridis_r', linecolor='black', linewidth=0.5, fmt=".2f")
    ax.set_title('Fill ratio (OC23)')
    ax.set_xlabel('')
    ax.set_xticklabels(fillratio2_df.index, rotation=45, rotation_mode='anchor', ha='right')
    plt.tight_layout()
    plt.show()


def load_pkl(pklpath):
	with open(pklpath, 'rb') as f:
		return pickle.load(f)

def plot_fluctuation_correlation(protein, data1_path, data3_path, fillratiopath):
    fill_ratio = pd.read_csv(fillratiopath, compression='gzip')
    fill_ratio.index = fill_ratio['Unnamed: 0']

    data1 = load_pkl(data1_path)
    data3 = load_pkl(data3_path)

    # Create the GridSpec object
    fig = plt.figure(figsize=(16, 3))
    gs = gridspec.GridSpec(1, 10, figure=fig)

    # Define subplots with different sizes
    ax1 = fig.add_subplot(gs[0, :6])  # First row, first two column
    ax2 = fig.add_subplot(gs[0, 6:8])   # First row, third column
    ax3 = fig.add_subplot(gs[0, 8:10])   # First row, third column

    common_residue_indices = data1[protein]['common_residue_indices']
    a1 = ax1.plot(data1[protein]['rmsf_baseline'][common_residue_indices], 
                label=r'$ \Delta $'+'C'+r'$ \alpha $'+' (References)', 
                linestyle='-', linewidth=1, color='black')
    #plt.grid(linestyle='--', zorder=-1)
    ax1_ = ax1.twinx()
    fill, corr = fill_ratio['AFvanilla'][protein], np.round(data1[protein]['r_spear'], 2)
    a2 = ax1_.plot(data1[protein]['model_rmsf'][common_residue_indices], 
                    label=f"RMSF afvanilla (Fill-ratio: {fill:.2f}, corr: {corr:.2f})",
                    alpha=0.8)

    fill, corr = fill_ratio['AFsample2'][protein], np.round(data3[protein]['r_spear'], 2)
    a2 = ax1_.plot(data3[protein]['model_rmsf'][common_residue_indices], 
                    label=f"RMSF afsample2 (Fill-ratio: {fill:.2f}, corr: {corr:.2f})",
                    alpha=0.8, c='darkorange')
    ax1.set_title(protein)
    #ax1.legend(ncol=4, bbox_to_anchor=(0.32, -0.19))
    ax1_.legend(bbox_to_anchor=(0.98, -0.17),  ncol=5)

    ax1_.set_ylabel("RMSF (ensemble) " + "$\mathrm{\AA}$")
    ax1_.set_ylabel(r'$ \Delta $'+'C'+r'$ \alpha $'+ ' (open, closed) '+ "$\mathrm{\AA}$")
    ax1_.set_xlabel('Residues')

    cx = ax2.scatter(data1[protein]['model_rmsf'][common_residue_indices],
                    data1[protein]['rmsf_baseline'][common_residue_indices],
                    c = data1[protein]['per_residue_bfactor_mean'][common_residue_indices], cmap='plasma_r', edgecolor='white', linewidth=0.5, label='AFvanilla', vmin=40)
    plt.colorbar(cx)
    ax2.set_xlabel('RMSF AFvanilla'+" ($\mathrm{\AA}$)")
    ax2.set_ylabel(r'$ \Delta $'+'C'+r'$ \alpha $'+ ' (open, closed) '+ "$\mathrm{\AA}$")
    ax2.set_title(protein+f" (r: {np.round(data1[protein]['r_spear'], 2)})")

    cx = ax3.scatter(data3[protein]['model_rmsf'][common_residue_indices],
                    data3[protein]['rmsf_baseline'][common_residue_indices],
                    c = data3[protein]['per_residue_bfactor_mean'][common_residue_indices], cmap='plasma_r', edgecolor='white', linewidth=0.5, label='AFsample2', vmin=40)
    plt.colorbar(cx)
    ax1.set_ylabel(r'$ \Delta $'+'C'+r'$ \alpha $'+ ' (open, closed) '+ "$\mathrm{\AA}$")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1_.spines["top"].set_visible(False)
    ax1_.spines["right"].set_visible(False)
    plt.tight_layout()
    ax3.set_xlabel('RMSF AFsample2'+" ($\mathrm{\AA}$)")
    ax3.set_ylabel(r'$ \Delta $'+'C'+r'$ \alpha $'+ ' (open, closed) '+ "$\mathrm{\AA}$")
    ax3.set_title(protein+f" (r: {np.round(data3[protein]['r_spear'], 2)})")
    plt.show()
    return None

def check_directory_structure(root_dir, required_structure):
    """
    Checks if the given directory structure and files exist.
    
    :param root_dir: The base directory to check.
    :param required_structure: A dictionary defining the expected directories and files.
    :return: None
    """
    missing_items = []
    print('\nChecking directory structure...')

    for dir_path, files in required_structure.items():
        full_dir_path = os.path.join(root_dir, dir_path)
        print(full_dir_path)
        # Check if directory exists
        if not os.path.isdir(full_dir_path):
            missing_items.append(f"Missing directory: {full_dir_path}")
        else:
            # Check for required files in the directory
            for file in files:
                file_path = os.path.join(full_dir_path, file)
                if not os.path.isfile(file_path):
                    missing_items.append(f"Missing file: {file_path}")

    # Output results
    if missing_items:
        print("Some required directories or files are missing (Available in Zenodo):")
        for item in missing_items:
            print(f"  - {item}")
    else:
        print("All required directories and files are present.")

def savedict(infile, outpath):
    with open(outpath, 'wb') as f:
        pickle.dump(infile, f)
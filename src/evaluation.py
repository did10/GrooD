# Import packages

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.utils.fixes import parse_version
import seaborn as sns
import math
import scipy.stats as stats
import xgboost as xgb


# Compare predictions and ground truth on a given dataset
def compare_data(x, y, MT, output, plotType):

    """
    Create regression, response or error plot to compare prediction to ground-truth
    x: ground truth
    y: prediction
    MT: MasterTable
    output: target directory
    plotType: regression, error or response plot
    """
    
    # Configure subplots
    n = x.shape[1]
    rows = math.floor(math.sqrt(n))
    cols = math.ceil(n/rows)

    groundTruthCells = x.columns.tolist()
    inferenceCells = y.columns.tolist()

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = np.array(axes)

    axes = axes.flatten()

    # Plot data
    MT['SampleID'] = MT.index.tolist()

    for i in range(0, n):
        if plotType == 'regression':
            X = groundTruthCells[i]
            Y= inferenceCells[i]
            sns.regplot(data=MT, x=X, y=Y, ax=axes[i])
            axes[i].set_title('Cell type: ' + X)
            axes[i].set_xlabel('Ground-truth proportions')
            axes[i].set_ylabel('Inferred proportions')

        elif plotType == 'response':
            X = groundTruthCells[i]
            Y= inferenceCells[i]
            df_melt = MT[[X, Y, 'SampleID']]
            df_melt.columns = ['Ground-truth', 'Prediction', 'SampleID']
            df_melt = df_melt.melt(id_vars=['SampleID'])

            if i < (n-1):
                sns.scatterplot(data = df_melt, x = "SampleID", y = "value", hue = "variable", ax=axes[i], legend=False)
                axes[i].set_title('Response plot for ' + X)
                axes[i].set_xlabel('Sample')
                axes[i].set_ylabel(X + ' proportion')
                axes[i].set_xticks([])
            if i == (n-1):
                sns.scatterplot(data = df_melt, x = "SampleID", y = "value", hue = "variable", ax=axes[i])
                axes[i].set_title('Response plot for ' + X)
                axes[i].set_xlabel('Sample')
                axes[i].set_ylabel(X + ' proportion')
                axes[i].set_xticks([])
                sns.move_legend(axes[i], "upper left", bbox_to_anchor=(1, 1))

        elif plotType == 'error':
            X = groundTruthCells[i]
            Y= inferenceCells[i]
            df_plot = MT
            df_plot['error'] = df_plot[X] - df_plot[Y]
            sns.barplot(data = df_plot, x = "SampleID", y = "error", ax=axes[i])
            axes[i].set_title('Prediction errors for ' + X)
            axes[i].set_xlabel('Sample')
            axes[i].set_ylabel('Deviance from ground-truth')
            axes[i].set_xticks([])

        # Hide any unused axes
    for j in range(n, len(axes)):
        fig.delaxes(axes[j])


    plt.tight_layout()
    plt.savefig(output + plotType + '_plot.pdf', transparent = True, dpi = 300, format = 'pdf')
    plt.savefig(output + plotType + '_plot.svg', transparent = True, dpi = 300, format = 'svg')
    plt.clf()


# Plotting feature importances
def plot_feature_importance(model, estimators, output):

    """
    For GrooD and XGrooD generates barplots of most important features per cell type model
    model: scikit-learn model for GrooD or XGrooD
    estimators: cell types in same order as described in model
    output: target directory for plot
    """
    
    # Get list of cellTypes (same order as in model estimators)
    cellTypes = estimators
    genes = model.estimators_[0].feature_names_in_.tolist()

    # Configure subplots
    n = len(cellTypes)
    rows = math.floor(math.sqrt(n))
    cols = math.ceil(n/rows)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5))
    axes = np.array(axes)

    axes = axes.flatten()

    # iterate over n or cellTypes
    for i in range(0,len(cellTypes)):
        cellType = cellTypes[i]
        feature_importance = model.estimators_[i].feature_importances_
        sorted_idx = np.argsort(feature_importance) # sort highest to lowest importance and get the feature indices
        most_important_features = sorted_idx[-20:] # get 20 most important features
        pos = np.arange(most_important_features.shape[0]) + 0.5
        axes[i].barh(pos, feature_importance[most_important_features], align="center")
        axes[i].set_yticks(pos, np.array(genes)[most_important_features])
        axes[i].set_title("Feature Importance (MDI) for " + cellType)

    # Hide any unused axes
    for j in range(len(cellTypes), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(output + 'Feature_importance_plot.pdf', transparent = True, dpi = 300, format = 'pdf')
    plt.savefig(output + 'Feature_importance_plot.svg', transparent = True, dpi = 300, format = 'svg')
    plt.clf()

# Plot annotated heatmap

def annotated_heatmap(df, output):

    """"
    Visualizes correlation and error metrics computed with getCorr in heatmaps with fixed scales
    df: output from getCorr
    output: target directory to save plot in
    """

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    df1 = df.loc[['PCC', 'SCC', 'CCC']] # data for SCC, PCC, CCC
    df2 = df.loc[['RMSE', 'RD', 'MAD']] # data for RMSE and percent deviation

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    im1 = ax1.imshow(df1, vmin=-1, vmax=1, cmap='viridis_r')

    for i in range(df1.shape[0]):
        for j in range(df1.shape[1]): 
            ax1.annotate(str(round(df1.iloc[i,j], 2)), xy=(j, i), 
                        ha='center', va='center', color='white')

    ax1.set_title('Correlation: ground-truth to prediction')
    ax1.set_xlabel('Cell type')
    ax1.set_ylabel('Parameter')
    ax1.set_yticks(range(df1.shape[0]), df1.index.tolist())
    ax1.set_xticks(range(df1.shape[1]), df1.columns.tolist(), rotation = 'vertical')
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im1, cax = cax, ax = ax1)

    im2 = ax2.imshow(df2, vmin=0, vmax=1, cmap='viridis')

    for i in range(df2.shape[0]):
        for j in range(df2.shape[1]): 
            ax2.annotate(str(round(df2.iloc[i,j], 2)), xy=(j, i), 
                        ha='center', va='center', color='white')

    ax2.set_title('Deviation: ground-truth to prediction')
    ax2.set_xlabel('Cell type')
    ax2.set_ylabel('Parameter')
    ax2.set_yticks(range(df2.shape[0]), df2.index.tolist())
    ax2.set_xticks(range(df2.shape[1]), df2.columns.tolist(), rotation = 'vertical')
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im2, cax = cax, ax = ax2)

    plt.savefig(output + 'QC_metrics_plot.pdf', transparent = True, dpi = 300, format = 'pdf')
    plt.savefig(output + 'QC_metrics_plot.svg', transparent = True, dpi = 300, format = 'svg')
    plt.clf()


# Correlation analysis

def getCorr(df, cellTypes):

    """
    Computes correlation and error metrics for predicitions vs. ground-truth as stored in the MasterTable
    df: MasterTable of samples x cell types (predicted proportions with prefix 'Pred ', groundtruth only with cell type name)
    cellTypes: basically list of cell types as in groundtruth
    """


    rsquared = []
    cccs = []
    rmses = []
    spearmans = []
    percent_dev = []
    absol_dev = []

    # Concordance correlation coefficient
    def ccc(x_data,y_data):
        sxy = np.sum((x_data - np.mean(x_data))*(y_data - np.mean(y_data)))/len(x_data)
        rhoc = 2*sxy / (np.var(x_data) + np.var(y_data) + (np.mean(x_data) - np.mean(y_data))**2)
        return rhoc


    # Pearson Correlation Coefficient
    def r(x_data,y_data):
        ''' Pearson Correlation Coefficient'''
        sxy = np.sum((x_data - np.mean(x_data))*(y_data - np.mean(y_data)))/len(x_data)
        rho = sxy / (np.std(x_data)*np.std(y_data))
        return rho


    # RMSE
    def rmse(x_data,y_data):
        MSE = np.mean(np.square((np.array(x_data)-np.array(y_data))))
        RMSE = np.sqrt(MSE)
        return RMSE

    # Spearmans rank correlation    
    def spearman(x_data, y_data):
        spearmanCorr, _ = stats.spearmanr(np.array(x_data), np.array(y_data))
        return spearmanCorr
    
    # Percent deviation function
    def frac_deviation(x_data, y_data):
        frac_devs = np.mean(np.divide((np.array(y_data) - np.array(x_data)),np.array(x_data), where=(np.array(x_data)!=0)))
        return frac_devs # for sparse data, quite some data points might be excluded due to the removal of 0-divisions from the mean
    
    # Absolute deviation function
    def abs_deviation(x_data, y_data):
        abs_devs = np.mean(abs(np.array(y_data) - np.array(x_data)))
        return abs_devs


    for cell in cellTypes:
        groundTruth = cell
        pred = 'Pred ' + cell
        
        xdata = df[groundTruth].tolist()
        ydata = df[pred].tolist()
        b = r(xdata, ydata)
        rsquared.append(b)
        s = spearman(xdata, ydata)
        spearmans.append(s)
        c = ccc(xdata, ydata)
        cccs.append(c)
        a = rmse(xdata, ydata)
        rmses.append(a)
        f = frac_deviation(xdata, ydata)
        percent_dev.append(f)
        g = abs_deviation(xdata, ydata)
        absol_dev.append(g)

    DF = pd.concat([pd.Series(rsquared), pd.Series(spearmans), pd.Series(cccs), pd.Series(rmses), pd.Series(percent_dev), pd.Series(absol_dev)], axis = 1, ignore_index = True)
    DF.columns = ['PCC', 'SCC', 'CCC', 'RMSE', 'RD', 'MAD']
    DF.index = cellTypes

    return DF


# Plot to visualize distribution of predicted cell type fractions

def boxplot_props(pred, output):

    """
    Creates boxplot of proportions per cell type across samples in pred
    pred: samples x cell types pandas dataframe
    output: target directory
    """

    pred_new = pred.copy()
    pred_new['SampleID'] = pred.index.tolist()
    df = pred_new.melt(id_vars='SampleID')

    p = sns.boxplot(data=df, x = "variable", y = "value", hue = "variable")
    p.set_xticklabels(p.get_xticklabels(), rotation=45, horizontalalignment='right')
    plt.tight_layout()
    plt.savefig(output + 'Boxplot_inferred_proportions.pdf', transparent = True, dpi = 300, format = 'pdf')
    plt.savefig(output + 'Boxplot_inferred_proportions.svg', transparent = True, dpi = 300, format = 'svg')
    plt.clf()


# Barplot for cell type proportions

def plotStackedBars(df, output):

    """
    Generates stacked bar plot per sample in prediction (df)
    df: samples x cell types pandas dataframe
    output: target directory
    """
    
    sns.set(style='white')

    df.plot(kind = 'bar', stacked = True)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, title = 'Cell Types')
    if df.shape[0] > 10:
        plt.xticks([])
    plt.title('Cell type proportions per sample')
    plt.tight_layout()
    plt.savefig(output + 'Cell_type_proportions_per_sample.pdf', transparent = True, dpi = 300, format = 'pdf')
    plt.savefig(output + 'Cell_type_proportions_per_sample.svg', transparent = True, dpi = 300, format = 'svg')
    plt.clf()


# Visualize predicted cell type proportions

def visualize_predict(pred, output):

    """
    Visualize predictions with stacked barplot and boxplot
    pred: samples x cell types pandas dataframe
    output: target directory
    """
    
    # Barplot
    plotStackedBars(pred, output)

    # Boxplot
    boxplot_props(pred, output)

    return None


# Heatmap explaining prediction on feature level/expression level
def explain_heatmap_features(model, estimators, data, prop, cell_type):

    # Get most important features
    index = estimators.index(cell_type)
    feature_importance = model.estimators_[index].feature_importances_
    sorted_idx = np.argsort(feature_importance) # sort highest to lowest importance and get the feature indices
    most_important_features = sorted_idx[-20:] # get 20 most important features
    genes = np.flip(np.array(model.estimators_[index].feature_names_in_.tolist())[most_important_features])

    # Format data
    data_selected = data[genes].transpose() # subset bulk data
    prop_series = pd.Series(prop[cell_type]) # series for proportion for cell type investigated here

    # Remove genes with all-zero expression across samples
    non_zero_genes = (data_selected.sum(axis=1) != 0)
    data_selected = data_selected.loc[non_zero_genes]

    # Remove genes with zero variance
    non_zero_var_genes = (data_selected.var(axis=1) != 0)
    data_selected = data_selected.loc[non_zero_var_genes]

    # Create clustermap
    g = sns.clustermap(
        data_selected,
        col_cluster=True,
        row_cluster=False,
        cmap="viridis",
        figsize=(8, 6),
        dendrogram_ratio=0.35,
        z_score=0
    )

    # Reorder the sample values to match clustered column order
    col_order = [data_selected.columns[i] for i in g.dendrogram_col.reordered_ind]
    prop_series_reordered = prop_series[col_order]

    # Add a new axis on top for the bar plot
    bar_height = 0.5
    heatmap_pos = g.ax_heatmap.get_position()

    bar_ax = g.fig.add_axes([
        heatmap_pos.x0,                      
        heatmap_pos.y1 + 0.02,               
        heatmap_pos.width,                   
        bar_height / g.fig.get_size_inches()[1] 
    ])

    # Plot bars
    bar_ax.bar(
        x=np.arange(len(prop_series_reordered)),
        height=prop_series_reordered,
        color="gray"
    )

    # Remove x ticks
    bar_ax.set_xticks([])
    bar_ax.set_xlim(-0.5, len(prop_series_reordered) - 0.5)
    bar_ax.set_ylabel(f"Proportion {cell_type}")

    # Optional: make y-axis more compact
    bar_ax.spines["top"].set_visible(False)
    bar_ax.spines["right"].set_visible(False)

    return g

def get_explain_heatmap(model, estimators, bulk, pred, output):

    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(f"{output}Explaining_predictions_heatmaps.pdf") as pdf:
        for cell_type in estimators:
            explain_heatmap_features(model, estimators, bulk, pred, cell_type)

            plt.figtext(
                0.5,               
                0.98,              
                f"Analysis for cell type: {cell_type}",  
                ha='center',       
                fontsize=10
            )

            pdf.savefig()
            plt.close()

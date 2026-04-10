import argparse
import anndata
import numpy as np
import scanpy as sc
import pandas as pd
from tqdm import tqdm
import random
from numpy.random import choice
import warnings
from src.tools import generate_simulated_data, generate_simulated_data_per_target, pseudobulk_norm

# Define simulator function

def simulator(ncells = 1000, samplenum = 5000,
              sparse = 0.2, rare = 0.3,
              propPath = None, sc_layer = "X", 
              filter_genes = "mRNA", norm = "CPM",
              sc_path = None, outdir = "./",
              threads = 8, target = None, target_name = None):
    
    """
    Simulates pseudobulks from scRNA-seq data

    ncells: number of cells to select for each pseudobulk
    samplenum: number of pseudobulks to generate in total
    sc_path: path to single-cell data in h5ad format
    sparse: probability for sparse cell types (0 prop)
    rare: probability for rare cell types (close to 0 prop)
    propPath: proportions can be specified in csv formatm, but should match samplenum
    filter_genes: select from 3k highly variable (3k), mRNA genes (mRNA) or all genes (all)
    norm: normalization of pseudobulks by CPM, rank or remaining as raw counts
    """

    # Step 1: Load single-cell data, select relevant pre-normalited layer; scData should already be QC'ed beforehand; load proportions if supplied

    # Load single-cell data
    inData = sc.read_h5ad(sc_path)

    if sc_layer not in inData.layers.keys():
        if sc_layer == "unspecified":
            sc_layer = 'X'
        else:
            print('Specified sc_layer ' + sc_layer + ' not in scRNA-seq data object. Using default layer.')
            sc_layer = 'X'
            

    # Extract relevant layer from sc data
    if sc_layer == 'X':
        print('Using default layer for pseudobulk simulation.')
        scData = inData.to_df() 
        scData['CellType'] = inData.obs['cell_type']
    elif sc_layer != 'X':
        print('Using layer ' + sc_layer + ' from scRNA-seq data for pseudobulk simulation.')
        scData = inData.to_df(layer = sc_layer) 
        scData['CellType'] = inData.obs['cell_type']

    if 'individual' in inData.obs.columns.tolist():
        scData['individual'] = inData.obs['individual']
    else:
        scData['individual'] = 'unspecified'

    if 'condition' in inData.obs.columns.tolist():
        scData['condition'] = inData.obs['condition']
    else:
        scData['condition'] = 'unspecified'

    # Load proportions, if supplied

    if propPath == None:
        props = None
    elif propPath.endswith('.csv'):
        props = pd.read_csv(propPath, index_col=0)
        if props.shape[0] != samplenum:
            print('Number of samples in proportions is not matching specified sample number. Sample number is now adjusted to ' + str(props.shape[0]) +'.')
            samplenum = props.shape[0]
        elif props.shape[1] != len(scData['CellType'].value_counts()):
            raise ValueError('Number of cell types in proportions is not matching number of cell types in scRNA-seq data.')
        else:
            print('Props match specified parameters.')
    else:
        raise ValueError('Proportions not in csv format. Please supply as csv file of samples x cell types.')


    # Step 2: Create simulated data

    if target == None:
        pseudobulks = generate_simulated_data(scData,
                                        n = ncells,
                                        samplenum = samplenum,
                                        props=props,
                                        sparse=True,
                                        sparse_prob=sparse,
                                        rare=True,
                                        rare_percentage=rare,
                                        n_jobs=threads)

    elif target != None:
        pseudobulks = generate_simulated_data_per_target(scData, target=target, target_name=target_name,
                                                        n = ncells,
                                                        samplenum = samplenum,
                                                        props=props,
                                                        sparse=True,
                                                        sparse_prob=sparse,
                                                        rare=True,
                                                        rare_percentage=rare,
                                                        n_jobs=threads)


    # Step 3: Normalize simulated data and filter data in different scenarios

    proportionsDF, pseudobulkDF = pseudobulk_norm(pseudobulks, norm, filter_genes)


    #Step 4: Export as csv
    print('Writing pseudobulks to output.')

    proportionsDF.to_csv(outdir + 'Pseudobulk_proprotions.csv')
    pseudobulkDF.to_csv(outdir + 'Pseudobulks.csv')

    return pseudobulkDF, proportionsDF
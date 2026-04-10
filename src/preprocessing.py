# Import packages

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

from sklearn import ensemble
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.utils.fixes import parse_version
from src.simulation import simulator
from src.tools import create_pseudobulk_dir, main_gene_selection, bulk_norm, remove_zero_variance
from pathlib import Path


# loading and formatting of pseudobulk or bulk data
# gene subsetting either based on highly variable genes or based on genes model has been trained with


def load_train_test_data(args):
    """
    EITHER use sc data for pseudobulk simulation, if provided
    If additional pseudobulk_props are provided these will be used for simulation

    OR pseudobulks and associated props, if provided - this will override the sc settings

    ERROR will be raised when pseudobulks are provided but no associated props
    """
    

    # CASE 1: single-cell data for simulation provided
    if args.sc is not None:
        print('Single-cell data provided. Please ensure that a column termed cell_type is included in obs of AnnData.')
        print('If bulk data path was provided, it will be overwritten by the pseudobulk simulation.')
        if args.pseudobulk_props is None:
            print('No cell type proportions provided. New proportions will be simulated.')
            pb, pb_props = simulator(samplenum=args.no_pseudobulks,
                                    sc_path=args.sc,
                                    ncells=args.no_cells,
                                    outdir=create_pseudobulk_dir(args.output),
                                    target=args.target,
                                    target_name=args.target_name,
                                    threads=args.threads,
                                    norm=args.norm,
                                    filter_genes=args.feature_curation)
        elif args.pseudobulk_props is not None:
            print('Using provided cell type proportions for simulation.')
            pb, pb_props = simulator(samplenum=args.no_pseudobulks,
                                    sc_path=args.sc,
                                    propPath=args.pseudobulk_props,
                                    ncells=args.no_cells,
                                    outdir=create_pseudobulk_dir(args.output),
                                    target=args.target,
                                    target_name=args.target_name,
                                    threads=args.threads,
                                    norm=args.norm,
                                    filter_genes=args.feature_curation)
            

    # CASE 2: pseudobulks and corresponding proportions provided
    elif args.pseudobulks is not None and args.pseudobulk_props is not None:

        # Pseudobulks
        if args.pseudobulks[-4:] == '.csv':
            pb = pd.read_csv(args.pseudobulks, index_col=0)
        elif args.pseudobulks[-4:] == '.tsv':
            pb = pd.read_csv(args.pseudobulks, index_col=0, sep='\t')
        elif args.pseudobulks[-5:] == '.h5ad':
            pb = sc.read_h5ad(args.pseudobulks)
            pb = pb.to_df()
        else:
            raise ValueError('Wrong input format for bulk data. Please supply in either csv, tsv or h5ad format.')


        ## Cell type proportions
        if args.pseudobulk_props[-4:] == '.csv':
            pb_props = pd.read_csv(args.pseudobulk_props, index_col=0)
        elif args.pseudobulk_props[-4:] == '.tsv':
            pb_props = pd.read_csv(args.pseudobulk_props, index_col=0, sep = '\t')
        else:
            raise ValueError('Wrong input format for cell type proportions. Please supply only in csv format.')


    # CASE 3: pseudobulks but no corresponding proportions provided
    elif args.pseudobulks is not None and args.pseudobulk_props is not None:
        raise ValueError('Pseudobulks provided for train_test mode, but associated proportions are missing.')
    
    return pb, pb_props


def load_inference_data(args):
    """
    Use bulk data for inference: samples x genes
    If additional pseudobulk_props are provided these will be used for evaluating the predictions later on: samples x cell types
    """

    # BULK
    if args.bulk is not None:
        print('Bulk data provided.')
        if args.bulk[-4:] == '.csv':
            bulk = pd.read_csv(args.bulk, index_col=0)
        elif args.bulk[-4:] == '.tsv':
            bulk = pd.read_csv(args.bulk, index_col=0, sep='\t')
        elif args.bulk[-5:] == '.h5ad':
            bulk = sc.read_h5ad(args.bulk)
            bulk = bulk.to_df()
        elif args.bulk is None:
            raise ValueError('No bulk data provided, but required for inference.')
        else:
            raise ValueError('Wrong input format for bulk data. Please supply in either csv, tsv or h5ad format.')
        

    # PROPORTIONS
    if args.props is not None:

        print('Proportions are provided and will be considered for evaluation.')

        ## Cell type proportions
        if args.props[-4:] == '.csv':
            props = pd.read_csv(args.props, index_col=0)
        elif args.props[-4:] == '.tsv':
            props = pd.read_csv(args.props, index_col=0, sep = '\t')
        else:
            raise ValueError('Wrong input format for cell type proportions. Please supply in csv or tsv format.')


        # CASE, if bulk and pseudobulks are not matched
        if props.shape[0] != bulk.shape[0]:
            print(f"Bulks and proportions are provided for different sample numbers. Bulk sample number: {bulk.shape[0]}, proportion sample number: {props.shape[0]}.")
            print("Proportions removed. Inference without evaluation.")
            props = None

    else:
        props = None
    
    return bulk, props



def load_all_data(args):
    """
    EITHER use sc data for pseudobulk simulation, if provided
    If additional pseudobulk_props are provided these will be used for simulation

    OR pseudobulks and associated props, if provided - this will override the sc settings

    ERROR will be raised when pseudobulks are provided but no associated props

    Also loads bulks and optional associated proportions

    On that basis, all data will be normalized (args.norm) and genes will be subset based on args.feature_curation
    """

    # CASE 1: single-cell data for simulation provided
    if args.sc is not None:
        print('Single-cell data provided. Please ensure that a column termed cell_type is included in obs of AnnData.')
        print('If bulk data path was provided, it will be overwritten by the pseudobulk simulation.')
        if args.pseudobulk_props is None:
            print('No cell type proportions provided. New proportions will be simulated.')
            pb, pb_props = simulator(samplenum=args.no_pseudobulks,
                                    sc_path=args.sc,
                                    ncells=args.no_cells,
                                    outdir=create_pseudobulk_dir(args.output),
                                    target=args.target,
                                    target_name=args.target_name,
                                    threads=args.threads,
                                    norm="none",
                                    filter_genes="all")
        elif args.pseudobulk_props is not None:
            print('Using provided cell type proportions for simulation.')
            pb, pb_props = simulator(samplenum=args.no_pseudobulks,
                                    sc_path=args.sc,
                                    propPath=args.pseudobulk_props,
                                    ncells=args.no_cells,
                                    outdir=create_pseudobulk_dir(args.output),
                                    target=args.target,
                                    target_name=args.target_name,
                                    threads=args.threads,
                                    norm="none",
                                    filter_genes="all")
            

    # Step 2: pseudobulks and corresponding proportions provided
    elif args.pseudobulks is not None and args.pseudobulk_props is not None:

        print(f"Please ensure provided pseudobulks are already normalized with given strategy: {args.norm}")

        # Pseudobulks
        if args.pseudobulks[-4:] == '.csv':
            pb = pd.read_csv(args.pseudobulks, index_col=0)
        elif args.pseudobulks[-4:] == '.tsv':
            pb = pd.read_csv(args.pseudobulks, index_col=0, sep='\t')
        elif args.pseudobulks[-5:] == '.h5ad':
            pb = sc.read_h5ad(args.pseudobulks)
            pb = pb.to_df()
        else:
            raise ValueError('Wrong input format for bulk data. Please supply in either csv, tsv or h5ad format.')


        ## Cell type proportions
        if args.pseudobulk_props[-4:] == '.csv':
            pb_props = pd.read_csv(args.pseudobulk_props, index_col=0)
        elif args.pseudobulk_props[-4:] == '.tsv':
            pb_props = pd.read_csv(args.pseudobulk_props, index_col=0, sep = '\t')
        else:
            raise ValueError('Wrong input format for cell type proportions. Please supply only in csv format.')


    # Scenario: pseudobulks but no corresponding proportions provided
    elif args.pseudobulks is not None and args.pseudobulk_props is None:
        raise ValueError('Pseudobulks provided for train_test mode, but associated proportions are missing.')
    

    # STEP 3: bulks and optional proportions
    # BULK
    if args.bulk is not None:
        print('Bulk data provided.')
        if args.bulk[-4:] == '.csv':
            bulk = pd.read_csv(args.bulk, index_col=0)
        elif args.bulk[-4:] == '.tsv':
            bulk = pd.read_csv(args.bulk, index_col=0, sep='\t')
        elif args.bulk[-5:] == '.h5ad':
            bulk = sc.read_h5ad(args.bulk)
            bulk = bulk.to_df()
        elif args.bulk is None:
            raise ValueError('No bulk data provided, but required for inference.')
        else:
            raise ValueError('Wrong input format for bulk data. Please supply in either csv, tsv or h5ad format.')
        

    # PROPORTIONS
    if args.props is not None:

        print('Proportions are provided and will be considered for evaluation.')

        ## Cell type proportions
        if args.props[-4:] == '.csv':
            props = pd.read_csv(args.props, index_col=0)
        elif args.props[-4:] == '.tsv':
            props = pd.read_csv(args.props, index_col=0, sep = '\t')
        else:
            raise ValueError('Wrong input format for cell type proportions. Please supply in csv or tsv format.')


        # scenario, if bulk and pseudobulks are not matched
        if props.shape[0] != bulk.shape[0]:
            print(f"Bulks and proportions are provided for different sample numbers. Bulk sample number: {bulk.shape[0]}, proportion sample number: {props.shape[0]}.")
            print("Proportions removed. Inference without evaluation.")
            props = None

    else:
        props = None

    # mRNA_file_path 
    script_dir = Path(__file__).resolve().parent

    # path to the text file
    file_path = script_dir / "mRNA_annotation.tsv"


    # STEP 4: subsetting data
    if args.feature_curation == "all":
        """
        Extract all genes from pb and take all of those & to subset bulk; missing genes will be imputed by 0
        """
        pb = pb
        genes = pb.columns.tolist()
        bulk, _, _ = main_gene_selection(bulk, genes)

    elif args.feature_curation == "mRNA":
        """
        Extract all mRNA genes from pb and bulk; missing genes will be imputed by 0
        """
        # Import gene list for filtering
        gene_list_df = pd.read_csv(file_path, header=0, delimiter='\t')
        gene_list = list(gene_list_df['gene_name'])

        # Subset pb & bulk
        pb, _, _ = main_gene_selection(pb, gene_list)
        bulk, _, _ = main_gene_selection(bulk, gene_list)
    
    elif args.feature_curation == "non_zero":
        """
        Extract all non_zero variance genes from pb and take all of those & to subset bulk; missing genes will be imputed by 0
        """
        # Remove zero variance genes from pb
        pb = remove_zero_variance(pb)

        # Subset bulk
        bulk, _, _ = main_gene_selection(bulk, pb.columns.tolist())        

    elif args.feature_curation == "intersect":
        """
        Extract all genes intersecting bulk and pseudobulk
        """
        common_genes = [g for g in bulk.columns.tolist() if g in set(pb.columns.tolist())]
        if len(common_genes) == 0:
            raise ValueError('No common genes between bulk and pseudobulk.')
        bulk = bulk[common_genes]
        pb = pb[common_genes]

    elif args.feature_curation == "mRNA_intersect":
        """
        Extract all mRNA genes from pb and bulk; missing genes will be imputed by 0
        """

        # Subset pb & bulk
        pb = remove_zero_variance(pb)
        bulk = remove_zero_variance(bulk)

        common_genes = [g for g in bulk.columns.tolist() if g in set(pb.columns.tolist())]
        if len(common_genes) == 0:
            raise ValueError('No common genes between bulk and pseudobulk.')
        bulk = bulk[common_genes]
        pb = pb[common_genes]

        # Import gene list for filtering
        gene_list_df = pd.read_csv(file_path, header=0, delimiter='\t')
        gene_list = list(gene_list_df['gene_name'])

        # Subset pb & bulk
        pb, _, _ = main_gene_selection(pb, gene_list)
        bulk, _, _ = main_gene_selection(bulk, gene_list)

    elif args.feature_curation == "non_zero_intersect":
        """
        Extract all non_zero genes from pb and bulk; missing genes will be imputed by 0
        """
        # Subset pb & bulk
        pb = remove_zero_variance(pb)
        bulk = remove_zero_variance(bulk)

        common_genes = [g for g in bulk.columns.tolist() if g in set(pb.columns.tolist())]
        if len(common_genes) == 0:
            raise ValueError('No common genes between bulk and single-cell after zero-variance filtering.')
        bulk = bulk[common_genes]
        pb = pb[common_genes]
        

    # STEP 5: normalizing data

    if args.norm == "CPM":
        pb = bulk_norm(pb, "CPM")
        bulk = bulk_norm(bulk, "CPM")

    elif args.norm == "log":
        pb = bulk_norm(pb, "log")
        bulk = bulk_norm(bulk, "log")
    
    elif args.norm == "rank":
        pb = bulk_norm(pb, "rank")
        bulk = bulk_norm(bulk, "rank")
    else:
        pb = pb
        bulk = bulk
    
    return pb, pb_props, bulk, props
# Load packages

print('Loading required packages...')
import warnings
warnings.filterwarnings('ignore')
import multiprocessing
multiprocessing.set_start_method('forkserver')
import argparse
import numpy as np
import pandas as pd
import scanpy as sc

from src.tools import create_pred_dir, create_inference_dir
from src.deconvolution import train_eval_GrooD, train_eval_XGrooD, train_eval_MultiGrooD, eval_inference, inference_grood_models, inference_loaded_grood
from src.preprocessing import load_train_test_data, load_inference_data, load_all_data
from src.evaluation import visualize_predict, get_explain_heatmap

# Set up argparser

def parse_args():

    """
    # Input data
    bulk: path to bulk data
    props: cell type proportions associated to bulk or indpendent proportions to be used in pseudobulk simulation
    sc: single-cell data path (h5ad object)
    OR:
    # Pre-generated pseudobulks
    pseudobulks: path to pseudobulks
    pseudobulk_props: path to props associated to supplied pseudobulks

    # Simulation parameters, if simulation employed
    no_pseudobulks: no of pseudobulks to simulate; will be overwritten if props provided
    no_cells: no of cells to sample for each pseudobulk
    target: feature in obs (condition/individual) to use for specific simulaion
    target_name: to use only a specific target_name, when target specified

    # Mode for model use/generation
    mode: train_test (for training and evaluation of model) or inference mode (for evalution on different data), or all for both modes together
    grood_mode: 

    # Model parameters training
    depth: maximum depth per tree
    n_estimators: number of trees to be trained
    loss_function: "squared_error", "absolute_error", "huber"
    learning_rate: default 0.01
    min_samples_split: minimum number of samples to justify a split

    # Input for inference mode
    model_path: path to trained model; !attention: GrooD generates a pickled model file in dictionary format not only containing the trained model!

    # Multithreading
    threads: number of cpu cores to be used for training

    # Output
    output: specified output directory, will be created if it does not exist
    """

    parser = argparse.ArgumentParser(description='GradientBoostedDeconvolution (GrooD) - deconvolution of bulk transcriptomes using GradientBoostingRegressors')

    #### Input data

    ###### Set 1: Bulk data and, optionally (depending on mode), cell type proportions (do not provide bulk, if providing set 2 --> set two will override bulk from set 1; props can be used to constrain simulation)
    parser.add_argument('--bulk', default=None, type=str, help='input bulk data path: either pseudobulk data for train-test mode, or independent bulk data for inference mode. Bulk can be in csv, tsv or h5ad format. As h5ad default layer X needs to contain the desired data.')
    parser.add_argument('--props', default=None, type=str, help='cell type proportions either used for training in train-test mode, or for evaluation of deconvolution of independent bulk data in inference mode. No proportions have to be supplied in inference mode. Supply in csv or tsv format.')

    ###### Set 2: Supply path to single-cell data to generate defined amount of pseudobulks with random proportions
    parser.add_argument('--sc', type=str, default=None, help='input scRNA-seq data path, data in h5ad format for pseudobulk simulation')
    parser.add_argument('--no_pseudobulks', type=int, default=1000, help='Number of pseudobulks to simulate.')
    parser.add_argument('--no_cells', type=int, default=1000, help='Number of cells to sample per pseudobulk.')
    parser.add_argument('--target', type=str, default=None, choices={"condition", "individual"}, help='Specifiy condition or individual as layer for specific pseudobulk simulation.')
    parser.add_argument('--target_name', type=str, default=None, help='Specify a condition or individual, if simulation should be conducted for the specific layer. Works only, if target is specified, and if contained in specified target column in observations.')
    # Optional pre-generated pseudobulks for training
    parser.add_argument('--pseudobulks', type=str, default=None, help='input scRNA-seq data path, data in h5ad format for pseudobulk simulation')
    parser.add_argument('--pseudobulk_props', type=str, default=None, help='input scRNA-seq data path, data in h5ad format for pseudobulk simulation')

    ###### Set 3: Data normalization and feature curation
    parser.add_argument('--norm', type=str, default='CPM', choices={'CPM', 'log', 'rank', 'none'}, help='Way how pseudobulks/bulks will be normalized prior to inference.')
    parser.add_argument('--feature_curation', type=str, default='grood', choices={'all', 'mRNA', 'non_zero', 'intersect', 'mRNA_intersect', 'non_zero_intersect'}, help='"all", "mRNA" and "non_zero" only work do not work in inference mode; "intersect" options only work in mode "all".')    


    #### Usage mode: train-test, inference or all
    parser.add_argument('--mode', type=str, default='inference', choices={'train_test', 'inference', 'all'}, help='train_test mode for model training and evaluation; inference mode for deconvolution of independent bulk transcriptomics data; all for both.')
    parser.add_argument('--grood_mode', type=str, default='grood', choices={'grood', 'xgrood', 'multigrood'}, help='train_test mode for model training and evaluation; inference mode for deconvolution of independent bulk transcriptomics data; all for both.')

    #### Parameters for gradient boosting regressor training
    parser.add_argument('--depth', type=int, default=4, help='maximal depth of the decision trees that will be trained. Only used in train-test mode.')
    parser.add_argument('--n_estimators', type=int, default=500, help='number of trees to train. Only used in train-test mode.')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='supply learning_rate. Only used in train-test mode.')
    parser.add_argument('--loss_function', type=str, default="squared_error", choices={"squared_error", "absolute_error", "huber"}, help='loss_function (huber combines absolute and squared error). Only used in train-test mode.')
    parser.add_argument('--min_samples_split', type=int, default=5, help='minimum number of samples to justify a split. Only used in train-test mode.')

    #### Path to already trained model in inference mode
    parser.add_argument('--model_path', type=str, default="/model/Model.pkl", help='supply a pre-trained model by specifying its path. Only used in inference mode. Model should be a .pkl file. Only accepts custom GrooD models in pkl format.')

    #### Specify number of threads (important for training speed)
    parser.add_argument('--threads', type=int, default=1, help='supply number of threads to use during training, inference and pseudobulk simulation.')

    #### Path to output
    parser.add_argument('--output', type=str, help='Specify a path to the output folder and a prefix added to all output files. Directory will be created, if it does not exist.')

    return parser.parse_args()

def main():

    """
    Run GrooD/XGrooD/MUltiGrooD in three different possible modi
    1) train_test: simulate (or use provided pseudobulks with proportions) to train the model and 
    2) inference: deconvolve bulk data with optional props on a trained model; does not need a flag specifying the model type
    3) all: train_test and inference; offers additional possibility to jointly process and subset pseudobulk and bulk data increasing accuracy due to convolution
    Always generates visualizations of predictions and optional visualizations of comparisons to ground-truth, if available
    """

    # Mode-dependent training or inference

    # MODE 1: train_test
    if args.mode == 'train_test':
        print('-------------------')
        print(f"{args.mode} mode selected.")
        print('-------------------')

        # Load (simulated) pseudobulks and probs for training
        pb, props = load_train_test_data(args)

        # Preprocess pseudobulks # TODO: implement 


        # Access either GrooD, XGrooD or MultiGrooD architecture for training
        if args.grood_mode == 'grood':

            print('-------------------')
            print('Selected architecture: GrooD')
            print('-------------------')

            # Construct parameters for GrooD model training
            params = {'max_depth' : args.depth,
                    'n_estimators' : args.n_estimators,
                    'learning_rate' : args.learning_rate,
                    'loss' : args.loss_function,
                    'min_samples_split' : args.min_samples_split
            }

            print('-------------------')
            print('Model will be trained with the following parameters: ')
            for key in params.keys():
                print(key, ':', params[key])
            print('-------------------')

            # Call training (this includes train-test split, training of the model, saving of the trained model and evaluation with the test data)
            print('-------------------')
            print('Starting train_test of GrooD...')
            model = train_eval_GrooD(pb, props, params, args.output, args.threads, args.norm)
            print('Training done. Evaluating now...')
        
        elif args.grood_mode == 'xgrood':

            print('-------------------')
            print('Selected architecture: XGrooD')
            print('-------------------')

            if args.loss_function == 'squared_error':
                lf = "reg:squarederror"
            elif args.loss_function == 'absolute_error':
                lf = "reg:absoluteerror"

            # Construct parameters for MultiGrooD model training
            params = {'max_depth' : args.depth,
                    'n_estimators' : args.n_estimators,
                    'learning_rate' : args.learning_rate,
                    'objective' : lf,
                    'min_child_weight' : args.min_samples_split
            }

            print('-------------------')
            print('Model will be trained with the following parameters: ')
            for key in params.keys():
                print(key, ':', params[key])
            print('-------------------')

            # Call training (this includes train-test split, training of the model, saving of the trained model and evaluation with the test data)
            print('-------------------')
            print('Starting train_test of XGrooD...')
            model = train_eval_XGrooD(pb, props, params, args.output, args.threads, args.norm)
            print('Training & evaluation done.')

        elif args.grood_mode == 'multigrood':

            print('-------------------')
            print('Selected architecture: MultiGrooD')
            print('-------------------')


            # Construct parameters for MultiGrooD model training
            params = {'max_depth' : args.depth,
                    'n_estimators' : args.n_estimators,
            }

            print('-------------------')
            print('Model will be trained with the following parameters: ')
            for key in params.keys():
                print(key, ':', params[key])
            print('-------------------')

            # Call training (this includes train-test split, training of the model, saving of the trained model and evaluation with the test data)
            print('-------------------')
            print('Starting train_test of MultiGrooD...')
            model = train_eval_MultiGrooD(pb, props, params, args.output, args.threads, args.norm)
            print('Training & evaluation done.')


    # MODE 2: inference only
    elif args.mode == 'inference':
        print('Starting inference mode...')

        # Create prediction directory
        inference_dir = create_inference_dir(args.output)

        # Load bulk data
        bulk, props = load_inference_data(args)

        # Inference
        model, pred, mode, bulk_processed = inference_grood_models(args.model_path, bulk, inference_dir)

        # Visualization of inference result
        print('Visualizing deconvolution results...')
        visualize_predict(pred, inference_dir)

        # Generate explaining prediction heatmaps
        if mode != "multigrood":
            get_explain_heatmap(model, pred.columns.tolist(), bulk_processed, pred, inference_dir)

        # Evaluation
        if props is None:
            print('No evalution of inferred results can be performed, as no (appropriate) proportions are supplied.')
        else:
            print('Comparing results to ground-truth...')
            masterTable = eval_inference(pred, props, inference_dir)
            masterTable.to_csv(inference_dir + 'MasterTable.csv')

    # MODE 3: train, test & inference
    elif args.mode == 'all':

        print('-------------------')
        print(f"{args.mode} mode selected.")
        print('-------------------')


        # Create prediction directory
        inference_dir = create_inference_dir(args.output)

        pb, pb_props, bulk, props = load_all_data(args)


        # Access either GrooD, XGrooD or MultiGrooD architecture for training
        if args.grood_mode == 'grood':

            print('-------------------')
            print('Selected architecture: GrooD')
            print('-------------------')

            # Construct parameters for GrooD model training
            params = {'max_depth' : args.depth,
                    'n_estimators' : args.n_estimators,
                    'learning_rate' : args.learning_rate,
                    'loss' : args.loss_function,
                    'min_samples_split' : args.min_samples_split
            }

            print('-------------------')
            print('Model will be trained with the following parameters: ')
            for key in params.keys():
                print(key, ':', params[key])
            print('-------------------')

            # Call training (this includes train-test split, training of the model, saving of the trained model and evaluation with the test data)
            print('-------------------')
            print('Starting train_test of GrooD...')
            model = train_eval_GrooD(pb, pb_props, params, args.output, args.threads, args.norm)
            print('Training done. Evaluating now...')
        
        elif args.grood_mode == 'xgrood':

            print('-------------------')
            print('Selected architecture: XGrooD')
            print('-------------------')

            if args.loss_function == 'squared_error':
                lf = "reg:squarederror"
            elif args.loss_function == 'absolute_error':
                lf = "reg:absoluteerror"

            # Construct parameters for MultiGrooD model training
            params = {'max_depth' : args.depth,
                    'n_estimators' : args.n_estimators,
                    'learning_rate' : args.learning_rate,
                    'objective' : lf,
                    'min_child_weight' : args.min_samples_split
            }

            print('-------------------')
            print('Model will be trained with the following parameters: ')
            for key in params.keys():
                print(key, ':', params[key])
            print('-------------------')

            # Call training (this includes train-test split, training of the model, saving of the trained model and evaluation with the test data)
            print('-------------------')
            print('Starting train_test of XGrooD...')
            model = train_eval_XGrooD(pb, pb_props, params, args.output, args.threads, args.norm)
            print('Training & evaluation done.')

        elif args.grood_mode == 'multigrood':

            print('-------------------')
            print('Selected architecture: MultiGrooD')
            print('-------------------')


            # Construct parameters for MultiGrooD model training
            params = {'max_depth' : args.depth,
                    'n_estimators' : args.n_estimators,
            }

            print('-------------------')
            print('Model will be trained with the following parameters: ')
            for key in params.keys():
                print(key, ':', params[key])
            print('-------------------')

            # Call training (this includes train-test split, training of the model, saving of the trained model and evaluation with the test data)
            print('-------------------')
            print('Starting train_test of MultiGrooD...')
            model = train_eval_MultiGrooD(pb, pb_props, params, args.output, args.threads, args.norm)
            print('Training & evaluation done.')

        _, pred = inference_loaded_grood(model, bulk, inference_dir)

        # Visualization of inference result
        print('Visualizing deconvolution results...')
        visualize_predict(pred, inference_dir)

        # Visualization of explain heatmap only for GrooD and XGrooD
        if args.grood_mode != "multigrood":
            get_explain_heatmap(model["model"], pred.columns.tolist(), bulk, pred, inference_dir)

        # Evaluation
        if props is None:
            print('No evalution of inferred results can be performed, as no (appropriate) proportions are supplied.')
        else:
            print('Comparing results to ground-truth...')
            masterTable = eval_inference(pred, props, inference_dir)
            masterTable.to_csv(inference_dir + 'MasterTable.csv')


    else:
        raise ValueError('Please supply correct mode. Select from train-test or inference.')


if __name__ == '__main__':
    args = parse_args()
    main()


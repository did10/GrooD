# Import packages

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import ensemble
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import train_test_split
from sklearn.utils.fixes import parse_version
from sklearn.multioutput import MultiOutputRegressor
from typing import Dict, List, Tuple
import xgboost as xgb
import joblib
from src.evaluation import compare_data, plot_feature_importance, getCorr, annotated_heatmap, visualize_predict
from src.tools import main_gene_selection, bulk_norm, create_train_dir


# Train and evaluated GrooD

def train_eval_GrooD(pb, props, params, output, threads, norm):

    """
    Train and evaluate GrooD model
    pb: simualted pseudobulks with genes in columns
    props: cell type proportions associated with pb
    params: parameters for setting up model
    output: base output directory
    threads: number of threads for MultiOutputRegressort
    norm: way how pseudobulks have been normalized, will be stored in annotated model

    Returns annotated model
    """

    # Create training directory
    train_path, model_path = create_train_dir(output)

    #### Training ####

    # Train test split
    X_train, X_test, Y_train, Y_test = train_test_split(pb, props, test_size=0.2, random_state=20)

    # Initialize model
    multi_reg_model = MultiOutputRegressor(ensemble.GradientBoostingRegressor(**params), n_jobs=threads)
    
    # Train model
    multi_reg_model.fit(X_train, Y_train)

    # Add metadata to model
    metadata = {"estimators" : Y_train.columns.tolist(), # estimators correspond to the cell types, one regressor per cell type
                "model_type" : "grood",
                "norm" : norm}
    
    annotated_model = {"metadata" : metadata,
                       "model" : multi_reg_model}
    
    # Save model
    joblib.dump(annotated_model, f"{model_path}Model.pkl")

    #### Evaluation ####

    # Evaluate with test data
    pred = multi_reg_model.predict(X_test)
    pred = rescale_pred(pred)
    pred = pd.DataFrame(data=pred,
                        # columns=['Pred ' + str(x) for x in Y_test.columns.tolist()],
                        columns=['Pred ' + str(x) for x in metadata['estimators']],
                        index=X_test.index.tolist())

    # Compute overall mean_squared error
    mse = mean_squared_error(np.array(Y_test), np.array(pred))
    print('------------------------------------')
    print("Total mean squared error: ", str(mse))
    print('------------------------------------')

    # Generate MasterTable for plotting
    masterTable = pd.concat([Y_test, pred], axis = 1, ignore_index=True)
    groundTruthSamples = Y_test.columns.tolist()
    predSamples = pred.columns.tolist()
    masterTable.columns = groundTruthSamples + predSamples
    masterTable.to_csv(train_path + 'MasterTable.csv')

    # Correlation parameters for all cell types
    correlation_table = getCorr(masterTable, Y_test.columns.tolist())
    correlation_table.to_csv(train_path + 'Evaluated_training_result.csv')
    df_plot = correlation_table.transpose()
    annotated_heatmap(df_plot, train_path)

    # Generate regression plots
    compare_data(Y_test, pred, masterTable, train_path, 'regression')

    # Generate response plots
    compare_data(Y_test, pred, masterTable, train_path, 'response')

    # Generate error plots
    compare_data(Y_test, pred, masterTable, train_path, 'error')

    # Analyse feature importance
    plot_feature_importance(multi_reg_model, metadata["estimators"], train_path)

    # Visualize prediction
    visualize_predict(pred, train_path)

    return annotated_model


# Train and evaluated XGrooD

def train_eval_XGrooD(pb, props, params, output, threads, norm):

    """
    Train and evaluate XGrooD model
    pb: simualted pseudobulks with genes in columns
    props: cell type proportions associated with pb
    params: parameters for setting up model
    output: base output directory
    threads: number of threads for MultiOutputRegressort
    norm: way how pseudobulks have been normalized, will be stored in annotated model

    Returns annotated model
    """

    # Create training directory
    train_path, model_path = create_train_dir(output)

    #### Training ####

    # Train test split
    X_train, X_test, Y_train, Y_test = train_test_split(pb, props, test_size=0.2, random_state=20)

    # Initialize model
    multi_reg_model = MultiOutputRegressor(xgb.XGBRegressor(**params), n_jobs=threads)
    
    # Train model
    multi_reg_model.fit(X_train, Y_train)

    # Add metadata to model
    metadata = {"estimators" : Y_train.columns.tolist(), # estimators correspond to the cell types, one regressor per cell type
                "model_type" : "xgrood",
                "norm" : norm}
    
    annotated_model = {"metadata" : metadata,
                       "model" : multi_reg_model}
    
    # Save model
    joblib.dump(annotated_model, f"{model_path}Model.pkl")

    #### Evaluation ####

    # Evaluate with test data
    pred = multi_reg_model.predict(X_test)
    pred = rescale_pred(pred)
    pred = pd.DataFrame(data=pred,
                        columns=['Pred ' + str(x) for x in Y_test.columns.tolist()],
                        index=X_test.index.tolist())

    # Compute overall mean_squared error
    mse = mean_squared_error(np.array(Y_test), np.array(pred))
    print('------------------------------------')
    print("Total mean squared error: ", str(mse))
    print('------------------------------------')

    # Generate MasterTable for plotting
    masterTable = pd.concat([Y_test, pred], axis = 1, ignore_index=True)
    groundTruthSamples = Y_test.columns.tolist()
    predSamples = pred.columns.tolist()
    masterTable.columns = groundTruthSamples + predSamples
    masterTable.to_csv(train_path + 'MasterTable.csv')

    # Correlation parameters for all cell types
    correlation_table = getCorr(masterTable, Y_test.columns.tolist())
    correlation_table.to_csv(train_path + 'Evaluated_training_result.csv')
    df_plot = correlation_table.transpose()
    annotated_heatmap(df_plot, train_path)

    # Generate regression plots
    compare_data(Y_test, pred, masterTable, train_path, 'regression')

    # Generate response plots
    compare_data(Y_test, pred, masterTable, train_path, 'response')

    # Generate error plots
    compare_data(Y_test, pred, masterTable, train_path, 'error')

    # Analyse feature importance
    plot_feature_importance(multi_reg_model, metadata["estimators"], train_path)

    # Visualize prediction
    visualize_predict(pred, train_path)

    return annotated_model


# MultiGrooD

def train_eval_MultiGrooD(pb, props, params, output, threads, norm) -> None:

    """
    Train and evaluate MultiGrooD model
    pb: simulated pseudobulks with genes in columns
    props: cell type proportions associated with pb
    params: parameters for setting up model
    output: base output directory
    threads: number of threads for MultiOutputRegressor
    norm: way how pseudobulks have been normalized, will be stored in annotated model

    Returns annotated model (with norm type and grood_mode in metadata)
    """

    # Create training directory
    train_path, model_path = create_train_dir(output)

    def deconv_loss(predt: np.ndarray, dtrain: xgb.DMatrix):
    
        y = dtrain.get_label().reshape(predt.shape)

        sum_preds = np.sum(predt, axis=1)
        penalty = np.maximum(0, sum_preds - 1)[:, None]
        grad = 2 * (predt - y) + 2 * penalty

        hess = 2 * np.ones_like(predt)

        return grad, hess

    def rmse(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[str, float]:

        y = dtrain.get_label().reshape(predt.shape)
        v = np.sqrt(np.sum(np.power(y - predt, 2)))

        return "PyRMSE", v
    
    X_train, X_test, Y_train, Y_test = train_test_split(pb, props, test_size=0.2, random_state=20)

    Xy = xgb.DMatrix(X_train, Y_train)
    results: Dict[str, Dict[str, List[float]]] = {}

    booster = xgb.train(
        {
            "tree_method": "hist",
            "num_target": Y_train.shape[1],
            "multi_strategy": "multi_output_tree",
            "max_depth": params["max_depth"],
            "n_jobs" : threads,
        },
        dtrain=Xy,
        num_boost_round=params["n_estimators"],
        obj=deconv_loss,
        evals=[(Xy, "Train")],
        evals_result=results,
        custom_metric=rmse,
    )

    # Add metadata to model
    metadata = {"estimators" : Y_train.columns.tolist(), # estimators correspond to the cell types predicted by the model
                "model_type" : "multigrood",
                "norm" : norm}
    
    annotated_model = {"metadata" : metadata,
                       "model" : booster}
    
    # Save model
    joblib.dump(annotated_model, f"{model_path}Model.pkl")
    
    #### Evaluation ####

    # Evaluate with test data
    pred = booster.inplace_predict(X_test)
    pred = pd.DataFrame(data=pred,
                        columns=['Pred ' + str(x) for x in props.columns.tolist()],
                        index=X_test.index.tolist())

    # Compute overall mean_squared error
    mse = mean_squared_error(np.array(Y_test), np.array(pred))
    print('------------------------------------')
    print("Total mean squared error: ", str(mse))
    print('------------------------------------')

    # Generate MasterTable for plotting
    masterTable = pd.concat([Y_test, pred], axis = 1, ignore_index=True)
    groundTruthSamples = Y_test.columns.tolist()
    predSamples = pred.columns.tolist()
    masterTable.columns = groundTruthSamples + predSamples
    masterTable.to_csv(train_path + 'MasterTable.csv')

    # Correlation parameters for all cell types
    correlation_table = getCorr(masterTable, Y_test.columns.tolist())
    correlation_table.to_csv(train_path + 'Evaluated_training_result.csv')
    df_plot = correlation_table.transpose()
    annotated_heatmap(df_plot, train_path)

    # Generate regression plots
    compare_data(Y_test, pred, masterTable, train_path, 'regression')

    # Generate response plots
    compare_data(Y_test, pred, masterTable, train_path, 'response')

    # Generate error plots
    compare_data(Y_test, pred, masterTable, train_path, 'error')

    # Analyse feature importance
    # plot_feature_importance(booster, metadata["estimators"], train_path) #TODO: add function for feature importances for booster

    # Visualize prediction
    visualize_predict(pred, train_path)

    return annotated_model


# Inference of cell type proportions using saved model (for inference mode)

def inference_grood_models(model_path, bulk, output):

    """
    Loads a model from train_test mode
    Extracts metadata to select correct GrooD mode for model
    Subsets bulk data to genes model has been trained on and normalizes according to saved normalization in model
    Outputs model and most importantly predictions
    """

    # Load model from path
    annotated_model = joblib.load(model_path)
    model = annotated_model["model"]
    metadata = annotated_model["metadata"]
    norm = metadata["norm"] # get info how training data was normalized prior to training for bulk normalization: rank, CPM (on TPM norm data after gene selection), log
    cellTypes = metadata["estimators"] # Get cellTypes for which model has been trained
    model_type = metadata["model_type"] # get model_type: grood, xgrood or multigrood

    if model_type == "grood":
        # Subset bulk data to same genes as in model
        genes = model.estimators_[0].feature_names_in_.tolist()
        bulk, to_fill_columns, _ = main_gene_selection(bulk, genes) # Select all genes from training data. Genes will be filled with 0, if genes do not exist in bulk data.
        
        # Add normalization
        bulk = bulk_norm(bulk, norm)
        if len(to_fill_columns) > 0:
            print(f'Not all genes required in bulk data. {str(len(to_fill_columns))} genes are added and filled with zeros.')

        # Inference
        pred = model.predict(bulk)
        pred = rescale_pred(pred)

        # Evaluate with test data
        pred = pd.DataFrame(data=pred)
        pred.columns = cellTypes
        pred.index = bulk.index.tolist()
        
        pred.to_csv(output + 'Predicted_cell_type_proportions.csv')

    elif model_type == "xgrood":
        # Subset bulk data to same genes as in model
        genes = model.estimators_[0].feature_names_in_.tolist()
        bulk, to_fill_columns, _ = main_gene_selection(bulk, genes) # Select all genes from training data. Genes will be filled with 0, if genes do not exist in bulk data.
        
        # Add normalization
        bulk = bulk_norm(bulk, norm)
        if len(to_fill_columns) > 0:
            print(f'Not all genes required in bulk data. {str(len(to_fill_columns))} genes are added and filled with zeros.')

        # Inference
        pred = model.predict(bulk)
        pred = rescale_pred(pred)

        # Evaluate with test data
        pred = pd.DataFrame(data=pred)
        pred.columns = cellTypes
        pred.index = bulk.index.tolist()
        
        pred.to_csv(output + 'Predicted_cell_type_proportions.csv')

    elif model_type == "multigrood":
        # Subset bulk data to same genes as in model
        genes = model.feature_names
        bulk, to_fill_columns, _ = main_gene_selection(bulk, genes) # Select all genes from training data. Genes will be filled with 0, if genes do not exist in bulk data.
        
        # Add normalization
        bulk = bulk_norm(bulk, norm)
        if len(to_fill_columns) > 0:
            print(f'Not all genes required in bulk data. {str(len(to_fill_columns))} genes are added and filled with zeros.')

        # Inference
        pred = model.inplace_predict(bulk)

        # Evaluate with test data
        pred = pd.DataFrame(data=pred)
        pred.columns = cellTypes
        pred.index = bulk.index.tolist()
        
        pred.to_csv(output + 'Predicted_cell_type_proportions.csv')

    return model, pred, model_type, bulk


# Inference of cell type proportions using saved model (for inference mode)

def inference_loaded_grood(annotated_model, bulk, output):

    """
    If a GrooD model is already loaded, the metadata info is extracted and deconvolution performed with respective model on bulk data
    """

    # Load model from path
    model = annotated_model["model"]
    metadata = annotated_model["metadata"]
    norm = metadata["norm"] # get info how training data was normalized prior to training for bulk normalization: rank, CPM (on TPM norm data after gene selection), log
    cellTypes = metadata["estimators"] # Get cellTypes for which model has been trained
    model_type = metadata["model_type"] # get model_type: grood, xgrood or multigrood

    if model_type == "grood":

        # Inference
        pred = model.predict(bulk)
        pred = rescale_pred(pred)

        # Evaluate with test data
        pred = pd.DataFrame(data=pred)
        pred.columns = cellTypes
        pred.index = bulk.index.tolist()
        
        pred.to_csv(output + 'Predicted_cell_type_proportions.csv')

    elif model_type == "xgrood":

        # Inference
        pred = model.predict(bulk)
        pred = rescale_pred(pred)

        # Evaluate with test data
        pred = pd.DataFrame(data=pred)
        pred.columns = cellTypes
        pred.index = bulk.index.tolist()
        
        pred.to_csv(output + 'Predicted_cell_type_proportions.csv')

    elif model_type == "multigrood":

        # Inference
        pred = model.inplace_predict(bulk)

        # Evaluate with test data
        pred = pd.DataFrame(data=pred)
        pred.columns = cellTypes
        pred.index = bulk.index.tolist()
        
        pred.to_csv(output + 'Predicted_cell_type_proportions.csv')

    return model, pred


# Evaluate inference

def eval_inference(pred, groundTruth, output):

    """
    Evaluates inferred proportions given ground truth for comparison
    
    Compute overall MSE across all samples
    Compile MasterTable and calculate QC metrics
    Plot QC metrics in annotated heatmap
    Create regression, error and response plot
    """

    groundTruth.sort_index(axis=1, inplace=True)
    pred.sort_index(axis=1, inplace=True)
    pred.columns = ['Pred ' + str(x) for x in pred.columns.tolist()]

    # Compute overall mean_squared error
    x = groundTruth.to_numpy()
    y = pred.to_numpy()
    mse = mean_squared_error(x, y)
    print("Total mean squared error: ", str(mse))

    # Generate MasterTable for plotting
    masterTable = pd.concat([groundTruth, pred], axis = 1, ignore_index=True)
    groundTruthSamples = groundTruth.columns.tolist()
    predSamples = pred.columns.tolist()
    samples = groundTruthSamples + predSamples
    masterTable.columns = samples

    # Correlation parameters for all cell types
    correlation_table = getCorr(masterTable, groundTruth.columns.tolist())
    correlation_table.to_csv(output + 'Evaluated_inference_result.csv')
    df_plot = correlation_table.transpose()
    annotated_heatmap(df_plot, output)

    # Generate regression plots
    compare_data(groundTruth, pred, masterTable, output, 'regression')

    # Generate response plots
    compare_data(groundTruth, pred, masterTable, output, 'response')

    # Generate error plots
    compare_data(groundTruth, pred, masterTable, output, 'error')

    return masterTable


# Rescaling predition to a sum of 100 % of all cell types

def rescale_pred(pred):

    """
    Rescale a prediction for a sample to sum of 1 only, if sum > 1
    """

    rescaled_pred = pred
    for row in range(pred.shape[0]):

        prop_sum = np.sum(pred[row,:])

        if prop_sum > 1:
            rescaled_pred[row,:] = pred[row,:] * (1/prop_sum)

    return rescaled_pred


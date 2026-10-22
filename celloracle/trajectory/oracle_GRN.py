# -*- coding: utf-8 -*-

from copy import deepcopy
import warnings
import logging
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import pandas as pd
import os

from sklearn.linear_model import Ridge
from tqdm.auto import tqdm

from ..utility import intersect
from .. import _accelerator as _acc
#from network_fitting import TransNet as tn


# Minimum number of floating point operations per propagation step for the
# GPU path to be worthwhile (n_cells * n_genes * n_genes).
_SIMULATION_GPU_MIN_FLOPS = 5e7


def _do_simulation_gpu(coef_matrix, simulation_input, gem, n_propagation):
    """Signal-propagation loop executed on the Apple GPU via MLX.

    The whole iteration is kept on the device: the coefficient matrix, the
    starting expression matrix and the update mask are transferred once, then
    ``n_propagation`` dense products are evaluated before a single transfer
    back to the host.  MLX/Metal uses ``float32``.
    """
    mx = _acc._import_mlx()

    delta_input_np = (simulation_input - gem).to_numpy(dtype=np.float32, copy=True)
    gem_np = gem.to_numpy(dtype=np.float32, copy=True)
    coef_np = np.ascontiguousarray(coef_matrix.to_numpy(dtype=np.float32))

    coef = _acc.to_gpu(coef_np)
    delta = _acc.to_gpu(delta_input_np)
    gem_gpu = _acc.to_gpu(gem_np)
    mask = _acc.to_gpu(delta_input_np != 0)
    delta_input_gpu = _acc.to_gpu(delta_input_np)

    for _ in range(n_propagation):
        delta = mx.matmul(delta, coef)
        delta = mx.where(mask, delta_input_gpu, delta)
        gem_tmp = mx.maximum(gem_gpu + delta, mx.array(0.0, dtype=mx.float32))
        delta = gem_tmp - gem_gpu

    gem_simulated = _acc.to_numpy(gem_gpu + delta).astype(np.float64, copy=False)
    return pd.DataFrame(gem_simulated, index=gem.index, columns=gem.columns)


def _do_simulation(coef_matrix, simulation_input, gem, n_propagation):
    """
    Silulate signal propagation in GRNs.

    Args:
        coef_matrix (pandas.DataFrame): 2d matrix that store GRN weights

        simulation_input (pandas.DataFrame): input for simulation

        gem (pandas.DataFrame): input for simulation

        n_propagation (int): number of propagation.

    Returns:
        pandas.DataFrame: simulated gene expression matrix after signal progagation
    """
    n_cells, n_genes = gem.shape
    if (n_propagation > 0
            and _acc.get_backend() == "gpu"
            and float(n_cells) * float(n_genes) * float(coef_matrix.shape[1]) >= _SIMULATION_GPU_MIN_FLOPS):
        try:
            return _do_simulation_gpu(coef_matrix, simulation_input, gem, n_propagation)
        except Exception as exc:  # pragma: no cover - fall back to CPU
            warnings.warn(f"GPU simulation failed ({type(exc).__name__}: {exc}); falling back to CPU.")

    delta_input = simulation_input - gem

    delta_simulated = delta_input.copy()
    for i in range(n_propagation):
        delta_simulated = delta_simulated.dot(coef_matrix)
        delta_simulated[delta_input != 0] = delta_input

        # gene expression cannot be negative. adjust delta values to make sure that gene expression are not netavive values.
        gem_tmp = gem + delta_simulated
        gem_tmp[gem_tmp<0] = 0
        delta_simulated = gem_tmp - gem

    gem_simulated = gem + delta_simulated

    return gem_simulated


def _getCoefMatrix(gem, TFdict, alpha=1, verbose=True):
    """
    Calculate GRN and return CoefMatrix (network weights)

    Args:
        gem (pandas.DataFrame): gene expression matrix to calculate GRN

        TFdict (dictionary): python dictionary of potential regulatory gene list

        alpha (float) : strength of regularization in Ridge.

    Returns:
        2d numpy array: numpy array
    """

    genes = gem.columns

    all_genes_in_dict = intersect(gem.columns, list(TFdict.keys()))
    zero_ = pd.Series(np.zeros(len(genes)), index=genes)

    def get_coef(target_gene):
        tmp = zero_.copy()

        # define regGenes
        reggenes = TFdict[target_gene]
        reggenes = intersect(reggenes, genes)

        if target_gene in reggenes:
            reggenes.remove(target_gene)
        if len(reggenes) == 0 :
            tmp[target_gene] = 0
            return(tmp)
        # prepare learning data
        Data = gem[reggenes]
        Label = gem[target_gene]
        # model fitting
        model = Ridge(alpha=alpha, random_state=123)
        model.fit(Data, Label)
        tmp[reggenes] = model.coef_

        return tmp

    li = []
    li_calculated = []
    with tqdm(genes, disable=(verbose==False)) as pbar:
        for i in pbar:
            if not i in all_genes_in_dict:
                tmp = zero_.copy()
                tmp[i] = 0
            else:
                tmp = get_coef(i)
                li_calculated.append(i)
            li.append(tmp)
    coef_matrix = pd.concat(li, axis=1)
    coef_matrix.columns = genes

    if verbose:
        print(f"genes_in_gem: {gem.shape[1]}")
        print(f"models made for {len(li_calculated)} genes")

    return coef_matrix #, li_calculated

def _shuffle_celloracle_GRN_coef_table(coef_dataframe, random_seed=123):
    """
    Shuffle grn coef table. Target genes were shuffled.
    """

    if coef_dataframe.values.max() == 1:
        _correct_coef_table(coef_dataframe=coef_dataframe)
    values = coef_dataframe.values.copy()

    np.random.seed(random_seed)
    random_index = np.arange(values.shape[1])
    np.random.shuffle(random_index)
    random_df = pd.DataFrame(values[:, random_index],
                            index=coef_dataframe.index,
                            columns=coef_dataframe.columns)
    return random_df

def _correct_coef_table(coef_dataframe):
    """
    Delete self regulating edge from coef table.
    """
    for i in range(coef_dataframe.shape[0]):
        coef_dataframe.iloc[i, i] = 0



def _coef_to_active_gene_list(coef_matrix):
    """
    Args:
        coef_matricx (pd.DataFrame): 2d dataframe (gene x gene). This is a result of GRN calculation.

    Return:
        list: list of active gene that have at least one target gene.
    """
    coef = coef_matrix.copy()

    active_TF_list = []
    for i in coef.columns.values:
        coef.loc[i, i] = 0
        if np.any(coef.loc[i, :] != 0):
            active_TF_list.append(i)

    return active_TF_list

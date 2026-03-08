from itertools import chain
import sys 

import anndata as ad 
from typing import Any 
import types
import torch
import torch.nn as nn
import scanpy as sc
import scglue
import matplotlib.pyplot as plt
import networkx as nx
import anndata as ad
import itertools
import networkx as nx
import pandas as pd
import scanpy as sc
import scglue
import seaborn as sns
from matplotlib import rcParams
import numpy as np
import gzip
import re
import scipy
import pyranges as pr
from scipy import sparse
import torch.nn.functional as F

from util.model import PriorGuidedTransformer








def build_cell_gene_graph(adata):
    data = HeteroData()
    data['cell'].num_nodes = adata.n_obs
    data['gene'].num_nodes = adata.n_vars

    row, col = adata.X.nonzero()
    edge_index = torch.tensor([row, col], dtype=torch.long)
    data['cell', 'expresses', 'gene'].edge_index = edge_index
    data['gene', 'rev_express', 'cell'].edge_index = edge_index.flip(0)

    data['cell'].x = torch.ones(adata.n_obs, 1)
    data['gene'].x = torch.ones(adata.n_vars, 1)
    return data


def build_cell_peak_graph(adata):
    data = HeteroData()
    data['cell'].num_nodes = adata.n_obs
    data['peak'].num_nodes = adata.n_vars

    row, col = adata.X.nonzero()
    edge_index = torch.tensor([row, col], dtype=torch.long)
    data['cell', 'accesses', 'peak'].edge_index = edge_index
    data['peak', 'rev_access', 'cell'].edge_index = edge_index.flip(0)

    data['cell'].x = torch.ones(adata.n_obs, 1)
    data['peak'].x = torch.ones(adata.n_vars, 1)
    return data
    
class Prior():
    def __init__(self, config):
        self.config = config
        self.rna_adata = None
        self.atac_adata = None
        self.guidance_graph = None

    def Prior_graph(self, rna_emb, atac_emb):
        # 1. Load
        rna_adata = sc.read_h5ad(f'{self.rna_h5ad_path[0]}')
        atac_adata = sc.read_h5ad(f'{self.atac_h5ad_path[0]}')

        outfile = self.gtf_path[0]


        gtf = pd.read_csv(
            outfile,
            sep="\t",
            comment="#",
            header=None,
            names=["seqname","source","feature","start","end","score","strand","frame","attribute"]
        )
        genes = gtf[gtf["feature"]=="gene"].copy()
        def get_gene_id(attr):
            m = re.search('gene_id "([^"]+)"', attr)
            return m.group(1) if m else None
        genes["gene_id"] = genes["attribute"].apply(get_gene_id)
        gene_coords = genes.set_index("gene_id")[["seqname","start","end","strand"]]


        print(rna_adata.var.columns) 
        print(rna_adata.var.head()) 

        print(atac_adata.var.columns) 
        print(atac_adata.var.head())


        #X to int
        if sparse.issparse(rna_adata.X):
            rna_adata.X.data = np.round(rna_adata.X.data).clip(0).astype(np.int32)
        else:
            rna_adata.X = np.round(rna_adata.X).clip(0).astype(np.int32)
        
        if sparse.issparse(atac_adata.X):
            atac_adata.X.data = np.round(atac_adata.X.data).clip(0).astype(np.int32)
        else:
            atac_adata.X = np.round(atac_adata.X).clip(0).astype(np.int32)  


        #atac_adata.X = atac_adata.X.astype(np.float64)
        

        
        print(rna_adata)
        print(atac_adata)

        rna_adata.layers["counts"] = rna_adata.X.copy() 
        sc.pp.highly_variable_genes(rna_adata, n_top_genes=2000, flavor="seurat_v3")
        sc.pp.normalize_total(rna_adata)
        sc.pp.log1p(rna_adata)
        sc.pp.scale(rna_adata)
        sc.tl.pca(rna_adata, n_comps=100, svd_solver="auto")

        sc.pp.neighbors(rna_adata, metric="cosine")
        sc.tl.umap(rna_adata)
        sc.pl.umap(rna_adata, color="cell_type",show=False)
        plt.savefig(f"rna_row.jpg", format="jpg", dpi=400,transparent=True,bbox_inches='tight')


        scglue.data.lsi(atac_adata, n_components=100, n_iter=15)
        sc.pp.neighbors(atac_adata, use_rep="X_lsi", metric="cosine")
        sc.tl.umap(atac_adata)
        sc.pl.umap(atac_adata, color="cell_type",show=False)
        plt.savefig(f"atac_row.jpg", format="jpg", dpi=400,transparent=True,bbox_inches='tight')


        gtf = pr.read_gtf(f"{outfile}")
        genes["gene_id"] = genes["gene_id"].str.split(".").str[0]
        rna_adata.var["gene_name"] = rna_adata.var.index
        scglue.data.get_gene_annotation(
            rna_adata, gtf=outfile,
            gtf_by="gene_name"
        )

        atac_adata.var["gene_name"] = atac_adata.var.index
        scglue.data.get_gene_annotation(
            atac_adata, gtf=outfile,
            gtf_by="gene_name"
        )

        print(atac_adata)
        print(rna_adata)


        #clear nan
        rna_adata = rna_adata[ :,~rna_adata.var["chrom"].isna()].copy()


        missing_strand = rna_adata.var["strand"].isna()

        rna_adata.var.loc[missing_strand, "strand"] = "+"
        print(rna_adata.var[["chrom", "chromStart", "chromEnd", "strand"]].isna().sum())
        print(rna_adata.var.loc[:, ["chrom", "chromStart", "chromEnd"]].head())

        atac_adata = atac_adata[ :,~atac_adata.var["chrom"].isna()].copy()


        missing_strand = atac_adata.var["strand"].isna()

        atac_adata.var.loc[missing_strand, "strand"] = "+"
        print(atac_adata.var[["chrom", "chromStart", "chromEnd", "strand"]].isna().sum())
        print(atac_adata.var.loc[:, ["chrom", "chromStart", "chromEnd"]].head())

        print( rna_adata.var["strand"].unique()) 
        print( rna_adata.var["strand"].isna().sum())

        print(atac_adata.var.loc[:, ["chrom", "chromStart", "chromEnd"]].head())



        rna_adata.var_names = rna_adata.var["gene_name"]

        atac_adata.var_names = atac_adata.var["gene_name"]
    

        guidance = scglue.genomics.rna_anchored_guidance_graph(rna_adata, atac_adata)
        scglue.graph.check_graph(guidance, [rna_adata, atac_adata])

        if "artif_dupl" in rna_adata.var.columns:
            rna_adata.var.drop(columns=["artif_dupl"], inplace=True)
            
        if "artif_dupl" in atac_adata.var.columns:
            atac_adata.var.drop(columns=["artif_dupl"], inplace=True)

        nx.write_graphml(guidance, "guidance.graphml.gz")

class PriorMask():
    def __init__(self,config, transformer_model, guidance_path, device):
        self.config = config
        
        rna_adata = sc.read_h5ad(f'{self.config.rna_h5ad_path[0]}')
        atac_adata = sc.read_h5ad(f'{self.config.atac_h5ad_path[0]}')
        rna_var_names=rna_adata.var_names
        atac_var_names=atac_adata.var_names

        self.transformer = transformer_model
        self.device = device
        
        print("Reading guidance graph and computing prior weights...")
        graph = nx.read_graphml(guidance_path)
        self.prior_weight = self._get_prior_weight(graph, rna_var_names, atac_var_names).to(device)

    def _get_prior_weight(self, graph, rna_names, atac_names):
        
        
        graph_nodes = set(graph.nodes())
    
        valid_rna = [n for n in rna_names if n in graph_nodes]
        valid_atac = [n for n in atac_names if n in graph_nodes]
        
        adj = nx.adjacency_matrix(graph, nodelist=valid_rna + valid_atac)
        n_rna = len(rna_names)

        cross_modal_adj = adj[:n_rna, n_rna:]

        weight = cross_modal_adj.sum() / (len(valid_rna) * len(valid_atac))
        return torch.tensor(weight).float()

    def get_refined_embeddings(self, rna_emb, atac_emb):

        return self.transformer(rna_emb, atac_emb)

    def train_step(self, rna_emb, atac_emb, optimizer):

        self.transformer.train()
        optimizer.zero_grad()


        rna_refined, atac_refined = self.transformer(rna_emb, atac_emb)


        cos_sim = F.cosine_similarity(rna_refined, atac_refined)
        

        loss_prior = (1.0 - cos_sim).mean() * self.prior_weight


        loss_prior.backward()
        optimizer.step()

        return loss_prior.item(), rna_refined, atac_refined

        
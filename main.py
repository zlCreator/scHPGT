import torch
import os
os.environ['OMP_NUM_THREADS'] = '1'
from datetime import datetime

import pandas as pd
import scanpy as sc
import os
import numpy as np

from config import Config
from util.train_recon import Recon
from util.train_adv import Adv
from util.Prior_mask import Prior, PriorMask
from util.preprocess import read_from_h5ad
from util.model import PriorGuidedTransformer

from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt



def main():    
    
    os.environ['OMP_NUM_THREADS'] = '1'
    torch.set_num_threads(1)
    
    config = Config()    
    device = config.device
    torch.manual_seed(config.seed)

    # Read data directly from h5ad files
    print('Reading data from h5ad files')
    rna_data, rna_labels, atac_data, atac_labels, input_size, combined_cat = read_from_h5ad(config.rna_h5ad_path[0], config.atac_h5ad_path[0], config.gtf_path[0])
    config.input_size = input_size
    config.type = len(combined_cat)
    
    # Recontruction training
    print('Reconstruction training')
    model_recon= Recon(config)
    for epoch in range(config.recon_ep):
        model_recon.train(epoch)
    rna_emb, _, atac_emb, _ = model_recon.get_embeddings()
    

    print('Get prior graph')
    Prior.Prior_graph(config, rna_emb, atac_emb)
    
    print('Mask training')
    hpgt_transformer = PriorGuidedTransformer().to(device)

    optimizer = torch.optim.Adam(
        list(hpgt_transformer.parameters()), 
        lr=1e-4
    )

    prior_manager = PriorMask(
        config,
        transformer_model=hpgt_transformer,
        guidance_path='guidance.graphml.gz',
        device=device
    )
    
    rna_tensor = torch.from_numpy(rna_data.toarray()).float()
    atac_tensor = torch.from_numpy(atac_data.toarray()).float()

    rna_dataset = TensorDataset(rna_tensor)
    atac_dataset = TensorDataset(atac_tensor)

    rna_loader = DataLoader(rna_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
    atac_loader = DataLoader(atac_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)

    
    if isinstance(rna_emb, np.ndarray):
        rna_emb = torch.from_numpy(rna_emb).to(device)
    if isinstance(atac_emb, np.ndarray):
        atac_emb = torch.from_numpy(atac_emb).to(device)
    for epoch in range(config.Prior_ep):
        for rna_batch, atac_batch in zip(rna_loader, atac_loader):

            _, rna_ref, atac_ref = prior_manager.train_step(rna_emb, atac_emb, optimizer)

    
    # Adversarial training
    print('Adversarial training')
    model_adv = Adv(config, rna_data=rna_ref, atac_data=atac_ref)
    for epoch in range(config.adv_ep):
       model_adv.train(epoch)
        
    rna_emb, rna_label, atac_emb, atac_label = model_adv.get_embeddings()


    output_dir = f'./output/'
    os.makedirs(output_dir, exist_ok=True)
    sc.settings.figdir = output_dir

    # Create anndata object with embeddings
    modality = ["RNA"] * len(rna_emb) + ["ATAC"] * len(atac_emb)
    
    adata_combined = sc.AnnData(np.vstack([rna_emb, atac_emb]))
    adata_combined.obs['source'] = modality
    
    
    
    
    # Get cell types
    rna_cell_types = rna_label
    atac_cell_types = atac_label

    rna_preds_idx = np.argmax(rna_cell_types, axis=1)
    atac_preds_idx = np.argmax(atac_cell_types, axis=1)
    
    all_preds_idx = np.concatenate([rna_preds_idx, atac_preds_idx])
    
    adata_combined.obs['cell_type_code'] = all_preds_idx  
    adata_combined.obs['cell_type'] = combined_cat[all_preds_idx]
    
    
    print(adata_combined)
    
    sc.pp.neighbors(adata_combined, use_rep='X')
    sc.tl.umap(adata_combined, min_dist=0.1)
    sc.pl.umap(adata_combined, color=['source','cell_type'],title=[''],wspace=0.3, legend_fontsize=10)

    
    # Save the combined anndata object as h5ad
    h5ad_output_path = os.path.join(output_dir, 'output.h5ad')
    adata_combined.write(h5ad_output_path)
    print(f'Combined embeddings saved to {h5ad_output_path}')
    

        

    
if __name__ == "__main__":
    main()

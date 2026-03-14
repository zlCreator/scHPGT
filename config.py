import torch
import os
import numpy as np
from scipy import sparse

class Config(object):
    def __init__(self):

        self.use_cuda = True

        if self.use_cuda:
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')

        self.name = 'PBMC3k'
        name = self.name
        # H5AD files paths
        self.rna_h5ad_path = [f'./{name}/RNA.h5ad']
        self.atac_h5ad_path = [f'./{name}/ATAC.h5ad']
        
        # Spices: "human" for gencode.v49.annotation.gtf,
        #         "mouse" for gencode.vM25.chr_patch_hapl_scaff.annotation.gtf
        self.gtf_path = [f'./gencode.v49.annotation.gtf']
        
        # Parameter settings         
        self.batch_size = 256
        self.seed = 1
        self.type = 0
        self.input_size = 0
        self.LRrecon = 0.005  
        self.Prior_edge = 20      
        self.Prior_ep = 20        
        self.emb_size = 64
        self.LRadv = 0.007 
        self.recon_ep = 20    
        self.adv_ep = 30  
        self.elpha = 0.8    
        self.beta = 5
        self.gamma = 0.7
        self.momentum = 0.9       
        self.use_cr = True


        

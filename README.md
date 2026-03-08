# scHPGT
Single-cell multi-omics technologies enable joint interrogation of gene expression and chromatin accessibility, yet integrating unpaired scRNA-seq and scATAC-seq remains challenging due to the modality gap, extreme sparsity, and batch effects. Here, we present scHPGT, a heterogeneous, prior-guided graph Transformer for robust integration of unpaired RNA--ATAC data. scHPGT combines modality-specific encoders with a prior-constrained cross-modal Transformer that leverages peak--gene regulatory links to guide attention, together with an adversarial alignment objective that reduces domain discrepancy while preserving biological structure. Across multiple benchmarks, including PBMC3k and mouse spleen, scHPGT consistently improves clustering and annotation accuracy and achieves strong cross-modality alignment while preserving biological separation. Comprehensive benchmarking analyses demonstrate that scHPGT provides an effective and interpretable solution for unpaired multi-omics integration across diverse and challenging scenarios.



![image](https://github.com/zlCreator/scHPGT/blob/main/Method.png)


# Examples
The required input files consist of scRNA-seq expression matrices in `.h5ad` format (with genes as features) and scATAC-seq chromatin accessibility matrices in `.h5ad` format (with peaks as features). We can configure the file paths and hyper-parameters via `config.py` and then execute the pipeline by running `main.py`.
## config.py
```python
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

        self.type = 0
        self.name = 'PBMC3k'
        name = self.name
        self.atac_labels = [] 
        # H5AD files paths
        self.rna_h5ad_path = [f'./{name}/RNA.h5ad']
        self.atac_h5ad_path = [f'./{name}/ATAC.h5ad']
        
        
        # Spices: "human" for gencode.v49.annotation.gtf,
        #         "mouse" for gencode.vM25.chr_patch_hapl_scaff.annotation.gtf
        self.gtf_path = [f'/home/czl/Datasets/gencode.v49.annotation.gtf']
        
        
        self.input_size = 0
        
        # Training config            
        self.batch_size = 256
        self.LRrecon = 0.007  
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
        self.seed = 1
        self.checkpoint = ''       
```


# Contact
24031212242@stu.xidian.edu.cn

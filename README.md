# scHPGT
Single-cell multi-omics technologies enable joint interrogation of gene expression and chromatin accessibility, yet integrating unpaired scRNA-seq and scATAC-seq remains challenging due to the modality gap, extreme sparsity, and batch effects. Here, we present scHPGT, a heterogeneous, prior-guided graph Transformer for robust integration of unpaired RNA--ATAC data. scHPGT combines modality-specific encoders with a prior-constrained cross-modal Transformer that leverages peak--gene regulatory links to guide attention, together with an adversarial alignment objective that reduces domain discrepancy while preserving biological structure. Across multiple benchmarks, including PBMC3k and mouse spleen, scHPGT consistently improves clustering and annotation accuracy and achieves strong cross-modality alignment while preserving biological separation. Comprehensive benchmarking analyses demonstrate that scHPGT provides an effective and interpretable solution for unpaired multi-omics integration across diverse and challenging scenarios.



[image](scHPGT/main/Method.png)

# Data
The required input files are scRNA-seq files in the `.h5ad` format and scATAC-seq files with gene activity matrices in the `.h5ad` format. Example files can be downloaded through the following links:
https://drive.google.com/drive/folders/1jHf1MnOtwjRPyy4XG3vPdfvxB52teq8A?usp=sharing

# Examples

```python
import anndata
from scMGCL import run
from evaluate_metrics import evaluate_model
import scanpy as sc


# load data
adata_atac = anndata.read_h5ad('ATAC.h5ad')
adata_rna = anndata.read_h5ad('RNA.h5ad')

adata_atac.obs['source'] = 'ATAC'
adata_rna.obs['source'] = 'RNA'


adata = adata_atac.concatenate(adata_rna,join='inner')

# annotation name
cell_type='cell_type'
adata.obs['cell_type']=adata.obs[f'{cell_type}']

# the scMGCL function is called to return the trained adata directly
integrated = run(adata)

RNA=integrated[integrated.obs['source'] == 'RNA']
ATAC=integrated[integrated.obs['source'] == 'ATAC']

z_rna=RNA.obsm['integrated_embeddings']
z_atac=ATAC.obsm['integrated_embeddings']

rna_cell_types=RNA.obs['cell_type']
atac_cell_types=ATAC.obs['cell_type']

results = evaluate_model(z_rna, z_atac, rna_cell_types, atac_cell_types, integrated)

# UMAP visualization
sc.set_figure_params(dpi=400, fontsize=10)
sc.pp.neighbors(integrated,use_rep='integrated_embeddings')
sc.tl.umap(integrated,min_dist=0.1)

sc.pl.umap(integrated, color=['source','cell_type'],title=['',''],wspace=0.3, legend_fontsize=10)

# save results
integrated.write('scMGCL_integrated.h5ad')
```

# Author
Zhenglong Cheng, Risheng Lu, Shixiong Zhang

School of Computer Science and Technology

Xidian University

Xi’an, Shaanxi 710071, China

# Contact
24031212242@stu.xidian.edu.cn

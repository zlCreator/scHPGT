import glob
import torch
import torch.utils.data as data
import numpy as np
import os
import os.path
import random
import csv
import scipy.sparse
import h5py
import episcanpy as epi
import scanpy as sc
import pandas as pd

from scipy import sparse
from config import Config

random.seed(1)



def h5ad_reader(file_name, modality='RNA'):

    adata = sc.read_h5ad(file_name)
    
    # Check and fix obs (cells)
    adata.obs_names_make_unique()
    
    # Check and fix var (genes/peaks)
    adata.var_names_make_unique()
    
    # Get data matrix
    X = adata.X
    if not scipy.sparse.issparse(X):
        X = scipy.sparse.csr_matrix(X)
    
    # Get labels if available
    labels = None
    if 'cell_type' in adata.obs:
        cell_types = adata.obs['cell_type'].astype('category')
        labels = cell_types.cat.codes.values
    
    return X, labels, adata
        



def read_from_h5ad(rna_h5ad_path, atac_h5ad_path, gtf_path):
    # Read RNA data from h5ad
    rna_data, rna_labels, rna_adata = h5ad_reader(rna_h5ad_path, 'RNA')
    
    # Read ATAC data from h5ad
    atac_data, atac_labels, adata = h5ad_reader(atac_h5ad_path, 'ATAC')
    atac_adata = epi.tl.geneactivity(
        adata,
        gtf_file=gtf_path,
        #field='gene_name',      
        #upstream=2000,         
        #downstream=0,
        feature_type='gene',
        #annotation_type='transcript'
    )
    
    atac_adata.obs = adata.obs.copy()
    # Check and fix obs (cells)
    atac_adata.obs_names_make_unique()
    
    # Check and fix var (genes/peaks)
    atac_adata.var_names_make_unique()
    
    atac_data = atac_adata.X
    
    c_g = rna_adata.var_names.intersection(atac_adata.var_names)
    
    rna_data = rna_data[:, rna_adata.var_names.isin(c_g)]
    atac_data = atac_data[:, atac_adata.var_names.isin(c_g)]
    
    if not scipy.sparse.issparse(rna_data):
        rna_data = scipy.sparse.csr_matrix(rna_data)
    if not scipy.sparse.issparse(atac_data):
        atac_data = scipy.sparse.csr_matrix(atac_data)

    all_types = pd.concat([rna_adata.obs['cell_type'], atac_adata.obs['cell_type']]).astype('category')
    combined_categories = all_types.cat.categories
    
    
    return rna_data, rna_labels, atac_data, atac_labels, len(c_g), combined_categories
    


class Dataloader(data.Dataset):
    def __init__(self, train = True, data_reader = None, labels = None, protein_reader = None):
        self.train = train        
        self.data_reader, self.labels, self.protein_reader = data_reader, labels, protein_reader
        self.input_size = self.data_reader.shape[1]
        self.sample_num = self.data_reader.shape[0]
        
        self.input_size_protein = None
        if protein_reader is not None:
            self.input_size_protein = self.protein_reader.shape[1]

    def __getitem__(self, index):
        if self.train:
            # get atac data            
            rand_idx = random.randint(0, self.sample_num - 1)
            sample = np.array(self.data_reader[rand_idx].todense())
            sample = sample.reshape((1, self.input_size))
            in_data = (sample>0).astype(float)  # binarize data
            
            if self.input_size_protein is not None:
                sample_protein = np.array(self.protein_reader[rand_idx].todense())
                sample_protein = sample_protein.reshape((1, self.input_size_protein))
                in_data = np.concatenate((in_data, sample_protein), 1)
                
            in_label = self.labels[rand_idx]
 
            return in_data, in_label

        else:
            sample = np.array(self.data_reader[index].todense())
            sample = sample.reshape((1, self.input_size))
            in_data = (sample>0).astype(float)  # binarize data

            if self.input_size_protein is not None:
                sample_protein = np.array(self.protein_reader[index].todense())
                sample_protein = sample_protein.reshape((1, self.input_size_protein))
                in_data = np.concatenate((in_data, sample_protein), 1)
                
            #in_data = in_data.reshape((1, self.input_size))
            in_label = self.labels[index]
 
            return in_data, in_label

    def __len__(self):
        return self.sample_num
                
              
class DataloaderWithoutLabel(data.Dataset):
    def __init__(self, train = True, data_reader = None, labels = None, protein_reader = None):
        self.train = train
        self.data_reader, self.labels, self.protein_reader = data_reader, labels, protein_reader
        self.input_size = self.data_reader.shape[1]
        self.sample_num = self.data_reader.shape[0]
        
        self.input_size_protein = None
        if protein_reader is not None:
            self.input_size_protein = self.protein_reader.shape[1]
            
            
    def __getitem__(self, index):
        if self.train:
            # get atac data
            rand_idx = random.randint(0, self.sample_num - 1)

            raw_data = self.data_reader[rand_idx]
            if sparse.issparse(raw_data):
                sample = np.array(raw_data.todense())
            else:
                sample = np.array(raw_data)
            sample = sample.reshape((1, self.input_size))
            in_data = (sample>0).astype(float)  # binarize data
            if self.input_size_protein is not None:
                sample_protein = np.array(self.protein_reader[rand_idx].todense())
                sample_protein = sample_protein.reshape((1, self.input_size_protein))
                in_data = np.concatenate((in_data, sample_protein), 1)
            #in_data = in_data.reshape((1, self.input_size)) 
            return in_data

        else:
            sample = np.array(self.data_reader[index].todense())
            sample = sample.reshape((1, self.input_size))
            in_data = (sample>0).astype(float)  # binarize data
            if self.input_size_protein is not None:
                sample_protein = np.array(self.protein_reader[index].todense())
                sample_protein = sample_protein.reshape((1, self.input_size_protein))
                in_data = np.concatenate((in_data, sample_protein), 1)
                
            #in_data = in_data.reshape((1, self.input_size)) 
            return in_data

    def __len__(self):
        return self.sample_num

                
                


class Load_data():
    def __init__(self, config):
        self.config = config
        # hardware constraint
        # hardware constraint
        num_workers = 0
        if num_workers < 0:
            num_workers = 0

        kwargs = {'num_workers': num_workers, 'pin_memory': False} 
        
        
        # load data
        train_rna_loaders = []
        test_rna_loaders = []
        train_atac_loaders = []
        test_atac_loaders = []
        self.num_of_atac = 0
        for rna_path, atac_path in zip(config.rna_h5ad_path, config.atac_h5ad_path):  
            rna_data, rna_labels, atac_data, atac_labels, input_size,_ = read_from_h5ad(rna_path, atac_path, config.gtf_path[0])
            # train loader 
            trainset = Dataloader(True, rna_data, rna_labels)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=
                            config.batch_size, shuffle=True, **kwargs)                        
            train_rna_loaders.append(trainloader)
            
            # test loader 
            trainset = Dataloader(False, rna_data, rna_labels)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=
                            config.batch_size, shuffle=False, **kwargs)                        
            test_rna_loaders.append(trainloader)


            # train loader
            trainset = DataloaderWithoutLabel(True, atac_data)
            self.num_of_atac += len(trainset)
            
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=
                            config.batch_size, shuffle=True, **kwargs)                        
            train_atac_loaders.append(trainloader)
            
            # test loader
            trainset = DataloaderWithoutLabel(False, atac_data)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=
                            config.batch_size, shuffle=False, **kwargs)                        
            test_atac_loaders.append(trainloader)

        self.train_rna_loaders = train_rna_loaders
        self.test_rna_loaders = test_rna_loaders
        self.train_atac_loaders = train_atac_loaders
        self.test_atac_loaders = test_atac_loaders
                    
        
    def getloader(self):
        return self.train_rna_loaders, self.test_rna_loaders, self.train_atac_loaders, self.test_atac_loaders, int(self.num_of_atac/self.config.batch_size)


if __name__ == "__main__":
    config = Config()
    rna_data = Dataloader(True, config.rna_paths[0], config.rna_labels[0])
    #print 'rna data:', rna_data.input_size, rna_data.input_size_protein, len(rna_data.data)
    
    atac_data = DataloaderWithoutLabel(True, config.atac_paths[0])
    #print 'atac data:', atac_data.input_size, atac_data.input_size_protein, len(atac_data.data)
    
    
    train_rna_loaders, test_rna_loaders, train_atac_loaders, test_atac_loaders = Load_data(config).getloader()
    print(len(train_rna_loaders), len(test_atac_loaders))
    
    print(len(train_rna_loaders[1]), len(train_atac_loaders[0]))

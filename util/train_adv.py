import torch
import torch.optim as optim
from torch.autograd import Variable
from itertools import cycle
from scipy.linalg import norm
from scipy.special import softmax
import numpy as np
import os

from util.dataload import Load_data
from util.model import HeteroEncoder, GRNModule
from util.loss import regu, PrioritizedLoss, HVGEncodLoss, AdvLoss



def inp(data_list, config):
    output = []
    for data in data_list:
        output.append(Variable(data).to(config.device))
    return output


def round(iterable):
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)


class Adv():
    def __init__(self, config, rna_data=None, rna_labels=None, atac_data=None):
        self.config = config
        
        # load data
        self.train_rna_loaders, self.test_rna_loaders, self.train_atac_loaders, self.test_atac_loaders, self.training_iters = Load_data(config).getloader()
        self.training_iteration = 0
        for atac_loader in self.train_atac_loaders:
            self.training_iteration += len(atac_loader)
        
        # initialize dataset       
        if self.config.use_cuda:  
            self.model_encoder = torch.nn.DataParallel(HeteroEncoder(config.input_size).to(self.config.device))
            self.model_cell = torch.nn.DataParallel(GRNModule(config.type).to(self.config.device))
        else:
            self.model_encoder = HeteroEncoder(config.input_size).to(self.config.device)
            self.model_cell = GRNModule(config.type).to(self.config.device)
                
        # initialize criterion (loss)
        self.criterion_cell = PrioritizedLoss()
        self.criterion_encoding = HVGEncodLoss(dim=64, p=config.elpha, use_gpu = self.config.use_cuda)
        self.criterion_center = AdvLoss(self.config.type, use_gpu = self.config.use_cuda)
        self.l1_regular = regu()
        
        # initialize optimizer (sgd/momemtum/weight decay)
        self.optimizer_encoder = optim.SGD(self.model_encoder.parameters(), lr=self.config.LRadv, momentum=self.config.momentum,
                                           weight_decay=0)
        self.optimizer_cell = optim.SGD(self.model_cell.parameters(), lr=self.config.LRadv, momentum=self.config.momentum,
                                        weight_decay=0)


    def adjust_learning_rate(self, optimizer, epoch):
        lr = self.config.LRadv * (0.1 ** ((epoch - 0) // self.config.Prior_ep))

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


    def load_checkpoint(self, args):
        if self.config.checkpoint is not None:
            if os.path.isfile(self.config.checkpoint):
                print("=> loading checkpoint '{}'".format(self.config.checkpoint))
                checkpoint = torch.load(self.config.checkpoint)                
                self.model_encoder.load_state_dict(checkpoint['model_encoding_state_dict'])
                self.model_cell.load_state_dict(checkpoint['model_cell_state_dict'])
            else:
                print("=> no resume checkpoint found at '{}'".format(self.config.checkpoint))


    def train(self, epoch):
        self.model_encoder.train()
        self.model_cell.train()
        total_encoding_loss, total_cell_loss, total_sample_loss, total_kl_loss, total_center_loss = 0., 0., 0., 0., 0.
        self.adjust_learning_rate(self.optimizer_encoder, epoch)
        self.adjust_learning_rate(self.optimizer_cell, epoch)

        # initialize iterator
        iter_rna_loaders = []
        iter_atac_loaders = []
        for rna_loader in self.train_rna_loaders:
            iter_rna_loaders.append(round(rna_loader))
        for atac_loader in self.train_atac_loaders:
            iter_atac_loaders.append(round(atac_loader))
                
        for batch_idx in range(self.training_iters):
            # rna forward
            rna_embeddings = []
            rna_cell_predictions = []
            rna_labels = []
            for iter_rna_loader in iter_rna_loaders:
                rna_data, rna_label = next(iter_rna_loader)    
                # prepare data
                rna_data, rna_label = inp([rna_data, rna_label], self.config)
                # model forward
                rna_embedding = self.model_encoder(rna_data)
                rna_cell_prediction = self.model_cell(rna_embedding)
                rna_embeddings.append(rna_embedding)
                rna_cell_predictions.append(rna_cell_prediction)
                rna_labels.append(rna_label)
                
            # atac forward
            atac_embeddings = []
            atac_cell_predictions = []
            atac_labels = []
            for iter_atac_loader in iter_atac_loaders:
                atac_data, atac_label = next(iter_atac_loader)    
                # prepare data
                atac_data, atac_label = inp([atac_data, atac_label], self.config)
                # model forward
                atac_embedding = self.model_encoder(atac_data)
                atac_cell_prediction = self.model_cell(atac_embedding)

                atac_embeddings.append(atac_embedding)
                atac_cell_predictions.append(atac_cell_prediction)
                atac_labels.append(atac_label)
            
            # caculate loss  
            if self.config.use_cr == True:
                cell_loss = self.criterion_cell(rna_cell_predictions[0], rna_labels[0])
                for i in range(1, len(rna_cell_predictions)):
                    cell_loss += self.criterion_cell(rna_cell_predictions[i], rna_labels[i])

                cell_loss = cell_loss/len(rna_cell_predictions)    

                atac_cell_loss = self.criterion_cell(atac_cell_predictions[0], atac_labels[0])
                for i in range(1, len(atac_cell_predictions)):
                    atac_cell_loss += self.criterion_cell(atac_cell_predictions[i], atac_labels[i])
                
                cell_loss += atac_cell_loss/len(atac_cell_predictions)
            else: 
                cell_loss = 0
                
            
            encoding_loss = self.criterion_encoding(atac_embeddings, rna_embeddings)
            center_loss = self.config.Prior_edge*(self.criterion_center(atac_embeddings, atac_labels) + self.criterion_center(rna_embeddings, rna_labels))
            regularization_loss_encoder = self.l1_regular(self.model_encoder)            
            
            # update encoding weights
            self.optimizer_encoder.zero_grad()  
            regularization_loss_encoder.backward(retain_graph=True)         
            #cell_loss.backward(retain_graph=True)
            encoding_loss.backward(retain_graph=True)    
            center_loss.backward(retain_graph=True)    
            #self.optimizer_encoder.step()
              
            
            regularization_loss_cell = self.l1_regular(self.model_cell)
            # update cell weights
            self.optimizer_cell.zero_grad() 
            if self.config.use_cr == True:
                cell_loss.backward(retain_graph=True)            
            regularization_loss_cell.backward(retain_graph=True)    
            self.optimizer_cell.step()            
            self.optimizer_encoder.step()

            # print log
            total_encoding_loss += encoding_loss.data.item()
            if self.config.use_cr == True:
                total_cell_loss += cell_loss.data.item()
            else: 
                total_cell_loss += 0
            total_center_loss += center_loss.data.item()

                             
                             
        
        
    
    def get_embeddings(self):
        self.model_encoder.eval()
        self.model_cell.eval()
        
        # Get RNA embeddings
        rna_embeddings = []
        rna_cell_predictions = []
        for rna_loader in self.test_rna_loaders:
            for batch_idx, (rna_data, rna_label) in enumerate(rna_loader):
                # prepare data
                rna_data, rna_label = inp([rna_data, rna_label], self.config)
                
                # model forward
                rna_embedding = self.model_encoder(rna_data)
                rna_cell_prediction = self.model_cell(rna_embedding)

                rna_embedding = rna_embedding.data.cpu().numpy()
                rna_cell_prediction = rna_cell_prediction.data.cpu().numpy()
                
                # normalization
                rna_embedding = rna_embedding / norm(rna_embedding, axis=1, keepdims=True)
                rna_cell_prediction = softmax(rna_cell_prediction, axis=1)
                rna_cell_predictions.append(rna_cell_prediction)
                rna_embeddings.append(rna_embedding)
                
        rna_embeddings = np.vstack(rna_embeddings)
        rna_cell_predictions = np.vstack(rna_cell_predictions)
        
        
        # Get ATAC embeddings
        atac_embeddings = []
        atac_cell_predictions = []
        for atac_loader in self.test_atac_loaders:
            for batch_idx, (atac_data, atac_label) in enumerate(atac_loader):
                # prepare data
                atac_data, atac_label = inp([atac_data, atac_label], self.config)
                
                # model forward
                atac_embedding = self.model_encoder(atac_data)                
                atac_cell_prediction = self.model_cell(atac_embedding)
                
                atac_embedding = atac_embedding.data.cpu().numpy()
                atac_cell_prediction = atac_cell_prediction.data.cpu().numpy()
                # normalization
                atac_embedding = atac_embedding / norm(atac_embedding, axis=1, keepdims=True)
                atac_cell_prediction = softmax(atac_cell_prediction, axis=1)
                atac_embeddings.append(atac_embedding)
                atac_cell_predictions.append(atac_cell_prediction)
        atac_embeddings = np.vstack(atac_embeddings)
        atac_cell_predictions = np.vstack(atac_cell_predictions)
        
        return rna_embeddings, rna_cell_predictions, atac_embeddings, atac_cell_predictions
    

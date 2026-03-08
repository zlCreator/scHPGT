import torch
import torch.nn as nn


class PriorGuidedTransformer(nn.Module):
    def __init__(self, embed_dim=64, nhead=4, num_layers=2):
        super(PriorGuidedTransformer, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=nhead, 
            dim_feedforward=embed_dim * 4,
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, rna_emb, atac_emb, mask=None):

        combined = torch.stack([rna_emb, atac_emb], dim=1) 
        
        refined = self.transformer(combined, mask=mask)
        
        return refined[:, 0, :], refined[:, 1, :]

class HeteroEncoder(nn.Module):
    def __init__(self, input_size):
        super(HeteroEncoder, self).__init__()
        self.input_size = input_size
        self.k = 64
        self.f = 64

        self.encoder = nn.Sequential(
            nn.Linear(self.input_size, 64)
        )

    def forward(self, data):
        data = data.float().view(-1, self.input_size)
        embedding = self.encoder(data)

        return embedding


class GRNModule(nn.Module):
    def __init__(self, num_of_class):
        super(GRNModule, self).__init__()
        self.cell = nn.Sequential(
            nn.Linear(64, num_of_class)
        )

    def forward(self, embedding):
        cell_prediction = self.cell(embedding)

        return cell_prediction

class Decoder(nn.Module):
    def __init__(self, output_size):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(64, output_size),
            nn.Sigmoid() 
        )

    def forward(self, embedding):
        return self.decoder(embedding)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, embedding):
        return self.model(embedding)


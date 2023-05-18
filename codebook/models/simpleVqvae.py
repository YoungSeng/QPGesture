import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal


class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class Decoder(nn.Module):
    def __init__(self, dim_pre=64, dim_out=45):
        super(Decoder, self).__init__()
        convolutions = []
        for i in range(3):
            conv_layer = nn.Sequential(
                ConvNorm(dim_pre,
                         dim_pre,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(dim_pre))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)
        self.linear_projection = LinearNorm(dim_pre, dim_out)

    def forward(self, z, target=None):
        z = F.interpolate(z.transpose(1, 2), scale_factor=2)
        for conv in self.convolutions:
            z = F.relu(conv(z))
        z = z.transpose(1, 2)       # (b, 240, 64)
        decoder_output = self.linear_projection(z)
        if target is None:
            return decoder_output
        else:
            loss = F.l1_loss(decoder_output, target)
            return loss, decoder_output


class Encoder(nn.Module):
    '''
    reference from: https://github.com/bshall/VectorQuantizedCPC/blob/master/model.py
    '''

    def __init__(self, in_channels, channels, n_embeddings, z_dim, c_dim):
        super(Encoder, self).__init__()
        self.conv = nn.Conv1d(in_channels, channels, 4, 2, 1, bias=False)       # T // 2
        # self.conv = nn.Conv1d(in_channels, channels, 3, 1, 1, bias=False)     # T
        self.encoder = nn.Sequential(
            nn.LayerNorm(channels),
            nn.ReLU(True),
            nn.Linear(channels, channels, bias=False),
            nn.LayerNorm(channels),
            nn.ReLU(True),
            nn.Linear(channels, channels, bias=False),
            nn.LayerNorm(channels),
            nn.ReLU(True),
            nn.Linear(channels, channels, bias=False),
            nn.LayerNorm(channels),
            nn.ReLU(True),
            nn.Linear(channels, channels, bias=False),
            nn.LayerNorm(channels),
            nn.ReLU(True),
            nn.Linear(channels, z_dim),
        )
        self.codebook = VQEmbeddingEMA(n_embeddings, z_dim)
        self.rnn = nn.LSTM(z_dim, c_dim, batch_first=True)

    def encode(self, mel):
        z = self.conv(mel)
        z_beforeVQ = self.encoder(z.transpose(1, 2))
        z, r, indices = self.codebook.encode(z_beforeVQ)
        c, _ = self.rnn(z)
        return z, c, z_beforeVQ, indices

    def forward(self, mels):
        z = self.conv(mels.float())  # (bz, 80, 128) -> (bz, 512, 128/2)
        z_beforeVQ = self.encoder(z.transpose(1, 2))  # (bz, 512, 128/2) -> (bz, 128/2, 512) -> (bz, 128/2, 64)
        z, r, loss, perplexity = self.codebook(z_beforeVQ)  # z: (bz, 128/2, 64)
        z, r, indices = self.codebook.encode(z_beforeVQ)
        c, _ = self.rnn(z)  # (64, 128/2, 64) -> (64, 128/2, 256)
        return z, c, z_beforeVQ, loss, perplexity


class VQEmbeddingEMA(nn.Module):
    '''
    reference from: https://github.com/bshall/VectorQuantizedCPC/blob/master/model.py
    '''

    def __init__(self, n_embeddings, embedding_dim, commitment_cost=2, decay=0.9999, epsilon=1e-7):
        super(VQEmbeddingEMA, self).__init__()
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        init_bound = 1 / 512
        embedding = torch.Tensor(n_embeddings, embedding_dim)
        embedding.uniform_(-init_bound, init_bound)
        self.register_buffer("embedding", embedding)  # only change during forward
        self.register_buffer("ema_count", torch.zeros(n_embeddings))
        self.register_buffer("ema_weight", self.embedding.clone())

    def encode(self, x):
        M, D = self.embedding.size()
        x_flat = x.detach().reshape(-1, D)

        distances = torch.addmm(torch.sum(self.embedding ** 2, dim=1) +
                                torch.sum(x_flat ** 2, dim=1, keepdim=True),
                                x_flat, self.embedding.t(),
                                alpha=-2.0, beta=1.0)

        indices = torch.argmin(distances.float(), dim=-1)
        quantized = F.embedding(indices, self.embedding)
        quantized = quantized.view_as(x)
        residual = x - quantized
        return quantized, residual, indices.view(x.size(0), x.size(1))

    def forward(self, x):
        M, D = self.embedding.size()
        x_flat = x.detach().reshape(-1, D)

        distances = torch.addmm(torch.sum(self.embedding ** 2, dim=1) +
                                torch.sum(x_flat ** 2, dim=1, keepdim=True),
                                x_flat, self.embedding.t(),
                                alpha=-2.0, beta=1.0)  # calculate the distance between each ele in embedding and x

        indices = torch.argmin(distances.float(), dim=-1)
        encodings = F.one_hot(indices, M).float()
        quantized = F.embedding(indices, self.embedding)
        quantized = quantized.view_as(x)

        if self.training:  # EMA based codebook learning
            self.ema_count = self.decay * self.ema_count + (1 - self.decay) * torch.sum(encodings, dim=0)

            n = torch.sum(self.ema_count)
            self.ema_count = (self.ema_count + self.epsilon) / (n + M * self.epsilon) * n

            dw = torch.matmul(encodings.t(), x_flat)
            self.ema_weight = self.decay * self.ema_weight + (1 - self.decay) * dw

            self.embedding = self.ema_weight / self.ema_count.unsqueeze(-1)

        e_latent_loss = F.mse_loss(x, quantized.detach())
        loss = self.commitment_cost * e_latent_loss

        residual = x - quantized

        quantized = x + (quantized - x).detach()

        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized, residual, loss, perplexity


class simpleVQVAE(nn.Module):
    def __init__(self):
        super(simpleVQVAE, self).__init__()
        self.encoder = Encoder(in_channels=15 * 3, channels=512, n_embeddings=512, z_dim=64, c_dim=256)
        self.decoder = Decoder(dim_pre=64, dim_out=45)

    def encode(self,x):
        z, _, _, indices = self.encoder.encode(x.transpose(1, 2))
        return [indices]

    def forward(self, x):
        z, c, z_beforeVQ, loss_vq, perplexity = self.encoder(x.transpose(1, 2))

        loss_recon, output = self.decoder(z, x)
        return output, loss_vq + loss_recon, perplexity


if __name__ == '__main__':
    '''
    cd codebook/
    python -m models.simpleVqvae
    '''
    # model = Encoder(in_channels=80, channels=512, n_embeddings=512, z_dim=64, c_dim=256)
    # x = torch.rand(2, 80, 128)
    # z, c, z_beforeVQ, loss, perplexity = model(x)
    '''
    z: (2, 64, 64)
    c: (2, 64, 256)
    z_beforeVQ: (2, 64, 64)
    loss
    perplexity
    '''

    model = Encoder(in_channels=15 * 3, channels=512, n_embeddings=512, z_dim=64, c_dim=256)
    model2 = Decoder(dim_pre=64, dim_out=45)
    x = torch.rand(2, 240, 15 * 3)
    z, c, z_beforeVQ, loss, perplexity = model(x.transpose(1, 2))
    pdb.set_trace()
    loss, output = model2(z, x)


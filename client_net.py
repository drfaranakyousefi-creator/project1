import torch
import torch.nn as nn
import torch.optim as optim

# Encoder network
class encoder(nn.Module): 
    def __init__(self, w): 
        super().__init__()
        # Simple feedforward layers to compress input
        self.dense_encoder = nn.Sequential(
            nn.Linear(w, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 6),
            nn.LeakyReLU(),
            nn.Linear(6, 1)
        )

    def forward(self, x):  # x: (batch, w)
        return self.dense_encoder(x)  # output: (batch, 1)


# Decoder network
class decoder(nn.Module): 
    def __init__(self, w): 
        super().__init__()
        # Simple feedforward layers to reconstruct the input
        self.dense_decoder = nn.Sequential(
            nn.Linear(1, 6),
            nn.LeakyReLU(),
            nn.Linear(6, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 32),
            nn.LeakyReLU(),
            nn.Linear(32, w)
        )

    def forward(self, x):  # x: (batch, 1)
        return self.dense_decoder(x)  # output: (batch, w)


# Autoencoder = Encoder + Decoder
class auto_encoder(nn.Module): 
    def __init__(self, w): 
        super().__init__()
        self.encoder = encoder(w)
        self.decoder = decoder(w)

    def forward(self, x):  # x: (batch, w)
        x = self.encoder(x)  # encode to (batch, 1)
        y = self.decoder(x)  # decode back to (batch, w)
        return x, y


# Multi AutoEncoder to handle multiple features
class Multi_autoEncoder(nn.Module): 
    def __init__(self, w, N) -> None:
        super().__init__()
        self.N = N
        self.w = w
        # Create N autoencoders
        self.autoEncoders = nn.ModuleList([auto_encoder(w) for _ in range(N)])
        self.loss_fn = nn.L1Loss()

    def forward(self, x):  # x: (batch, w, N)
        encoder_output = []
        decoder_output = []
        # Pass each feature through its own autoencoder
        for i in range(self.N):
            enc_out, dec_out = self.autoEncoders[i](x[:, :, i])
            encoder_output.append(enc_out)
            decoder_output.append(dec_out)

        # Combine all encoder and decoder outputs
        encoder_output = torch.concat(encoder_output, dim=1)  # (batch, N)
        decoder_output = torch.stack(decoder_output, dim=2)   # (batch, w, N)
        return encoder_output, decoder_output


# Sparse network to compress information
class sparse(nn.Module): 
    def __init__(self, N):
        super().__init__()
        self.N = N
        self.sparse_net = nn.Sequential(
            nn.Linear(N, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 6),
            nn.LeakyReLU(),
            nn.Linear(6, 1)
        )

    def forward(self, x):  # x: (batch, w, N)
        B, w, N = x.shape
        # Reshape for linear layers
        x = x.reshape(-1, N)
        x = self.sparse_net(x)
        # Reshape back to (batch, w)
        return x.reshape(B, w)


# Client network that combines dense and sparse models
class client_network(nn.Module):
    def __init__(self, w, n_features_input, lr):
        super(client_network, self).__init__()
        self.MultiAutoEncoder = Multi_autoEncoder(w, n_features_input)
        self.sparse_net = sparse(n_features_input)
        self.loss_fn = nn.L1Loss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        # x: (batch, 2, w, N)
        sparse_inp = x[:, 0, :, :].clone()  # first channel for sparse input
        dense_inp = x[:, 1, :, :].clone()   # second channel for dense input

        sparse_out = self.sparse_net(sparse_inp)  # (B, w)
        dense_encoder_out, dense_decoder_out = self.MultiAutoEncoder(dense_inp)

        # Concatenate outputs from sparse and dense parts
        prediction_inp = torch.concat([sparse_out, dense_encoder_out], dim=1)  # (B, w + N)
        return prediction_inp, dense_decoder_out, dense_inp

    def train_one_batch(self, prediction_inp, dense_decoder_out, dense_inp, grad):
        # Compute L1 loss between reconstruction and real input
        loss = self.loss_fn(dense_decoder_out, dense_inp)
        loss.backward(retain_graph=True)
        prediction_inp.backward(grad)  # backward gradient
        self.optimizer.step()
        self.optimizer.zero_grad()

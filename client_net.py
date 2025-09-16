import torch
import torch.nn as nn
import torch.optim as optim
class encoder(nn.Module) : 
    def __init__(self , w ) : 
        super().__init__()
        self.dense_encoder = nn.Sequential(
            nn.Linear(w, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 6),
            nn.LeakyReLU(),
            nn.Linear(6, 1)
        )
    def forward(self , x ) : # x : (batch , w)
        return self.dense_encoder(x)

class decoder(nn.Module): 
    def __init__(self , w ): 
        super().__init__()
        self.dense_decoder = nn.Sequential(
            nn.Linear(1, 6),
            nn.LeakyReLU(),
            nn.Linear(6, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 32),
            nn.LeakyReLU(),
            nn.Linear(32, w)
        )
    def forward(self , x) : #x: (batch , 1)
        return self.dense_decoder(x) # out : (Batch , w)

class auto_encoder (nn.Module) : 
    def __init__(self , w ) : 
        super().__init__()
        self.encoder = encoder(w)
        self.decoder = decoder(w)
    def forward(self , x ): #x (batch , w)
        x = self.encoder(x) #x (batch , 1)
        y = self.decoder(x)
        return x , y 

class Multi_autoEncoder(nn.Module): 
    def __init__(self, w , N ) -> None:
        super().__init__()
        self.N = N
        self.w = w 
        self.autoEncoders = nn.ModuleList([auto_encoder(w) for _ in range(N)])
        self.loss_fn = nn.L1Loss()
    def forward(self , x ) : #x (batch , w , N )
        encoder_output = []
        decoder_output = []
        for i in range(self.N) :
            enc_out ,dec_out = self.autoEncoders[i](x[: , : , i ])
            encoder_output.append(enc_out)
            decoder_output.append(dec_out)
        encoder_output = torch.concat(encoder_output , dim=1) #(batch , N )
        decoder_output = torch.stack(decoder_output , dim=2) #(batch , w , N )
        return encoder_output  , decoder_output 



# ____sparse NN_____ 
class sparse(nn.Module) : 
    def __init__(self , N):
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
    def forward(self , x ):  #x : (batch , w  , N  )
        B , w , N = x.shape
        x = x.reshape(-1 , N )
        x = self.sparse_net(x)
        return x.reshape(B , w ) 
         




class client_network(nn.Module):
    def __init__(self, w, n_features_input , lr):
        super(client_network, self).__init__()
        self.MultiAutoEncoder = Multi_autoEncoder(w , n_features_input)
        self.sparse_net = sparse(n_features_input)
        self.loss_fn = nn.L1Loss()
        self.optimizer = optim.Adam(self.parameters() , lr=lr)
    def forward(self, x):
        # x  : (batch , 2 , w , N)
        sparse_inp = x[:, 0, :, :].clone()
        dense_inp  = x[:, 1, :, :].clone()
        sparse_out = self.sparse_net(sparse_inp) #(B , w)
        dense_encoder_out , dense_decoder_out = self.MultiAutoEncoder(dense_inp)
        prediction_inp = torch.concat([sparse_out,dense_encoder_out ] , dim=1) # (B , W + N)
        return prediction_inp , dense_decoder_out  , dense_inp

    def train_one_batch(self, prediction_inp,dense_decoder_out  , dense_inp , grad ) :
        loss = self.loss_fn(dense_decoder_out,dense_inp ) 
        loss.backward(retain_graph= True)
        self.optimizer.step()
        self.optimizer.zero_grad()


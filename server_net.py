import nest_asyncio
import torch
import torch.nn as nn
import torch.optim as optim

# Prediction network model
class prediction_net(nn.Module):
    def __init__(self, w, n_features_input, lr=0.01):
        super().__init__()
        # Use GPU if available, otherwise CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Simple fully connected network
        self.prediction_net = nn.Sequential(
            nn.Linear(w + n_features_input, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 6),
            nn.LeakyReLU(),
            nn.Linear(6, 1)
        ).to(self.device)

        # Mean squared error loss and Adam optimizer
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, combined_embedded, label=None, status='test'):
        # Move input tensor to correct device and enable gradients
        combined_embedded = torch.tensor(combined_embedded, dtype=torch.float, device=self.device)
        combined_embedded.requires_grad_(True)

        if status == 'train':
            # Training mode: compute loss and backpropagate
            label = torch.tensor(label, dtype=torch.float, device=self.device)
            self.optimizer.zero_grad()
            output = self.prediction_net(combined_embedded)
            loss = self.loss_fn(output, label)
            loss.backward()
            input_grad = combined_embedded.grad.detach().cpu().tolist()
            self.optimizer.step()
            # Return gradient for use in another network
            result = {'grad': input_grad}
            return result

        else:  # Testing mode: just return prediction
            output = self.prediction_net(combined_embedded)
            result = output.detach().cpu().tolist()
            result = {'prediction': result}
            return result

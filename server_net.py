from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import nest_asyncio
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.optim as optim

class prediction_net(nn.Module):
    def __init__(self, w, n_features_input, lr=0.01):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.prediction_net = nn.Sequential(
            nn.Linear(w + n_features_input, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 6),
            nn.LeakyReLU(),
            nn.Linear(6, 1)
        ).to(self.device)

        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, combined_embedded, label=None, status='test'):
        combined_embedded = combined_embedded.to(self.device).float()
        combined_embedded.requires_grad_(True)

        if status == 'train':
            if label is None:
                raise ValueError("label must be provided in train mode")
            label = label.to(self.device).float()

            self.optimizer.zero_grad()
            output = self.prediction_net(combined_embedded)
            loss = self.loss_fn(output, label)
            loss.backward()
            input_grad = combined_embedded.grad.detach().cpu() 

            self.optimizer.step()
            return input_grad

        elif status == 'test':
            output = self.prediction_net(combined_embedded)
            return output.detach().cpu()



class Data(BaseModel):

    prediction_iput:list
    label: list
    status : list
            


app = FastAPI()

@app.post("/prediction")

def prediction(data: Data):
    client_message =  data.dict()
    model = prediction_net(3  , 4 , lr=0.01)
    combined_embedded = client_message['prediction_iput']
    label = client_message['label']
    status =client_message['status']
    data = model( combined_embedded, label, status)

    return data.dict()


import nest_asyncio
from pyngrok import ngrok

ngrok_tunnel = ngrok.connect(8000)
print(f"Public URL: {ngrok_tunnel.public_url}")

nest_asyncio.apply()
uvicorn.run(app, port=8000)





















نسخه قدیمی که کار میکرد 



from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import nest_asyncio
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.optim as optim

class prediction_net(nn.Module):
    def __init__(self, w, n_features_input, lr=0.01):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.prediction_net = nn.Sequential(
            nn.Linear(w + n_features_input, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 6),
            nn.LeakyReLU(),
            nn.Linear(6, 1)
        ).to(self.device)

        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, combined_embedded, label=None, status='test'):
        combined_embedded = combined_embedded.to(self.device).float()
        combined_embedded.requires_grad_(True)

        if status == 'train':
            if label is None:
                raise ValueError("label must be provided in train mode")
            label = label.to(self.device).float()

            self.optimizer.zero_grad()
            output = self.prediction_net(combined_embedded)
            loss = self.loss_fn(output, label)
            loss.backward()
            input_grad = combined_embedded.grad.detach().cpu() 

            self.optimizer.step()
            return input_grad

        elif status == 'test':
            output = self.prediction_net(combined_embedded)
            return output.detach().cpu()


# مدل داده‌ها: دیکشنری با دو لیست
class Data(BaseModel):
    list1: list
    list2: list

app = FastAPI()

@app.post("/echo_lists")
def echo_lists(data: Data):
    # فقط چاپ می‌کنیم
    print(f"Server received: {data.dict()}")
    # برگردوندن همون دیکشنری
    return data.dict()

# راه‌اندازی ngrok و سرور
import nest_asyncio
from pyngrok import ngrok

ngrok_tunnel = ngrok.connect(8000)
print(f"Public URL: {ngrok_tunnel.public_url}")

nest_asyncio.apply()
uvicorn.run(app, port=8000)

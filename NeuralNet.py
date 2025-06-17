import torch
from torch.autograd import grad

class Hnet(torch.nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, layers=1, output_dim=1):
        super(Hnet, self).__init__()
        
        # Define the neural network layers
        self.layers_list = []
        for i in range(layers):
            if i == 0:
                self.layers_list.append(torch.nn.Linear(input_dim, hidden_dim))
            else:
                self.layers_list.append(torch.nn.Linear(hidden_dim, hidden_dim))
            self.layers_list.append(torch.nn.Tanh())
        
        # Output layer
        self.layers_list.append(torch.nn.Linear(hidden_dim, output_dim))
        
        # Combine all layers into a sequential model
        self.model = torch.nn.Sequential(*self.layers_list)

    def forward(self, x):
        x = x.requires_grad_(True)  # Ensure x is a new tensor with requires_grad=True
        # Forward pass through the neural network
        H = self.model(x)  # Ensure output is a scalar
        return H
    

class Jmat(torch.nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, layers=1, output_dim=4):
        super(Jmat, self).__init__()

        self.layers_list = []
        for i in range(layers):
            if i == 0:
                self.layers_list.append(torch.nn.Linear(input_dim, hidden_dim))
            else:
                self.layers_list.append(torch.nn.Linear(hidden_dim, hidden_dim))
            self.layers_list.append(torch.nn.Tanh())
        
        # Output layer
        self.layers_list.append(torch.nn.Linear(hidden_dim, output_dim))
        
        # Combine all layers into a sequential model
        self.model = torch.nn.Sequential(*self.layers_list)
        

    def forward(self, x):
        # Transform the output to a antisymmetric 2 x 2 matrix
        mat = self.model(x)
        J = torch.reshape(mat, (-1, 2, 2))      # this -1 is a placeholder for the batch size
        J = (J - J.transpose(-1, -2)) / 2
        return J


class HamODE(torch.nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, layers=1):
        super(HamODE, self).__init__()
        self.Hnet = Hnet(input_dim, hidden_dim, layers)
        self.Jmat = Jmat(input_dim, hidden_dim, layers)
        
    def forward(self, t, x):
        with torch.enable_grad():
            x = x.requires_grad_(True).squeeze()
            H = self.Hnet(x).squeeze()  # Ensure H is a scalar
            # Compute the gradient of H with respect to x
            dH_dx = grad(H.sum(), x, create_graph=True)[0]
            J = self.Jmat(x).squeeze()  # Ensure J is a 2x2 matrix
            return torch.matmul(J, dH_dx.unsqueeze(-1)).squeeze(-1)
    
# in NeuralNet.py

# class HamODE(torch.nn.Module):
#     def __init__(self, input_dim=2, hidden_dim=64, layers=1):
#         super(HamODE, self).__init__()
#         self.Hnet = Hnet(input_dim, hidden_dim, layers)
#         self.Jmat = Jmat(input_dim, hidden_dim, layers)

#     def forward(self, t, x):
#         with torch.enable_grad():
#             x = x.requires_grad_(True)
#             print("x: ", x.shape)
#             H = self.Hnet(x)

#             dH_dx = torch.autograd.grad(
#                 H,                
#                 x,
#                 create_graph=True,
#                 allow_unused=True
#             )[0]

#             print("dH_dx: ", dH_dx.shape)

#             J  = self.Jmat(x)
#             print("J: ", J.shape)

#             out = (J @ dH_dx.unsqueeze(-1)).squeeze(-1)
#         return out
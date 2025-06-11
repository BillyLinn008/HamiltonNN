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
            self.layers_list.append(torch.nn.ReLU())
        
        # Output layer
        self.layers_list.append(torch.nn.Linear(hidden_dim, output_dim))
        
        # Combine all layers into a sequential model
        self.model = torch.nn.Sequential(*self.layers_list)

    def forward(self, x):
        x = x.detach().requires_grad_(True)  # Enable gradient computation for input x
        # Forward pass through the neural network
        H = self.model(x).sum()  # Ensure output is a scalar
        dH_dx = grad(H, x, create_graph=True)[0] # Compute the gradient of H with respect to x
        return H, dH_dx
    

class Jmat(torch.nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, layers=1, output_dim=4):
        super(Jmat, self).__init__()

        self.layers_list = []
        for i in range(layers):
            if i == 0:
                self.layers_list.append(torch.nn.Linear(input_dim, hidden_dim))
            else:
                self.layers_list.append(torch.nn.Linear(hidden_dim, hidden_dim))
            self.layers_list.append(torch.nn.ReLU())
        
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
        H, dH_dx = self.Hnet(x)
        J = self.Jmat(x)
        return torch.matmul(J, dH_dx.unsqueeze(-1)).squeeze(-1)
    
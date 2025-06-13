import numpy
import matplotlib.pyplot as plt
from torchdiffeq import odeint
import torch

class Data():
    def __init__(self, N_data=100, t_tot=10, ratio=0.75, r_initial=torch.tensor([1.0, 1.0]), 
                 noise_level=0.1, seed=42):
        self.N_data = N_data
        self.t_tot = t_tot
        self.ratio = ratio
        self.seed = seed
        self.noise_level = noise_level
        self.split = int(N_data * ratio)
        self.r_initial = r_initial

        self.t = torch.linspace(0, t_tot, N_data)
        self.r = self.solver()
        
    # Define the dynamics of the system 
    def dynamics(self, t, r):
        x, y = r[0], r[1]
        dxdt = y
        dydt = -x
        return torch.stack([dxdt, dydt])
    
    # Use the ODE solver to compute the trajectory
    def solver(self):
        return odeint(self.dynamics, self.r_initial, self.t, method='dopri5')

    # Generate the data in the transformed phase space
    # where a = q + p and b = q - p
    def training_data(self):
        # generate noise
        torch.manual_seed(self.seed)
        noise = self.noise_level * torch.normal(mean=0, std=0.1, size=self.r.shape)
        real_data = self.r + noise
        a, b = real_data[:self.split,0] + real_data[:self.split, 1], real_data[:self.split, 0] - real_data[:self.split, 1]
        x = torch.stack((a, b), axis=1)
        return self.t[:self.split], x

    def test_data(self):
        # generate noise
        torch.manual_seed(self.seed)
        noise = self.noise_level * torch.normal(mean=0, std=0.1, size=self.r.shape)
        real_data = self.r + noise
        a, b = real_data[self.split:,0] + real_data[self.split:, 1], real_data[self.split:, 0] - real_data[self.split:, 1]
        x = torch.stack((a, b), axis=1)
        return self.t[self.split:], x

    def plot_data(self):
        t, x = self.training_data()
        t, x = t.numpy(), x.numpy()
        q, p = self.r[:, 0], self.r[:, 1]
        q, p = q.numpy(), p.numpy()

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.scatter(x[:, 0], x[:, 1], s=1)
        plt.xlabel('a = q + p')
        plt.ylabel('b = q - p')
        plt.title('Transformed Phase Space Plot')

        plt.subplot(1, 2, 2)
        plt.scatter(q, p, s=1)
        plt.xlabel('q')
        plt.ylabel('p')
        plt.title('Original Phase Space Plot')
        plt.tight_layout()
        plt.show()
    
if __name__ == "__main__":
    data = Data()
    data.plot_data()

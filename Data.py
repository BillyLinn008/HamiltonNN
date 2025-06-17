import numpy
import matplotlib.pyplot as plt
from torchdiffeq import odeint
import torch

class Data():
    def __init__(self, N_data=100, t_steps=50, t_tot=10, ratio=0.75, r_initial=torch.tensor([1.0, 0.0]), 
                 noise_level=0.1, seed=42):
        self.N_data = N_data
        self.seed = seed
        self.noise_level = noise_level
        self.split = int(N_data * ratio)
        self.r_initial = r_initial

        self.t = torch.linspace(0, t_tot, t_steps)

        # self.r records the trajectories of the system: r is of shape (N_data, t_steps, 2)
        # where the last dimension corresponds to (q, p), and the first dimension corresponds to the different trajectories
        self.r = self.solver()


    # Define the dynamics of the system 
    def dynamics(self, t, r):
        q, p = r[0], r[1]
        dqdt = p
        dpdt = -q
        return torch.stack([dqdt, dpdt])

    # Use the ODE solver to compute the trajectory
    def solver(self):
        r_initial_expanded = self.r_initial.unsqueeze(0).expand(self.N_data, 2)
        initial_data = torch.normal(mean=r_initial_expanded, std=0.1)
        input = (initial_data[:, 0], initial_data[:, 1])
        q, p = odeint(self.dynamics, input, self.t, method='dopri5')
        q, p = q.T, p.T
        r = torch.stack((q, p), axis=-1)
        return r

    # Generate the data in the transformed phase space
    # where a = q + p and b = q - p
    def training_data(self):
        # generate noise
        torch.manual_seed(self.seed)
        # noise = self.noise_level * torch.normal(mean=0, std=0.1, size=self.r.shape)
        real_data = self.r # + noise
        a, b = real_data[:self.split, :, 0] + real_data[:self.split, :, 1], real_data[:self.split, :, 0] - real_data[:self.split, :, 1]
        x = torch.stack((a, b), axis=-1)
        return self.t, x

    def test_data(self):
        # generate noise
        torch.manual_seed(self.seed)
        # noise = self.noise_level * torch.normal(mean=0, std=0.1, size=self.r.shape)
        real_data = self.r # + noise (Let's Ignore Noise For Now)
        a, b = real_data[self.split:, :, 0] + real_data[self.split:, :, 1], real_data[self.split:, :, 0] - real_data[self.split:, :, 1]
        x = torch.stack((a, b), axis=-1)
        return self.t, x

    def plot_data(self):
        # pull out your data
        t, x = self.training_data()
        r = self.r
        x = x.numpy()
        r = r.numpy()

        # create one Figure with two subplots (Axes)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # loop over each trajectory
        for i in range(self.split):
            a = x[i, :, 0]    # transformed q+p
            b = x[i, :, 1]    # transformed q−p
            q = r[i, :, 0]    # original q
            p = r[i, :, 1]    # original p

            ax1.scatter(a, b, s=1, label=f'Traj {i+1}')
            ax2.scatter(q, p, s=1, label=f'Traj {i+1}')

        # label & title each subplot
        ax1.set_xlabel('a = q + p')
        ax1.set_ylabel('b = q - p')
        ax1.set_title('Transformed Phase Space')
        # ax1.legend(loc='best', markerscale=5, fontsize='small')

        ax2.set_xlabel('q')
        ax2.set_ylabel('p')
        ax2.set_title('Original Phase Space')
        # ax2.legend(loc='best', markerscale=5, fontsize='small')

        fig.tight_layout()
        plt.show()

    
if __name__ == "__main__":
    data = Data()
    data.plot_data()

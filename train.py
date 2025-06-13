import torch
from torchdiffeq import odeint_adjoint
# from torchdiffeq import odeint
from NeuralNet import HamODE
from Data import Data
device='cpu'

def data_prepare():
    data = Data()
    t_train, x_train = data.training_data()
    t_test, x_test = data.test_data()

    # prepare training data: input is X_train and ground truth is y_train
    X_train, y_train, t_train = x_train[:-1], x_train[1:], t_train[:-1]

    # prepare test data: input is X_test and ground truth is y_test
    X_test, y_test, t_test = x_test[:-1], x_test[1:], t_test[:-1]
    return (X_train, y_train, t_train), (X_test, y_test, t_test)


# LET'S FIGURE OUT HOW TO USE ODEINT_ADJOINT

def train():
    (X_train, y_train, t_train), (X_test, y_test, t_test) = data_prepare()
    X_train, y_train, t_train = X_train.to(device), y_train.to(device), t_train.to(device)

    # Initialize the neural net:
    hamODE = HamODE().to(device)
    optimizer = torch.optim.Adam(hamODE.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()
    num_epochs = 100

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # Solve the ODE for the whole batch of initial states:
        y_pred = odeint_adjoint(hamODE, X_train[0], t_train, method='dopri5')

        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()


        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


if __name__ == "__main__":
    train()
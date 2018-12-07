import torch
import torch.nn as nn
import torch.optim as optim
import random


class SiameseAutoencoder(nn.Module):

    def __init__(self, inputNeurons=4, hiddenNeurons=3, outputNeurons=4, actionDim=1):
        super(SiameseAutoencoder, self).__init__()

        self.encoder = nn.Linear(inputNeurons, hiddenNeurons)

        # decoderState decodes current state from state latent space
        self.decoderState = nn.Linear(hiddenNeurons, outputNeurons)
        # decoderNextState decodes next state from state latent space + action
        self.decoderNextState = nn.Linear(hiddenNeurons + actionDim, outputNeurons)

        self.activation = torch.sigmoid

    def forward(self, stateinput, action):

        # encode current state and create latent space
        latent_space = self.activation(self.encoder(stateinput))

        # decode current state
        outState = (self.decoderState(latent_space))

        # append action to latent space
        latent_space_action = torch.cat((latent_space, action), 0)

        # decode next state from latent space with action
        outNextState = (self.decoderNextState(latent_space_action))

        return torch.cat((outState, outNextState), 0)

if __name__ == "__main__":

    with open("../../data/cartpole.data") as f:
        data = [list(map(eval, l[:-1].split("\t"))) for l in f.readlines()]
    print("READ FILE")

    # print(data[0])
    net = SiameseAutoencoder(4, 3, 4, 1)
    optimizer = optim.SGD(net.parameters(), lr=0.1)
    criterion = nn.MSELoss()

    for epoch in range(10):
        for i in range(10000):
            rand = random.randint(0, len(data) - 1)
            sample_id = rand
            next_sample_id = rand + 1

            current_state = torch.tensor(data[sample_id][0])
            action = torch.tensor([data[sample_id][1]]).float()

            next_state = torch.tensor(data[next_sample_id][0])

            target = torch.cat((current_state, next_state), 0)
            optimizer.zero_grad()
            out = net(current_state, action)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
        if epoch % 1 == 0:
            print(epoch)

    torch.save(net.state_dict(), "../../models/siamese.model")



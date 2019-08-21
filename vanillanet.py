import torch


class ConvModel(torch.nn.Module):
    def __init__(self, in_channel, mapsize, n_act):
        super().__init__()
        self.convnet = torch.nn.Sequential(
            torch.nn.Conv2d(in_channel, 64, 5, 1, padding=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 32, 5, 1, padding=2),
            torch.nn.ReLU(),
        )

        self.policy = torch.nn.Sequential(
            torch.nn.Linear(mapsize*mapsize*32, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, n_act)
        )
        self.value = torch.nn.Sequential(
            torch.nn.Linear(mapsize*mapsize*32, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1)
        )

        self.mapsize = mapsize
        gain = torch.nn.init.calculate_gain("relu")

        def param_init(module):
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_normal_(module.weight, gain)
                torch.nn.init.zeros_(module.bias)
            if isinstance(module, torch.nn.Conv2d):
                torch.nn.init.dirac_(module.weight)
                torch.nn.init.zeros_(module.bias)
        self.apply(param_init)

    def forward(self, state):
        encode = self.convnet(state)

        value = self.value(
            encode.reshape(-1, self.mapsize*self.mapsize*32))
        logits = self.policy(
            encode.reshape(-1, self.mapsize*self.mapsize*32))

        return logits, value

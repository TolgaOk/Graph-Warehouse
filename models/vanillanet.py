import torch


class ConvModel(torch.nn.Module):
    def __init__(self, in_channel, mapsize, n_act, conv_in_size, conv_out_size, dense_size, **kwargs):
        super().__init__()
        self.convnet = torch.nn.Sequential(
            torch.nn.Conv2d(in_channel + 2, conv_in_size, 5, 1, padding=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(conv_in_size, conv_out_size, 5, 1, padding=2),
            torch.nn.ReLU(),
        )

        self.policy = torch.nn.Sequential(
            torch.nn.Linear(conv_out_size, dense_size),
            torch.nn.ReLU(),
            torch.nn.Linear(dense_size, n_act)
        )
        self.value = torch.nn.Sequential(
            torch.nn.Linear(conv_out_size, dense_size),
            torch.nn.ReLU(),
            torch.nn.Linear(dense_size, 1)
        )
        self.pool = torch.nn.MaxPool2d(mapsize)

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
        bs, channel, height, width = state.shape
        # ----------------- Spatial Concatatination -----------------
        x_cords, y_cords = torch.meshgrid(torch.arange(-1, 1, 2/height),
                                          torch.arange(-1, 1, 2/width))
        x_cords = x_cords.reshape(1, 1, height, width).repeat(bs, 1, 1, 1)
        y_cords = y_cords.reshape(1, 1, height, width).repeat(bs, 1, 1, 1)
        state = torch.cat([state,
                           y_cords.to(state.device).float(),
                           x_cords.to(state.device).float()], dim=1)
        # -----------------------------------------------------------

        encode = self.convnet(state)
        # encode = encode.mean((-1, -2))
        encode = self.pool(encode)

        value = self.value(
            encode.reshape(-1, 32))
        logits = self.policy(
            encode.reshape(-1, 32))

        return logits, value

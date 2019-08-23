import torch
import math


class RelationalModule(torch.nn.Module):
    def __init__(self, in_feature, dim_value, dim_key):
        super().__init__()

        self.in_feature = in_feature
        self.dim_key = dim_key
        self.query_fc = torch.nn.Linear(in_feature + 2, dim_key)
        self.key_fc = torch.nn.Linear(in_feature + 2, dim_key)
        self.value_fc = torch.nn.Linear(in_feature + 2, dim_value)
        self.entitywise_fc = torch.nn.Linear(dim_value, in_feature)
        self.layer_norm = torch.nn.LayerNorm(in_feature)
        self.instance_norm = torch.nn.InstanceNorm2d(in_feature)
        self.apply(self.param_init)

    def forward(self, input, activation=lambda x: x):
        """
            input -> (B ,F ,S ,S)   :4D
            input -> (B ,S ,S ,F)   :4D
            input -> (B*S**2, F+2)  :2D
        """
        input = input.permute(0, 2, 3, 1)
        bs, heigth, width, fs = input.shape
        # ----------- ! Spatial Coordinate Concatenation ----------------
        x_cords, y_cords = torch.meshgrid(torch.arange(-1, 1, 2/heigth),
                                          torch.arange(-1, 1, 2/width))
        x_cords = x_cords.reshape(1, heigth, width, 1).repeat(bs, 1, 1, 1)
        y_cords = y_cords.reshape(1, heigth, width, 1).repeat(bs, 1, 1, 1)
        coord_input = torch.cat([input,
                                 y_cords.to(input.device).float(),
                                 x_cords.to(input.device).float()], dim=-1)
        # ------------ Spatial Coordinate Concatenation ! ---------------

        coord_input = coord_input.reshape(bs*heigth*width, fs+2)
        # (B * S**2, F_k)
        query = self.query_fc(coord_input).reshape(bs, heigth*width, -1)
        query = activation(query)
        # (B, S**2, F_k)
        key = self.key_fc(coord_input).reshape(bs, heigth*width, -1)
        key = activation(key)
        # (B, S**2, F_v)
        value = self.value_fc(coord_input).reshape(bs, heigth*width, -1)
        value = activation(value)

        dim_key_sqrt = math.sqrt(self.dim_key)
        # (B, S**2, S**2)
        weights = torch.matmul(query, key.permute(0, 2, 1))/dim_key_sqrt
        attn_weights = torch.nn.functional.softmax(weights, dim=-1)

        # (B, S**2, F_v)
        attened_values = torch.matmul(attn_weights, value)
        # (B, S**2, F)
        output = self.entitywise_fc(
            attened_values.reshape(bs*heigth*width, -1))
        output = activation(output)

        # feature = torch.mean(output.reshape(bs, heigth, width, -1) + input,
        #                      dim=(1, 2))
        # feature = self.layer_norm(feature)
        # return feature

        feature = output.reshape(bs, heigth, width, -1) + input
        feature = self.instance_norm(feature.permute(0, 3, 1, 2))
        return feature

    def param_init(self, module):
        gain = torch.nn.init.calculate_gain("relu")
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_normal_(module.weight, gain)
            torch.nn.init.zeros_(module.bias)


class RelationalNet(torch.nn.Module):
    def __init__(self, in_channel, mapsize, n_act):
        super().__init__()
        self.in_channel = in_channel
        self.mapsize = mapsize

        # Input
        self.convnet = torch.nn.Sequential(
            torch.nn.Conv2d(in_channel, 64, 5, 1, padding=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 32, 5, 1, padding=2),
            torch.nn.ReLU(),
        )

        # Relational
        self.relational_module = RelationalModule(32, 16, 16)

        # Output
        self.policy = torch.nn.Sequential(
            torch.nn.Linear(32*mapsize*mapsize, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, n_act)
        )
        self.value = torch.nn.Sequential(
            torch.nn.Linear(32*mapsize*mapsize, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1)
        )
        self.apply(self.param_init)

    def forward(self, state):
        bs = state.shape[0]
        state = self.convnet(state)

        feature = self.relational_module(state, torch.nn.functional.relu)
        feature = feature.reshape(bs, -1)

        policy = self.policy(feature)
        value = self.value(feature)
        return policy, value

    def param_init(self, module):
        gain = torch.nn.init.calculate_gain("relu")
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_normal_(module.weight, gain)
            torch.nn.init.zeros_(module.bias)
        if isinstance(module, torch.nn.Conv2d):
            torch.nn.init.dirac_(module.weight)
            torch.nn.init.zeros_(module.bias)


if __name__ == "__main__":
    BS = 32
    S = 10
    F = 5
    x = torch.ones(BS, F, S, S).float()

    model = RelationalNet(F, 10, 4)
    print([i.shape for i in model(x)])

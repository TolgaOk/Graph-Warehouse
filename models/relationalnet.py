import torch
import math


class RelationalModule(torch.nn.Module):
    def __init__(self, in_feature, dim_value, dim_key, n_heads,
                 mapsize, dropout=0.1):
        super().__init__()
        assert n_heads <= dim_key, ("Argment <n_heads> should not be "
                                    "greater than argument <dim_key>")
        assert n_heads <= dim_value, ("Argment <n_heads> should not be "
                                      "greater than argument <dim_value>")
        assert dim_key % n_heads == 0,   ("Argument <dim_key>   must be"
                                          " divisible without reminder")
        assert dim_value % n_heads == 0, ("Argument <dim_value> must be"
                                          " divisible without reminder ")
        self.n_heads = n_heads
        self.in_feature = in_feature
        self.dim_key = dim_key
        self.query_fc = torch.nn.Linear(in_feature + 2, dim_key)
        self.query_layernorm = torch.nn.LayerNorm(
            (mapsize*mapsize, dim_key//n_heads))
        self.key_fc = torch.nn.Linear(in_feature + 2, dim_key)
        self.key_layernorm = torch.nn.LayerNorm(
            (mapsize*mapsize, dim_key//n_heads))
        self.value_fc = torch.nn.Linear(in_feature + 2, dim_value)
        self.value_layernorm = torch.nn.LayerNorm(
            (mapsize*mapsize, dim_value//n_heads))
        self.entitywise_fc = torch.nn.Linear(dim_value, in_feature)
        self.layer_norm = torch.nn.LayerNorm(mapsize, mapsize, in_feature)
        self.instance_norm = torch.nn.InstanceNorm2d(in_feature)
        self.dropout = torch.nn.Dropout(dropout)
        self.apply(self.param_init)

    def forward(self, input, activation=lambda x: x):
        """
            input -> (B ,F ,S ,S)   :4D
            input -> (B ,S ,S ,F)   :4D
            input -> (B*S**2, F+2)  :2D
        """
        input = input.permute(0, 2, 3, 1)
        bs, height, width, fs = input.shape
        # ----------- ! Spatial Coordinate Concatenation ----------------
        x_cords, y_cords = torch.meshgrid(torch.arange(-1, 1, 2/height),
                                          torch.arange(-1, 1, 2/width))
        x_cords = x_cords.reshape(1, height, width, 1).repeat(bs, 1, 1, 1)
        y_cords = y_cords.reshape(1, height, width, 1).repeat(bs, 1, 1, 1)
        coord_input = torch.cat([input,
                                 y_cords.to(input.device).float(),
                                 x_cords.to(input.device).float()], dim=-1)
        # ------------ Spatial Coordinate Concatenation ! ---------------

        coord_input = coord_input.reshape(bs*height*width, fs+2)
        # (B * S**2, F_k)
        query = self.query_fc(coord_input)
        query = query.reshape(bs, height*width, self.n_heads, -1)
        query = query.permute(0, 2, 1, 3)
        query = query.reshape(bs*self.n_heads, height*width, -1)
        query = activation(self.query_layernorm(query))
        # (B, S**2, F_k)
        key = self.key_fc(coord_input)
        key = key.reshape(bs, height*width, self.n_heads, -1)
        key = key.permute(0, 2, 1, 3)
        key = key.reshape(bs*self.n_heads, height*width, -1)
        key = activation(self.key_layernorm(key))
        # (B, S**2, F_v)
        value = self.value_fc(coord_input)
        value = value.reshape(bs, height*width, self.n_heads, -1)
        value = value.permute(0, 2, 1, 3)
        value = value.reshape(bs*self.n_heads, height*width, -1)
        value = activation(self.value_layernorm(value))

        dim_key_sqrt = math.sqrt(self.dim_key)
        # (B*H, S**2, S**2)
        weights = torch.matmul(query, key.permute(0, 2, 1))/dim_key_sqrt
        attn_weights = torch.nn.functional.softmax(weights, dim=-1)

        attn_entropy = -(attn_weights*weights).sum(-1)

        # (B*H, S**2, F_v//H)
        attened_values = torch.matmul(attn_weights, value)
        attened_values = attened_values.reshape(
            bs, self.n_heads, height*width, -1)
        attened_values = attened_values.permute(0, 2, 1, 3)
        attened_values = attened_values.reshape(bs, height*width, -1)
        # (B, S**2, F)
        output = self.entitywise_fc(
            attened_values.reshape(bs*height*width, -1))
        output = activation(output)

        output = self.dropout(output)
        feature = output.reshape(bs, height, width, -1) + input
        feature = feature.permute(0, 3, 1, 2)
        feature = self.layer_norm(feature)

        return feature, attn_weights, attn_entropy

    def param_init(self, module):
        gain = torch.nn.init.calculate_gain("relu")
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_normal_(module.weight, gain)
            torch.nn.init.zeros_(module.bias)


class RelationalNet(torch.nn.Module):
    def __init__(self, in_channel, mapsize, n_act, conv_size, attn_size,
                 qkv_dim, n_heads, dense_size, **kwargs):
        super().__init__()
        self.in_channel = in_channel
        self.mapsize = mapsize

        # Input
        self.convnet = torch.nn.Sequential(
            torch.nn.Conv2d(in_channel, conv_size, 5, 1, padding=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(conv_size, attn_size, 5, 1, padding=2),
            torch.nn.ReLU(),
        )

        # Relational
        self.relational_module = RelationalModule(attn_size, qkv_dim, qkv_dim,
                                                  n_heads=n_heads,
                                                  mapsize=mapsize)
        self.pool = torch.nn.MaxPool2d(mapsize)

        # Output
        self.policy = torch.nn.Sequential(
            torch.nn.Linear(attn_size, dense_size),
            torch.nn.ReLU(),
            torch.nn.Linear(dense_size, n_act)
        )
        self.value = torch.nn.Sequential(
            torch.nn.Linear(attn_size, dense_size),
            torch.nn.ReLU(),
            torch.nn.Linear(dense_size, 1)
        )
        self.apply(self.param_init)
        self.attn_weights = None

    def forward(self, state):
        bs = state.shape[0]
        state = self.convnet(state)

        feature, attn_weights, attn_entropy = self.relational_module(
            state, torch.nn.functional.relu)
        # ---------------------------
        self.attn_weights = attn_weights
        self.attn_entropy = attn_entropy
        # ---------------------------
        feature = self.pool(feature)
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

import torch
import math


class VisualAttn(torch.nn.Module):

    def __init__(self, in_channel, n_entity):
        super().__init__()
        self.pre_conv = torch.nn.Conv2d(
            in_channel, out_channels=in_channel,
            kernel_size=3, padding=1)
        self.visattn_conv = torch.nn.Conv2d(
            in_channel, out_channels=n_entity,
            kernel_size=3, padding=1)
        self.conv3d = torch.nn.Conv3d(
            in_channel, in_channel,
            kernel_size=(1, 1, 1), padding=(0, 1, 1))

    def forward(self, state):
        x = torch.relu(self.pre_conv(state))
        attn = torch.sigmoid(self.visattn_conv(x))

        x = torch.einsum("bfyx, beyx->bfeyx", state, attn)
        x = torch.relu(self.conv3d(x)).permute(0, 2, 1, 3, 4)
        x = x.mean((-1, -2))
        return x, attn


class SelfAttn(torch.nn.Module):
    def __init__(self, in_dim, n_entity, qkv_dim, n_heads):
        super().__init__()
        self.qkv_dim = qkv_dim
        self.n_heads = n_heads

        self.query_fc = torch.nn.Linear(in_dim, qkv_dim)
        self.query_layernorm = torch.nn.LayerNorm(
            (n_entity, qkv_dim//n_heads))

        self.key_fc = torch.nn.Linear(in_dim, qkv_dim)
        self.key_layernorm = torch.nn.LayerNorm(
            (n_entity, qkv_dim//n_heads))

        self.value_fc = torch.nn.Linear(in_dim, qkv_dim)
        self.value_layernorm = torch.nn.LayerNorm(
            (n_entity, qkv_dim//n_heads))

        self.entitywise_fc = torch.nn.Linear(qkv_dim, in_dim)
        self.layer_norm = torch.nn.LayerNorm(n_entity, in_dim)

    def forward(self, features):

        bs, n_entity, n_feature = features.shape

        query = self.query_layernorm(self.query_fc(features).reshape(
            bs, n_entity, self.n_heads, self.qkv_dim//self.n_heads).permute(0, 2, 1, 3))
        query = torch.relu(query)

        key = self.key_layernorm(self.key_fc(features).reshape(
            bs, n_entity, self.n_heads, self.qkv_dim//self.n_heads).permute(0, 2, 1, 3))
        key = torch.relu(key)

        value = self.value_layernorm(self.value_fc(features).reshape(
            bs, n_entity, self.n_heads, self.qkv_dim//self.n_heads).permute(0, 2, 1, 3))
        value = torch.relu(value)

        qkv_dim_sqrt = math.sqrt(self.qkv_dim)

        attn = torch.einsum("bhef, bhxf->bhxe", key, query)
        attn = torch.nn.functional.softmax(attn/qkv_dim_sqrt, dim=-1)
        attned_value = torch.einsum("bhxe, bhef->bxhf", attn, value)
        features = attned_value.reshape(bs, n_entity, self.qkv_dim)

        features = torch.relu(self.entitywise_fc(features))
        return features

    def broadcast(self, features, vis_attns):
        return torch.einsum("bef, beyx->bfyx", features, vis_attns)


class OurAttnModule(torch.nn.Module):

    def __init__(self, in_channel, out_channel, qkv_dim,  n_entity, n_heads):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channel, out_channel,
                                    kernel_size=3, padding=1)
        self.visual_attn = VisualAttn(out_channel, n_entity)
        self.self_attn = SelfAttn(out_channel, n_entity, qkv_dim, n_heads)

        self.instance_norm = torch.nn.InstanceNorm2d(qkv_dim)

    def forward(self, state):
        state = torch.relu(self.conv(state))
        features, attn = self.visual_attn(state)
        features = self.self_attn(features)
        post_state = self.self_attn.broadcast(features, attn)
        return self.instance_norm(post_state + state)


class OurAttnNet(torch.nn.Module):

    def __init__(self, in_channel, mapsize, n_act, n_entity=4, n_heads=4):
        super().__init__()

        self.convnet = torch.nn.Sequential(
            torch.nn.Conv2d(in_channel, 64, 3, 1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 32, 3, 1, padding=1),
            torch.nn.ReLU(),
        )

        self.attn_module = OurAttnModule(32, 32, 16,
                                         n_entity=n_entity, n_heads=n_heads)

        self.policy = torch.nn.Sequential(
            torch.nn.Linear(32, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, n_act)
        )
        self.value = torch.nn.Sequential(
            torch.nn.Linear(32, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1)
        )
        self.apply(self.param_init)

    def forward(self, state):
        height, width = state.shape[-2:]
        state = self.convnet(state)
        state = self.attn_module(state)
        feature = torch.nn.functional.max_pool2d(
            state, (height, width)).squeeze()
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
    in_channel = 5
    out_channel = 16
    qkv_dim = 16
    n_entity = 5
    n_heads = 4
    height = 10
    width = 10
    bs = 32
    n_act = 4

    state = torch.ones(bs, in_channel, height, width)

    net = OurAttnNet(in_channel, None, n_act, n_entity, n_heads)
    # layer = OurAttnModule(in_channel, out_channel, qkv_dim, n_entity, n_heads)

    print(net(state)[0].shape)

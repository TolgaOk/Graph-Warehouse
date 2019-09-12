import torch
import math
from itertools import product
from numpy import pi


class VisualAttn(torch.nn.Module):

    def __init__(self, in_channel, n_entity, mapsize):
        super().__init__()
        self.fourier_bases = FourierBases([1, 2, 3, 4],
                                          [1, 2, 3, 4],
                                          mapsize,
                                          mapsize)
        self.pre_conv = torch.nn.Conv2d(
            in_channel+self.fourier_bases.channel_size,
            out_channels=in_channel,
            kernel_size=3, padding=1)
        self.pre_instance_norm = torch.nn.InstanceNorm2d(in_channel)
        self.visattn_conv = torch.nn.Conv2d(
            in_channel, out_channels=n_entity,
            kernel_size=3, padding=1)
        self.conv3d = torch.nn.Conv3d(
            in_channel+self.fourier_bases.channel_size, in_channel,
            kernel_size=(1, 1, 1), padding=(0, 1, 1))

    def forward(self, state):
        statistics = {}
        state = self.fourier_bases.apply(state)
        statistics["FourierState_std"] = torch.std(state)
        statistics["FourierState_mean"] = torch.mean(state)
        x = torch.relu(self.pre_conv(state))
        x = self.pre_instance_norm(x)
        statistics["PreAtten_std"] = torch.std(x)
        statistics["PreAtten_mean"] = torch.mean(x)
        attn = torch.sigmoid(self.visattn_conv(x))
        # print(attn[0, 0])
        statistics["Atten_std"] = torch.std(attn)
        statistics["Atten_mean"] = torch.mean(attn)
        x = torch.einsum("bfyx, beyx->bfeyx", state, attn)
        x = torch.relu(self.conv3d(x)).permute(0, 2, 1, 3, 4)
        statistics["EndVisAttn_std"] = torch.std(x)
        statistics["EndVisAttn_mean"] = torch.mean(x)
        self.statistics = statistics
        x = x.mean((-1, -2))
        return x, attn


class FourierBases:

    def __init__(self, u, v, height, width, device="cpu"):
        y_coords, x_coords = torch.meshgrid(
            torch.linspace(-1, 1, height),
            torch.linspace(-1, 1, width))

        even_bases = torch.stack(
            [torch.cos(pi*x_coords*u_)*torch.cos(pi*y_coords*v_)
             for v_, u_ in product(v, u)])
        odd_bases = torch.stack(
            [torch.sin(pi*x_coords*u_)*torch.sin(pi*y_coords*v_)
             for v_, u_ in product(v, u)])

        self._bases = torch.cat([even_bases, odd_bases], dim=0).to(device)
        self.bases = self._bases.unsqueeze(0)

    def apply(self, state):
        bs, channel, height, width = state.shape
        device = state.device
        if self.bases.shape[0] != bs or device != self.bases.device:
            self.bases = self._bases.unsqueeze(
                0).repeat(bs, 1, 1, 1).to(device)
        return torch.cat([state, self.bases], dim=1)

    @property
    def channel_size(self):
        return self._bases.shape[0]


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
            bs, n_entity, self.n_heads,
            self.qkv_dim//self.n_heads).permute(0, 2, 1, 3))
        query = torch.relu(query)

        key = self.key_layernorm(self.key_fc(features).reshape(
            bs, n_entity, self.n_heads,
            self.qkv_dim//self.n_heads).permute(0, 2, 1, 3))
        key = torch.relu(key)

        value = self.value_layernorm(self.value_fc(features).reshape(
            bs, n_entity, self.n_heads,
            self.qkv_dim//self.n_heads).permute(0, 2, 1, 3))
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

    def __init__(self, in_channel, out_channel, qkv_dim,
                 n_entity, n_heads, mapsize):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channel, out_channel,
                                    kernel_size=3, padding=1)
        self.visual_attn = VisualAttn(out_channel, n_entity, mapsize)
        self.self_attn = SelfAttn(out_channel, n_entity, qkv_dim, n_heads)

        self.instance_norm = torch.nn.InstanceNorm2d(qkv_dim)

    def forward(self, state):
        state = torch.relu(self.conv(state))
        features, attn = self.visual_attn(state)
        features = self.self_attn(features)
        post_state = self.self_attn.broadcast(features, attn)
        return self.instance_norm(post_state + state)


class OurAttnNet(torch.nn.Module):

    def __init__(self, in_channel, mapsize, n_act, n_entity,
                 n_heads, conv_size, attn_size, qkv_dim, dense_size, **kwargs):
        super().__init__()

        self.convnet = torch.nn.Sequential(
            torch.nn.Conv2d(in_channel, conv_size, 3, 1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(conv_size, attn_size, 3, 1, padding=1),
            torch.nn.ReLU(),
        )

        self.attn_module = OurAttnModule(attn_size, attn_size, qkv_dim,
                                         n_entity=n_entity,
                                         n_heads=n_heads,
                                         mapsize=mapsize)

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

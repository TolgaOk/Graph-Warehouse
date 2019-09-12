import torch


class EdgeGraphConv(torch.nn.Module):

    def __init__(self, n_edges, in_size, out_size):
        super().__init__()
        init_weight = torch.rand(n_edges, in_size, out_size)
        self.weight = torch.nn.Parameter(init_weight)
        self.bias = torch.nn.Parameter(torch.zeros(out_size))

    def forward(self, adjacency, feature):
        """
            feature: (B, N, F)
            adjacency: (E, N, N)
            weight: (E, F, F')
            support: (B, E, N, F')
            ouput: (B, E, N, F')
                   (B, N, F')
        """
        assert len(feature.shape) == 3, "Feature must be 3 dimensional"
        feature = torch.unsqueeze(feature, dim=1)
        support = torch.matmul(feature, self.weight)
        output = torch.matmul(adjacency, support)
        # Summing for all edge types(0th dim is the batch dim)
        output = torch.mean(output, dim=1)
        output = output + self.bias.reshape(1, 1, -1)
        output = torch.relu(output)
        return output


class GraphA2C(torch.nn.Module):
    def __init__(self, n_edge, n_node, mapsize, n_act, adjacency, eg_size, kg_size, dense_in_channel, dense_size, **kwargs):
        super().__init__()
        self.n_node = n_node
        self.adj = adjacency

        self.egconv1 = EdgeGraphConv(n_edge, n_node, eg_size)
        self.egconv2 = EdgeGraphConv(n_edge, eg_size, eg_size)
        self.egconv3 = EdgeGraphConv(n_edge, eg_size, eg_size)
        self.egconv4 = EdgeGraphConv(n_edge, eg_size, eg_size)

        self.kgconv1 = torch.nn.Conv2d(kg_size, kg_size, 3, padding=1)
        self.kgconv2 = torch.nn.Conv2d(kg_size, kg_size, 1, padding=0)
        self.kgconv3 = torch.nn.Conv2d(kg_size, dense_in_channel, 3, padding=1)
        self.kgconv4 = torch.nn.Conv2d(kg_size, dense_in_channel, 1, padding=0)

        self.instance_norm = torch.nn.InstanceNorm2d(dense_in_channel)
        init_weight = torch.rand(kg_size, kg_size)
        self.pool_weight = torch.nn.Parameter(init_weight)

        self.policy = torch.nn.Sequential(
            torch.nn.Linear(dense_in_channel, dense_size),
            torch.nn.ReLU(),
            torch.nn.Linear(dense_size, n_act)
        )
        self.value = torch.nn.Sequential(
            torch.nn.Linear(dense_in_channel, dense_size),
            torch.nn.ReLU(),
            torch.nn.Linear(dense_size, 1)
        )

        self.conv_pool = torch.nn.MaxPool2d(mapsize)

    def forward(self, objmap):

        # shape: (B, N, S, S)
        bs, _, height, width = objmap.shape

        kg_features = torch.eye(self.n_node).unsqueeze(0)*torch.ones(bs, 1, 1)
        kg_features = kg_features.to(objmap.device)

        # Graph conv
        kg_features = self.egconv1(self.adj, kg_features)
        kg_features = self.egconv2(self.adj, kg_features)

        # Broadcast
        state = self.broadcast(kg_features, objmap)

        # KG conv
        skip_state = self.kgconv1(state) + self.kgconv2(state)
        skip_state = torch.relu(skip_state)

        # Pooling
        kg_features = self.pooling(skip_state, objmap)

        #   Graph conv
        kg_features = self.egconv3(self.adj, kg_features)
        kg_features = self.egconv4(self.adj, kg_features)

        #   Broadcast
        state = self.broadcast(kg_features, objmap)

        # KG conv

        state = self.kgconv3(skip_state) + self.kgconv4(state)
        state = torch.relu(state)

        # Dense Layers
        # state = state.mean((-1, -2))
        state = self.conv_pool(state).reshape(bs, -1)
        policy = self.policy(state)
        value = self.value(state)
        return policy, value

    def broadcast(self, feature, objectmap):
        """ objectmap: (B, N, H, W)
            feature: (B, N, F)

            objectmap -> (B, N, 1, H, W)
            feature ->   (B, N, F, 1, 1)

            result = objectmap * feature
            result.sum(1)
        """
        assert len(feature.shape) == 3, ("feature argument must be 3"
                                         " dimensional. (B, N, F): batch"
                                         " size, #nodes, #features")
        assert len(objectmap.shape) == 4, ("objectmap argument must be 4"
                                           "dimensional. (B, N, H, W): "
                                           "batch size, #nodes, height, width")
        state_tensor = (objectmap.unsqueeze(2) *
                        feature.reshape(*feature.shape, 1, 1))
        return state_tensor.sum(1)

    def pooling(self, state, objectmap):
        """ objectmap: (B, N, H, W)
            state:     (B, F, H, W)
            weight:    (F, F')

            objectmap -> (B, N, 1, H, W)
            state ->     (B, 1, F, H, W)

            result = objectmap * state
            result.sum((-2, -1))
        """
        assert len(state.shape) == 4, ("state must be 4 dimensional "
                                       "(B, F, H, W): batch size, #features"
                                       ", heigth, width")
        assert len(objectmap.shape) == 4, ("objectmap argument must be 4"
                                           "dimensional. (B, N, H, W): "
                                           "batch size, #nodes, height, width")
        n_occurrence = objectmap.sum((-2, -1))
        n_occurrence[n_occurrence == 0] = 1
        feature = (objectmap.unsqueeze(2) *
                   state.unsqueeze(1))
        feature = torch.matmul(feature.sum((-2, -1)), self.pool_weight)
        # feature = torch.relu(feature)
        return feature / (n_occurrence.unsqueeze(-1))
    
    def to(self, device):
        super().to(device)
        self.adj = self.adj.to(device)

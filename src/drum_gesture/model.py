import torch


class GestureRNN(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        rnn_layers: int = 1,
        mlp_layers: int = 3,
        layer_norm: bool = False,
    ):
        super(GestureRNN, self).__init__()
        self.rnn = torch.nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=rnn_layers,
            batch_first=True,
        )

        # MLP layers
        self.mlp_layers = mlp_layers
        assert mlp_layers >= 1, "Number of MLP layers must be at least 1"

        layers = []
        for i in range(mlp_layers):
            layers.append(torch.nn.Linear(hidden_size, hidden_size))
            if layer_norm:
                layers.append(torch.nn.LayerNorm(hidden_size))
            layers.append(torch.nn.ReLU())

        layers.append(torch.nn.Linear(hidden_size, output_size))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(
        self, x: torch.Tensor, h_n: torch.Tensor = None, c_n: torch.Tensor = None
    ) -> torch.Tensor:
        out, h_n = self.rnn(x, h_n)  # GRU does not use c_n
        out = self.mlp(out)  # Apply MLP layers
        out = torch.sigmoid(out)  # Apply sigmoid activation for binary classification
        return out, h_n


class BuzzRNN(torch.nn.Module):
    def __init__(
        self, input_size: int, hidden_size: int, output_size: int, num_layers: int = 1
    ):
        super(BuzzRNN, self).__init__()
        self.rnn = torch.nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
    
        self.fc1 = torch.nn.Linear(hidden_size, hidden_size)
        #self.ln1 = torch.nn.LayerNorm(hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        #self.ln2 = torch.nn.LayerNorm(hidden_size)
        self.fc3 = torch.nn.Linear(hidden_size, hidden_size)
        #self.ln3 = torch.nn.LayerNorm(hidden_size)
        self.out_project = torch.nn.Linear(hidden_size, output_size)


    def forward(
        self, x: torch.Tensor, h_n: torch.Tensor = None, c_n: torch.Tensor = None
    ) -> torch.Tensor:
        # out, (h_n, c_n) = self.rnn(
        #     x, (h_n, c_n) if h_n is not None and c_n is not None else None
        # )
        
        out, h_n = self.rnn(x, h_n)  # GRU does not use c_n

        out = self.fc1(out)
        #out = self.ln1(out)
        out = torch.relu(out)
        out = self.fc2(out)
        #out = self.ln2(out)
        out = torch.relu(out)
        out = self.fc3(out)
        #out = self.ln3(out)
        out = torch.relu(out)
        out = self.out_project(out)

        out = torch.sigmoid(out)  # Apply sigmoid activation for binary classification
        return out, h_n
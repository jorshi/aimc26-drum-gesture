"""
Export trained models to the JSON format read by RTNeural.

The weight layout follows RTNeural's Keras-derived convention, which differs from
PyTorch's in two ways:

* GRU gates are ordered ``[z, r, n]`` rather than PyTorch's ``[r, z, n]``.
* Weight matrices are stored transposed, ``(in, out)`` rather than ``(out, in)``.

The GRU itself uses the ``reset_after=True`` formulation, which keeps the input and
recurrent biases separate::

    z = sigmoid(W_z x + b^i_z + U_z h + b^h_z)
    r = sigmoid(W_r x + b^i_r + U_r h + b^h_r)
    n = tanh(W_n x + b^i_n + r * (U_n h + b^h_n))
    h' = z * h + (1 - z) * n

That is algebraically identical to ``torch.nn.GRU``, so the conversion is a pure
reshuffle -- no approximation is involved.

``forward_from_dict`` is a numpy reference implementation of the exported model. It
exists so the export can be verified without a working RTNeural build, and doubles as
the reference for ports to other languages.
"""

import json
from typing import Any, Dict, List

import numpy as np
import torch

from drum_gesture.model import GestureRNN, BuzzRNN


def export_model(model: torch.nn.Module, output_path: str) -> Dict[str, Any]:
    """
    Export ``model`` to an RTNeural JSON file at ``output_path``.

    Returns the exported dict so callers can verify it without re-reading the file.
    """
    model_dict = model_to_dict(model)
    with open(output_path, "w") as f:
        json.dump(model_dict, f, indent=4)
    return model_dict


def model_to_dict(model: torch.nn.Module) -> Dict[str, Any]:
    """
    Convert a ``GestureRNN`` or ``BuzzRNN`` to an RTNeural model dict.
    """
    gru = _get_gru(model)
    dense_layers = _get_dense_layers(model)

    layers = [_gru_to_dict(gru)]

    # Both models apply ReLU after every hidden Linear and a sigmoid on the output.
    for i, linear in enumerate(dense_layers):
        is_output = i == len(dense_layers) - 1
        layers.append(_dense_to_dict(linear, "sigmoid" if is_output else "relu"))

    return {"in_shape": [1, 0, gru.input_size], "layers": layers}


def _get_gru(model: torch.nn.Module) -> torch.nn.GRU:
    if not isinstance(model, (GestureRNN, BuzzRNN)):
        raise TypeError(f"Expected a GestureRNN or BuzzRNN, got {type(model).__name__}")
    if not isinstance(model.rnn, torch.nn.GRU):
        raise TypeError(
            f"Expected the recurrent layer to be a GRU, got {type(model.rnn).__name__}"
        )
    if model.rnn.num_layers != 1:
        raise ValueError(
            f"Export only supports a single GRU layer, got {model.rnn.num_layers}"
        )
    if model.rnn.bidirectional:
        raise ValueError("Export does not support bidirectional GRUs")
    return model.rnn


def _get_dense_layers(model: torch.nn.Module) -> List[torch.nn.Linear]:
    """
    Return the post-GRU Linear layers, in forward order.
    """
    if isinstance(model, BuzzRNN):
        return [model.fc1, model.fc2, model.fc3, model.out_project]

    linears = []
    for layer in model.mlp:
        if isinstance(layer, torch.nn.Linear):
            linears.append(layer)
        elif isinstance(layer, torch.nn.ReLU):
            continue
        elif isinstance(layer, torch.nn.LayerNorm):
            raise ValueError(
                "LayerNorm is not supported by RTNeural; train with layer_norm=False"
            )
        else:
            raise ValueError(f"Unsupported layer type: {type(layer).__name__}")

    if not linears:
        raise ValueError("Model has no Linear layers to export")
    return linears


def _gru_to_dict(gru: torch.nn.GRU) -> Dict[str, Any]:
    with torch.no_grad():
        # (3H, N) [r|z|n] -> (N, 3H) [z|r|n]
        w_ih = _rzn_to_zrn(gru.weight_ih_l0.detach().cpu().numpy()).T
        w_hh = _rzn_to_zrn(gru.weight_hh_l0.detach().cpu().numpy()).T

        if gru.bias:
            b_ih = _rzn_to_zrn(gru.bias_ih_l0.detach().cpu().numpy())
            b_hh = _rzn_to_zrn(gru.bias_hh_l0.detach().cpu().numpy())
        else:
            b_ih = np.zeros(3 * gru.hidden_size, dtype=w_ih.dtype)
            b_hh = np.zeros_like(b_ih)

        biases = np.stack([b_ih, b_hh], axis=0)

    return {
        "type": "gru",
        "activation": "tanh",
        "shape": [1, 0, gru.hidden_size],
        "weights": [w_ih.tolist(), w_hh.tolist(), biases.tolist()],
    }


def _dense_to_dict(linear: torch.nn.Linear, activation: str) -> Dict[str, Any]:
    with torch.no_grad():
        # PyTorch stores (out, in); RTNeural expects (in, out).
        kernel = linear.weight.detach().cpu().numpy().T
        if linear.bias is not None:
            bias = linear.bias.detach().cpu().numpy()
        else:
            bias = np.zeros(linear.out_features, dtype=kernel.dtype)

    return {
        "type": "dense",
        "activation": activation,
        "shape": [1, 0, linear.out_features],
        "weights": [kernel.tolist(), bias.tolist()],
    }


def _rzn_to_zrn(x: np.ndarray) -> np.ndarray:
    """
    Reorder the leading axis from PyTorch's [r|z|n] gate order to Keras' [z|r|n].
    """
    r, z, n = np.split(x, 3, axis=0)
    return np.concatenate([z, r, n], axis=0)


def _zrn_to_rzn(x: np.ndarray) -> np.ndarray:
    """
    Inverse of ``_rzn_to_zrn``.
    """
    z, r, n = np.split(x, 3, axis=0)
    return np.concatenate([r, z, n], axis=0)


def model_from_dict(
    model_dict: Dict[str, Any], dtype: torch.dtype = torch.float32
) -> "GestureRNN":
    """
    Load an RTNeural model dict back into a PyTorch model.

    The inverse of ``model_to_dict``. Useful for evaluating a model that only exists as
    exported JSON -- including one produced by a non-Python trainer, which has no
    ``model.pt`` to load.

    Always returns a ``GestureRNN``, since it is the shape-flexible of the two classes.
    ``BuzzRNN`` computes the same function with a fixed three hidden layers.
    """
    layers = model_dict["layers"]
    if not layers or layers[0]["type"] != "gru":
        raise ValueError("Expected the first layer to be a gru")

    dense_layers = [layer for layer in layers[1:] if layer["type"] == "dense"]
    if len(dense_layers) != len(layers) - 1:
        kinds = [layer["type"] for layer in layers[1:]]
        raise ValueError(f"Expected only dense layers after the gru, got {kinds}")

    input_size = model_dict["in_shape"][2]
    hidden_size = layers[0]["shape"][2]
    output_size = dense_layers[-1]["shape"][2]

    model = GestureRNN(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        rnn_layers=1,
        mlp_layers=len(dense_layers) - 1,
    ).to(dtype)

    gru = layers[0]
    with torch.no_grad():
        model.rnn.weight_ih_l0.copy_(
            torch.tensor(_zrn_to_rzn(np.array(gru["weights"][0]).T), dtype=dtype)
        )
        model.rnn.weight_hh_l0.copy_(
            torch.tensor(_zrn_to_rzn(np.array(gru["weights"][1]).T), dtype=dtype)
        )
        biases = np.array(gru["weights"][2])
        model.rnn.bias_ih_l0.copy_(torch.tensor(_zrn_to_rzn(biases[0]), dtype=dtype))
        model.rnn.bias_hh_l0.copy_(torch.tensor(_zrn_to_rzn(biases[1]), dtype=dtype))

        linears = [layer for layer in model.mlp if isinstance(layer, torch.nn.Linear)]
        if len(linears) != len(dense_layers):
            raise ValueError(
                f"Rebuilt model has {len(linears)} dense layers, dict has {len(dense_layers)}"
            )
        for linear, layer in zip(linears, dense_layers):
            linear.weight.copy_(
                torch.tensor(np.array(layer["weights"][0]).T, dtype=dtype)
            )
            linear.bias.copy_(torch.tensor(np.array(layer["weights"][1]), dtype=dtype))

    return model.eval()


# ---------------------------------------------------------------------------
# numpy reference implementation of the exported model
# ---------------------------------------------------------------------------


def forward_from_dict(
    model_dict: Dict[str, Any], x: np.ndarray, h0: np.ndarray = None
) -> np.ndarray:
    """
    Run an exported RTNeural model dict over a sequence, in numpy.

    Args:
        model_dict: as produced by ``model_to_dict``.
        x: input of shape (time, in_features).
        h0: optional initial GRU state of shape (hidden,). Defaults to zeros.

    Returns:
        Output of shape (time, out_features).
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"Expected (time, features) input, got shape {x.shape}")

    out = x
    for layer in model_dict["layers"]:
        if layer["type"] == "gru":
            out = _gru_forward(layer, out, h0)
        elif layer["type"] == "dense":
            out = _dense_forward(layer, out)
        else:
            raise ValueError(f"Unsupported layer type in model dict: {layer['type']}")

    return out


def _gru_forward(
    layer: Dict[str, Any], x: np.ndarray, h0: np.ndarray = None
) -> np.ndarray:
    w_ih = np.asarray(layer["weights"][0], dtype=np.float64)  # (in, 3H)
    w_hh = np.asarray(layer["weights"][1], dtype=np.float64)  # (H, 3H)
    biases = np.asarray(layer["weights"][2], dtype=np.float64)  # (2, 3H)
    b_ih, b_hh = biases[0], biases[1]

    hidden = w_hh.shape[0]
    h = np.zeros(hidden) if h0 is None else np.asarray(h0, dtype=np.float64).copy()

    outputs = np.empty((x.shape[0], hidden))
    for t in range(x.shape[0]):
        gates_x = x[t] @ w_ih + b_ih
        gates_h = h @ w_hh + b_hh

        xz, xr, xn = np.split(gates_x, 3)
        hz, hr, hn = np.split(gates_h, 3)

        z = _sigmoid(xz + hz)
        r = _sigmoid(xr + hr)
        n = np.tanh(xn + r * hn)

        h = z * h + (1.0 - z) * n
        outputs[t] = h

    return outputs


def _dense_forward(layer: Dict[str, Any], x: np.ndarray) -> np.ndarray:
    kernel = np.asarray(layer["weights"][0], dtype=np.float64)  # (in, out)
    bias = np.asarray(layer["weights"][1], dtype=np.float64)  # (out,)
    out = x @ kernel + bias

    activation = layer["activation"]
    if activation == "relu":
        return np.maximum(out, 0.0)
    if activation == "sigmoid":
        return _sigmoid(out)
    if activation == "tanh":
        return np.tanh(out)
    if activation in ("", "linear"):
        return out
    raise ValueError(f"Unsupported activation: {activation}")


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

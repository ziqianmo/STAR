"""Load STAR weights while requiring a complete, compatible retrieval backbone."""

from collections.abc import Mapping

import torch


def load_checkpoint(model, filename):
    state = torch.load(filename, map_location="cpu", weights_only=True)
    for _ in range(3):
        if not isinstance(state, Mapping):
            break
        nested = next(
            (
                state[key]
                for key in ("state_dict", "model_state_dict", "model")
                if key in state and isinstance(state[key], Mapping)
            ),
            None,
        )
        if nested is None:
            break
        state = nested
    if (
        not isinstance(state, Mapping)
        or not state
        or not all(isinstance(k, str) and torch.is_tensor(v) for k, v in state.items())
    ):
        raise ValueError("Checkpoint must contain a model state_dict of tensors.")
    state = {key.removeprefix("module."): value for key, value in state.items()}
    expected = model.state_dict()
    required = [key for key in expected if key.startswith("model_1.convnext.")]
    if not required:
        raise ValueError(
            "The evaluation model does not expose a STAR retrieval backbone."
        )
    incompatible = [
        key
        for key in required
        if key not in state or state[key].shape != expected[key].shape
    ]
    if incompatible:
        raise ValueError(
            "Checkpoint has missing or incompatible backbone weights: "
            + ", ".join(incompatible[:5])
        )
    compatible = {
        key: value
        for key, value in state.items()
        if key in expected and value.shape == expected[key].shape
    }
    model.load_state_dict(compatible, strict=False)
    return {
        "loaded_keys": len(compatible),
        "backbone_keys": len(required),
        "skipped_checkpoint_keys": sorted(set(state) - set(compatible)),
        "unloaded_model_keys": sorted(set(expected) - set(compatible)),
    }

"""Grad-CAM: visual explanation of where the model "looks".

Implemented from scratch with forward/backward hooks so it has no extra
dependency. Returns a [0, 1] heatmap the size of the input image.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    """Compute a Grad-CAM heatmap for a target conv layer."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model.eval()
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        target_layer.register_forward_hook(self._save_activations)
        target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, _module, _inp, output) -> None:
        self.activations = output.detach()

    def _save_gradients(self, _module, _grad_in, grad_out) -> None:
        self.gradients = grad_out[0].detach()

    @torch.enable_grad()
    def __call__(
        self, input_tensor: torch.Tensor, class_idx: int | None = None
    ) -> tuple[np.ndarray, int]:
        """Return (heatmap HxW in [0,1], predicted_or_target_class_idx)."""
        input_tensor = input_tensor.requires_grad_(True)
        logits = self.model(input_tensor)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())

        self.model.zero_grad()
        logits[0, class_idx].backward()

        # weight each activation channel by the mean of its gradient
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1,1,h,w)
        cam = F.relu(cam)
        cam = F.interpolate(
            cam,
            size=input_tensor.shape[2:],
            mode="bilinear",
            align_corners=False,
        )
        cam = cam.squeeze().cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam, class_idx

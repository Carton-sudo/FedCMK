"""
Model decoupling: split FedAvgNet{CIFAR,MNIST} into Body + G-Head.

Body for CIFAR:
    conv2d_1 → relu → maxpool → conv2d_2 → relu → maxpool → flatten → linear_1 → relu

G-Head for CIFAR:
    classifier  (Linear(512, num_classes))

Body for MNIST:
    conv2d_1 → relu → maxpool → conv2d_2 → relu → maxpool → flatten → linear_1 → relu

G-Head for MNIST:
    linear_2  (Linear(512, num_classes))
"""

import torch
import torch.nn as nn
import copy


# ---------------------------------------------------------------------------
#  Properly-defined Body modules (with their own ReLU instances)
# ---------------------------------------------------------------------------
class CIFARBody(nn.Module):
    """Feature extractor for CIFAR-10/100 — outputs 512-dim features."""
    def __init__(self, conv2d_1, conv2d_2, max_pooling, flatten, linear_1):
        super().__init__()
        self.conv2d_1 = conv2d_1
        self.relu1 = nn.ReLU()
        self.max_pooling = max_pooling
        self.conv2d_2 = conv2d_2
        self.relu2 = nn.ReLU()
        self.flatten = flatten
        self.linear_1 = linear_1
        self.relu3 = nn.ReLU()

    def forward(self, x):
        x = self.conv2d_1(x)
        x = self.relu1(x)
        x = self.max_pooling(x)
        x = self.conv2d_2(x)
        x = self.relu2(x)
        x = self.max_pooling(x)
        x = self.flatten(x)
        x = self.linear_1(x)
        x = self.relu3(x)
        return x


class MNISTBody(nn.Module):
    """Feature extractor for MNIST — outputs 512-dim features."""
    def __init__(self, conv2d_1, conv2d_2, max_pooling, flatten, linear_1):
        super().__init__()
        self.conv2d_1 = conv2d_1
        self.relu1 = nn.ReLU()
        self.max_pooling = max_pooling
        self.conv2d_2 = conv2d_2
        self.relu2 = nn.ReLU()
        self.flatten = flatten
        self.linear_1 = linear_1
        self.relu3 = nn.ReLU()

    def forward(self, x):
        if x.ndim < 4:
            x = torch.unsqueeze(x, 1)
        x = self.conv2d_1(x)
        x = self.relu1(x)
        x = self.max_pooling(x)
        x = self.conv2d_2(x)
        x = self.relu2(x)
        x = self.max_pooling(x)
        x = self.flatten(x)
        x = self.linear_1(x)
        x = self.relu3(x)
        return x


# ---------------------------------------------------------------------------
#  Decouple & Assemble
# ---------------------------------------------------------------------------
def decouple_fedavgnet(model):
    """Split a FedAvgNet{CIFAR,MNIST} into (body, g_head)."""
    head_names = {"classifier", "linear_2"}
    classname = model.__class__.__name__

    # Collect components by name
    components = {}
    for name, module in model.named_children():
        components[name] = module

    # Extract shared components
    conv2d_1 = components["conv2d_1"]
    conv2d_2 = components["conv2d_2"]
    max_pooling = components["max_pooling"]
    flatten = components["flatten"]
    linear_1 = components["linear_1"]

    # Build body
    if "MNIST" in classname:
        body = MNISTBody(conv2d_1, conv2d_2, max_pooling, flatten, linear_1)
    else:
        body = CIFARBody(conv2d_1, conv2d_2, max_pooling, flatten, linear_1)

    # Find G-Head
    g_head = None
    for name in head_names:
        if name in components:
            g_head = components[name]
            break
    if g_head is None:
        raise ValueError(f"Cannot find head. Looked for: {head_names}")

    # Verify
    dummy = torch.randn(2, *get_input_shape(model))
    with torch.no_grad():
        feat = body(dummy)
    assert feat.shape[1] == 512, f"Expected 512-dim features, got {feat.shape[1]}"

    return body, g_head


def get_input_shape(model):
    classname = model.__class__.__name__
    if "MNIST" in classname:
        return (1, 28, 28)
    return (3, 32, 32)


def assemble_model(body, g_head):
    """Reassemble for evaluation — matches FedAvgNetCIFAR forward()."""
    class AssembledModel(nn.Module):
        def __init__(self, body, g_head):
            super().__init__()
            self.body = body
            self.classifier = g_head

        def forward(self, x, get_features=False):
            z = self.body(x)
            x = self.classifier(z)
            if get_features:
                return x, z
            return x

    return AssembledModel(body, g_head)

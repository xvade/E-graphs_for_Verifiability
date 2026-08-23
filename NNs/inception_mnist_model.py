# A small MNIST CNN with a genuine parallel-branch structure: two convs
# (1x1 and 3x3, both stride=1) applied to the SAME upstream tensor, merged
# by addition afterward. Built specifically because no real-weight model
# in alpha-beta-CROWN's examples has this shape -- every ResNet-family
# model there only ever has a parallel shortcut conv at stride=2
# (downsampling), and tensat's parallel-conv-fusion rule (PRE_DEFINED_MULTI
# in tensat/src/rewrites.rs) hardcodes stride=1. This one's branchA/branchB
# share input `x` at stride=1, matching that rule's literal pattern
# exactly (padding both import as SAME/0 since PyTorch exports explicit
# nonzero `pads`, and neither branch has a fused activation, so both
# import with activation=NONE/0 -- see taso/python/taso/__init__.py's
# _get_conv_pool_pads_attr).
import torch
import torch.nn as nn
import torch.nn.functional as F


class InceptionMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(1, 8, 3, stride=1, padding=1)
        self.branchA = nn.Conv2d(8, 8, 1, stride=1, padding=0)
        self.branchB = nn.Conv2d(8, 8, 3, stride=1, padding=1)
        self.fc1 = nn.Linear(8 * 28 * 28, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.stem(x))
        a = self.branchA(x)
        b = self.branchB(x)
        x = F.relu(a + b)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

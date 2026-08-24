# Structural diversity vs. verifiability -- sweep results


## inception_mnist / fused_auto / repvar1

- Structure: 33 nodes, depth 15, max branching 2, Concat/Split present: True
  - axes: [{'guid': 499, 'op': 'Concat', 'axis': 1}, {'guid': 501, 'op': 'Split', 'axis': 1}]
  - op counts: {'Add': 6, 'Concat': 1, 'Conv': 3, 'Input': 1, 'Matmul': 2, 'Relu': 3, 'Reshape': 4, 'Split': 1, 'Transpose': 2, 'Weight': 10}

| epsilon | verified_acc% | verified/total | mean_time(s) | max_time(s) | wall_time(s) |
|---|---|---|---|---|---|
| 0.1 | 10.0 | 1/10 | 55.95 | 66.20 | 828.6 |

## inception_mnist / fused_v2 / handverified

- Structure: 33 nodes, depth 15, max branching 2, Concat/Split present: True
  - axes: [{'guid': 497, 'op': 'Concat', 'axis': 1}, {'guid': 499, 'op': 'Split', 'axis': 1}]
  - op counts: {'Add': 6, 'Concat': 1, 'Conv': 3, 'Input': 1, 'Matmul': 2, 'Relu': 3, 'Reshape': 4, 'Split': 1, 'Transpose': 2, 'Weight': 10}

| epsilon | verified_acc% | verified/total | mean_time(s) | max_time(s) | wall_time(s) |
|---|---|---|---|---|---|
| 0.02 | 90.0 | 9/10 | 11.43 | 107.64 | 161.9 |
| 0.05 | 50.0 | 5/10 | 35.37 | 99.02 | 504.1 |
| 0.1 | 10.0 | 1/10 | 55.46 | 63.85 | 734.8 |
| 0.15 | 0.0 | 0/10 | 62.44 | 64.19 | 715.9 |
| 0.2 | 0.0 | 0/10 | 62.77 | 64.38 | 896.7 |

## inception_mnist / unfused / baseline

- Structure: 31 nodes, depth 13, max branching 2, Concat/Split present: False
  - op counts: {'Add': 6, 'Conv': 3, 'Input': 1, 'Matmul': 2, 'Relu': 3, 'Reshape': 4, 'Transpose': 2, 'Weight': 10}

| epsilon | verified_acc% | verified/total | mean_time(s) | max_time(s) | wall_time(s) |
|---|---|---|---|---|---|
| 0.02 | 90.0 | 9/10 | 7.86 | 60.36 | 295.4 |
| 0.05 | 70.0 | 7/10 | 19.53 | 63.64 | 345.6 |
| 0.1 | 20.0 | 2/10 | 51.75 | 65.93 | 631.2 |
| 0.15 | 10.0 | 1/10 | 68.41 | 188.78 | 754.1 |
| 0.2 | 0.0 | 0/10 | 63.85 | 66.16 | 699.0 |

## mnist_cnn_a / unfused / baseline

- Structure: 25 nodes, depth 12, max branching 2, Concat/Split present: False
  - op counts: {'Add': 4, 'Conv': 2, 'Input': 1, 'Matmul': 2, 'Relu': 3, 'Reshape': 3, 'Transpose': 2, 'Weight': 8}

| epsilon | verified_acc% | verified/total | mean_time(s) | max_time(s) | wall_time(s) |
|---|---|---|---|---|---|
| 0.1 | 100.0 | 10/10 | 1.23 | 9.28 | 95.1 |

## resnet2b / unfused / baseline

- Structure: 50 nodes, depth 23, max branching 2, Concat/Split present: False
  - op counts: {'Add': 10, 'Conv': 6, 'Input': 1, 'Matmul': 2, 'Relu': 6, 'Reshape': 7, 'Transpose': 2, 'Weight': 16}

| epsilon | verified_acc% | verified/total | mean_time(s) | max_time(s) | wall_time(s) |
|---|---|---|---|---|---|
| 0.031 | 0.0 | 0/10 | 67.92 | 70.12 | 743.3 |

import torch
import torch.nn as nn

pool = nn.MaxPool2d((2, 2), stride=2, return_indices=True)
input = torch.tensor([[[[ 1.,  2,  3,  4],
                            [ 5,  6,  7,  8],
                            [ 9, 10, 11, 12],
                            [13, 14, 15, 16]]]])
print(input.shape) # torch.Size([1, 1, 4, 4])

output, indices = pool(input)
print(output)       # 4*4の行列の最大値
print(output.shape) # torch.Size([1, 1, 2, 2])
print(indices)      # 4*4の行列を1次元にしたインデックスを考える

unpool = nn.MaxUnpool2d(2, stride=2)

result1 = unpool(output, indices)
result2 = unpool(output, indices, output_size=torch.Size([1, 1, 5, 5]))

print(result1) # 最大値は同じ値、それ以外は全て0
print(result1.shape)
print(result2)
print(result2.shape)
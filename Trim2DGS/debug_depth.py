import torch
import numpy as np

depths = torch.tensor(np.arange(1, 11), dtype=torch.float32, requires_grad=True)
FAR_PLANE = 100.0
NEAR_PLANE = 0.2
def turn_to_md(c_d):
    return c_d
    return (FAR_PLANE * c_d - FAR_PLANE * NEAR_PLANE) / ((FAR_PLANE - NEAR_PLANE) * c_d)

T = 1.0
opac = 0.5
rend_depth = 0.0
for i in range(10):
    if T < 0.0001:
        break
    rend_depth += turn_to_md(depths[i]) * T * opac
    T *= (1.0 - opac)

loss = 0.0
for i in range(10):
    loss += 0.5 * (turn_to_md(depths[i]) - rend_depth) ** 2

loss.backward()
print(depths.grad)
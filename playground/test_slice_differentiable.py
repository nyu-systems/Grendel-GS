import torch

# Create a tensor that requires gradient
x = torch.randn(3, 3, requires_grad=True)
x0 = torch.randn(3, requires_grad=True)
# Perform a partial update
# Let's update the first row of the tensor
x[0, :] = x[0, :] + x0

# Use the updated tensor in a subsequent operation
y = x.sum()

# Compute gradients
y.backward()

# x.grad will contain the gradients of y with respect to x
print(x.grad)

# (base) [hz3496@log-2 playground]$ python test_slice_differentiable.py 
# \Traceback (most recent call last):
#   File "/home/hz3496/gaussian-splatting/playground/test_slice_differentiable.py", line 8, in <module>
#     x[0, :] = x[0, :] + x0
#     ~^^^^^^
# RuntimeError: a view of a leaf Variable that requires grad is being used in an in-place operation.
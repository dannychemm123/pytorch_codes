import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torchvision


fx = lambda x: x**2 + 1.5*x - 1
x = np.linspace(-10, 8.5, 100)
# plt.plot(x, fx(x), label='y = x^2 + 2x + 1')
# plt.show()


x_ = torch.randn(1)
x_.requires_grad = True

x_logger = []
y_logger = []

counter = 0

learning_rate = 0.01

dy_dx = 1000
max_iter = 1000

while np.abs(dy_dx)>0.001:
    y_ = fx(x_)
    y_.backward()
    dy_dx = x_.grad.item()

    with torch.no_grad():
        x_ -= learning_rate * x_.grad
        x_.grad.zero_()

        x_logger.append(x_.item())
        y_logger.append(y_.item())

    counter += 1

    if counter == max_iter:
        print("Maximum iterations reached.")
        break

print("Y minimum is %.2f and is when X = %.2f, found after %d steps" % (y_.item(), x_.item(), counter))

plt.plot(x_logger, y_logger, '-o', markersize=3)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Gradient descent on y = x^2 + 2x + 1')
plt.grid(True)
plt.show()
# optionally: plt.savefig('convergence.png', dpi=150)
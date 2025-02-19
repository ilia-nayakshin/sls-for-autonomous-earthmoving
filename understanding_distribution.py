import matplotlib.pyplot as plt
import numpy as np

A = 100
B = 10
C = 1

z = np.linspace(0, 2, 1000)
x = A * z/(B - C * z)

PIX_LEN = x[-1] / 1000

x_mod_pix_len = x % PIX_LEN
x_diff = x_mod_pix_len - PIX_LEN/2

print(x_diff)

# plt.plot(z, x)
# plt.plot(z, x_mod_pix_len)
plt.hist(x_diff, 20)
plt.show()
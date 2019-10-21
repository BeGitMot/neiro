import math
def tnh(x):
    return 1 - math.tanh(x)**2

x = 100.0
res1 = []
res2 = []
for _ in range(4):
    x = tnh(x)
    res1 += [x]

print(res1)





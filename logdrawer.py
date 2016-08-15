import numpy as np
import matplotlib.pyplot as plt

plt.interactive(False)
f = open('losslog.txt', 'r')
lines = f.read().split('\n')
f.close()
filtered = []
for line in lines:
    if line.find('- loss: ') != -1:
        filtered.append(float(line.split(' ')[-1]))
means = []
for i in range(len(filtered)/10):
    means.append(sum(filtered[i*10:(i+1)*10])/10)

step = 100
x = np.arange(step, step*(len(means))+1, step)
plt.plot(x, means)
plt.show()


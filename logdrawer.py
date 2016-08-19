import numpy as np
import matplotlib.pyplot as plt

plt.interactive(False)
step = 50
epochs = step

f = open('losslog.txt', 'r')
lines = f.read().split('\n')
f.close()
filtered_loss = []
count_e = 0.0
count_y = 0.0
count_w = 0.0
finished = []
walls = []
filtered_reaches = []
for line in lines:
    if line.find('- loss: ') != -1:
        filtered_loss.append(float(line.split(' ')[-1]))
    if line.find('Epoch') != -1:
        filtered_reaches.append(0)
    if line.find('FINISHED') != -1:
        filtered_reaches.append(1)
    if line.find('WALL') != -1:
        filtered_reaches.append(2)

for i in filtered_reaches:
    if i == 1:
        count_y += 1
    elif i == 0:
        count_e += 1
    else:
        count_w += 1
    if count_e % step == 0 and count_e:
        finished.append(count_y/count_e)
        walls.append(count_w/count_e)
        count_y, count_e, count_w = 0.0, 0.0, 0.0


means = []

for i in range(len(filtered_loss)/epochs):
    means.append(sum(filtered_loss[i*epochs:(i+1)*epochs])/epochs)


x = np.arange(step, step*(len(means))+1, step)
print len(finished), len(walls), len(x)
plt.plot(x, means, color='b', lw=2)
plt.plot(x, finished, color='g', lw=2)
plt.plot(x, walls, color='r', lw=2)


#x = range(0, 5000, 1)
#y = [0.05 + 0.95*0.999**i for i in x]
#y = [10*0.9**i for i in x]
#plt.plot(x, y, color='b')
plt.show()

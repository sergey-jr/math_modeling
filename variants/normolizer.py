import json
import numpy as np
import matplotlib.pyplot as plt

for i in range(1, 16):
    name = f'var{i}'
    file = open(f'{name}.prn')
    x = file.read().strip().split('\n')
    x = np.array(x, dtype=float)
    fig = plt.figure(dpi=100)
    plt.hist(x, 9)
    fig.savefig(f'./normalized/{name}.png')
    plt.close(fig=fig)
    file = open(f'./normalized/{name}.json', mode='w')
    x = list(x)
    file.write(json.dumps({"x": x}))
    file.close()

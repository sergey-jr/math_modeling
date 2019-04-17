import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import json
import math


def set_narrow(arr, arr1):
    arr = list(arr)
    arr1 = list(arr1)
    arr2 = list(arr)
    i = 0
    while any([item < 5 for item in arr2]):
        if i < len(arr2):
            if arr2[i] < 5:
                if i == 0:
                    arr[i + 1] += arr[i]
                    arr2[i + 1] += arr[i]
                elif i == len(arr2) - 2:
                    arr[i] += arr[i + 1]
                    arr2[i] += arr[i + 1]
                    arr2.pop()
                    arr1.pop()
                    i -= 1
                else:
                    arr[i - 1] += arr[i]
                    arr2[i - 1] += arr[i]
                if i not in [len(arr2) - 2, len(arr2) - 1]:
                    arr2.pop(i)
                    arr1.pop(i)
                    i -= 1
            i += 1
        else:
            break
    return np.array(arr2), np.array(arr1)


variant = int(input("Вариант: "))
file = open(f'./variants/normalized/var{variant}.json')
data = json.loads(file.read())
x = np.array(data['x'], dtype=float)
N = len(x)  # count of points
m = 9  # round(3.32 * math.log(N) + 1)
p, intervals = np.histogram(x, m)  # p - array of count of points that fall into the intervals
a, b = x.min(), x.max()
print(a, b)
print("Исходное множество:")
print("p=", p)
print("delta=", intervals)
print("m=", m)
p, delta = set_narrow(p, intervals)  # narrowing of the set
m1 = len(p)
print("Суженное множество:")
print("p=", p)
print("delta=", delta)
print("m1=", m1)
# plotting bar chart
X = np.array([(delta[j] + delta[j + 1]) / 2 for j in range(m1)])
Y = np.array([p[j] / (delta[j + 1] - delta[j]) for j in range(m1)])
fig = plt.figure(dpi=100)
plt.bar(X, Y, 1)
# set interval min=-inf; max=inf
delta[0] = -np.inf
delta[-1] = np.inf
# setting distribution type
if data['low'] == 'lognorm':
    mu, sigma = np.log(x).mean(), np.sqrt(np.log(x).var())
    dist = stats.lognorm(sigma, scale=np.exp(mu))
    print(mu, sigma)
elif data['low'] == 'exp':
    la = 1 / x.mean()
    dist = stats.expon(scale=1 / la)
    print(la)
else:
    dist = stats.uniform(a, b)
    print(a, b)
# setting real percentage of fall into intervals multiply by N
nt = np.array([dist.cdf(delta[j + 1]) - dist.cdf(delta[j]) for j in range(m1 - 1)]) * N
print("nt", nt)
# calculating chi
chi = np.array([(p[j] - nt[j]) ** 2 / nt[j] for j in range(m1 - 1)]).sum()
# finding table value
krit = stats.chi2.ppf(1 - 0.05, m - 3)
# plotting
h = 10 ** -3
r = np.arange(a - h * 2, b + h * 2, h)
y1 = dist.pdf(r) * N
plt.plot(r, y1, linewidth=2, color='y')
plt.show()
fig.savefig('x-var-{}.png'.format(variant))
print(chi, krit)

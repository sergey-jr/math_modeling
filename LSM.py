from numpy import array, linspace, ones, matmul, exp, cos, sin, log, sqrt, ndarray
from numpy.linalg import inv
import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.stats as stats
from scitools.StringFunction import StringFunction
import json


def Y2(x, a, b, c):
    return a * x * x + b * x + c


def Y2_arr(x):
    return array([x * x, x, 1])


def centered(txt, sym):
    return format(txt, sym + '^100')


def vec_to_str(arr):
    if arr.dtype == float or type(arr[0]) == float:
        return '\t'.join(['{:.5}'.format(item) for item in arr])
    else:
        return '\t'.join([str(item) for item in arr])


def matrix_to_str(arr):
    return '\n'.join([vec_to_str(item) for item in arr])


v = int(input("Вариант: "))
file = open('in.json')
s = file.read()
alpha = 0.05
data = json.loads(s)[v - 1]
functions = data['functions']
f = StringFunction(functions, independent_variables=('x', 'a', 'b', 'c', 'd'), globals=globals())
f_arr = data['f_arr']
x = array(data['x'])
N = len(x)  # count of points
y = array(data['y'])
x_lin = linspace(x.min(), x.max(), 1000)  # divide interval for 1000 points

print(centered('Y', '='))
w, _ = opt.curve_fit(f, x, y)  # finding coefficients of model Y
print("b=", f"[{vec_to_str(w)}]")
y_model = f(x_lin, *w)  # for fit Y
y_new = f(x, *w)  # for finding q_glob, q_reg
f = StringFunction(f_arr, globals=globals())
fi = array([array(f(item)) for item in x])
fi_t = fi.transpose()
fi_t_fi = matmul(fi_t, fi)
c = inv(fi_t_fi)
q_glob = ((y - y.mean()) ** 2).sum()
q_reg = ((y_new - y.mean()) ** 2).sum()
r = q_reg / q_glob
s = 1 / (N - len(w)) * ((y - y_new) ** 2).sum()
D = array([s * array(f(item)).transpose().dot(c).dot(array(f(item))) for item in x])
# Student's criteria finding
tk = stats.t.ppf(1 - alpha / 2, N - len(w))  # finding table value of T-distribution for model Y
STD = array([w[i] / (sqrt(s * c[i][i])) for i in range(len(w))])
# Fisher's criteria finding
fk = stats.f.ppf(1 - alpha, len(w) - 1, N - len(w))  # finding table value of F-distribution for model Y
q_ost = q_glob - q_reg
F = (q_reg / (len(w) - 1)) / (q_ost / (N - len(w)))
print('Q_общ={:.5} Q_рег={:.5} s^2={:.5} r^2={:.2%} d={:.5}'.format(q_glob, q_reg, pow(s, 0.5), r, D.max()))
print(centered('Распределение Стьюдента', ' '))
print('STD=', "[{}]".format(vec_to_str(STD)))
print("T_кр=", "{:.5}".format(tk))
print("C:")
print(matrix_to_str(c))
print(centered('Распределение Фишера', ' '))
print('Q_рег={:.5} Q_ост={:.5}'.format(q_reg, q_ost))
print("F=", F)
print("F_кр=", "{:.5}".format(fk))

print(centered('Y2', '='))
w1, _ = opt.curve_fit(Y2, x, y)  # finding coefficients of model Y2
print("b=", f"[{vec_to_str(w1)}]")
y2 = Y2(x_lin, *w1)  # for fit
y2_new = Y2(x, *w1)  # for finding q_glob, q_reg
fi = array([Y2_arr(item) for item in x])
fi_t = fi.transpose()
fi_t_fi = matmul(fi_t, fi)
c = inv(fi_t_fi)
q_reg = ((y2_new - y.mean()) ** 2).sum()
s = 1 / (N - len(w1)) * ((y - y2_new) ** 2).sum()
r = q_reg / q_glob
D = array([s * Y2_arr(item).transpose().dot(c).dot(Y2_arr(item)) for item in x])
# Student's criteria finding
tk_1 = stats.t.ppf(1 - alpha / 2, N - len(w1))  # finding table value of T-distribution for model Y2
STD = array([w1[i] / (sqrt(s * c[i][i])) for i in range(len(w1))])
# Fisher's criteria finding
fk_1 = stats.f.ppf(1 - alpha, len(w1) - 1, N - len(w1))  # finding table value of F-distribution for model Y2
q_ost = q_glob - q_reg
F = (q_reg / (len(w1) - 1)) / (q_ost / (N - len(w1)))
print('Q_общ={:.5} Q_рег={:.5} s^2={:.5} r^2={:.2%} d={:.5}'.format(q_glob, q_reg, pow(s, 0.5), r, D.max()))
print(centered('Распределение Стьюдента', ' '))
print("C:")
print(matrix_to_str(c))
print('STD=', "[{}]".format(vec_to_str(STD)))
print("T_кр=", "{:.5}".format(tk_1))
print(centered('Распределение Фишера', ' '))
print('Q_рег={:.5} Q_ост={:.5}'.format(q_reg, q_ost))
print("F=", F)
print("F_кр=", "{:.5}".format(fk_1))

fig = plt.figure(dpi=100)
plt.plot(x, y, "ko", label="y")
plt.plot(x_lin, y_model, "b", label="Y")
plt.plot(x_lin, y2, "r", label="Y2")
plt.title("Least squares regression")
plt.legend(loc="lower right")
plt.show()
fig.savefig('var-{}.png'.format(v))

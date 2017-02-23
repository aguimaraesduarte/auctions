import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import triang
import random
from collections import OrderedDict

step = 0.001

################# PLOT
def plot_functions(func, list_n, rng, title=""):
	x = np.arange(rng[0], rng[1], step)
	for n in list_n:
		plt.plot(x, [func(v, n) for v in x], label="n={}".format(n))
	plt.legend(loc=2)
	plt.title(title)
	plt.show()

################# UNIFORM [0,1]
## Bid functions
def fp_Uni_0_1(v, n):
	return ((n-1.)*v)/n

def sp_Uni_0_1(v, n):
	return v

## Revenue
def fp_rev_Uni_0_1(n):
	return (n-1.)/(n+1.)

def sp_rev_Uni_0_1(n):
	return (n-1.)/(n+1.)

# St. Dev
def fp_stdev_Uni_0_1(n):
	return np.sqrt((((n-1.)**2)/(n*(n+2)))-fp_rev_Uni_0_1(n)**2)

def sp_stdev_Uni_0_1(n):
	return np.sqrt(((n*(n-1.))/((n+1)*(n+2)))-sp_rev_Uni_0_1(n)**2)

################# UNIFORM [5,10]
# Bid functions
def fp_Uni_5_10(v, n):
	return ((n-1.)*v+5)/n

def sp_Uni_5_10(v, n):
	return v

# Revenue
def fp_rev_Uni_5_10(n):
	return 5*(n-1.)/(n+1.)

def sp_rev_Uni_5_10(n):
	return 5*(n-1.)/(n+1.)

# St. Dev
def fp_stdev_Uni_5_10(n):
	return np.sqrt((5**2)*(((n-1.)**2)/(n*(n+2)))-fp_rev_Uni_5_10(n)**2) #not sure

def sp_stdev_Uni_5_10(n):
	return np.sqrt((5**2)*((n*(n-1.))/((n+1)*(n+2)))-sp_rev_Uni_5_10(n)**2) #not sure

################# TRIANGLE [0, 1]
def fp_triangle(v, n):
	if v == 0:
		return 0
	x_array = np.arange(0, v+step, step)
	y_array = [Fn1_triangle(x, n) for x in x_array]
	return v - np.trapz(y=y_array, x=x_array)/Fn1_triangle(v, n)

def Fn1_triangle(x, n):
	if x <= 0.5:
		return ((2*x**2)**(n-1))
	else:
		return ((1-2*((1-x)**2))**(n-1))

def sp_triange(v, n):
	return v

# Revenue
def fp_rev_triangle(n):
	x_array = np.arange(0, 1+step, step) # should not be 1 I think
	y_array = [fp_triangle(x, n)*x for x in x_array]
	return np.trapz(y=y_array, x=x_array)

def sp_rev_triangle(n):
	x_array_1 = np.arange(0, .5+step, step)
	x_array_2 = np.arange(.5, 1+step, step)
	y_array_1 = [(x**2)*(1-2*x**2)*(x**(2*n-4)) for x in x_array_1]
	y_array_2 = [((1-x)**3)*((1-2*(1-x**2))**(n-2))*x for x in x_array_2]
	return n*(n-1)*(2**n)*np.trapz(y=y_array_1, x=x_array_1) + 8*n*(n-1)*np.trapz(y=y_array_2, x=x_array_2)

# St. Dev
def fp_stdev_triangle(n):
	x_array = np.arange(0, 1+step, step) # should not be 1 I think
	y_array = [fp_triangle(x, n)*(x**2) for x in x_array]
	return np.sqrt(np.trapz(y=y_array, x=x_array)-fp_rev_triangle(n)**2)

def sp_stdev_triangle(n):
	x_array_1 = np.arange(0, .5+step, step)
	x_array_2 = np.arange(.5, 1+step, step)
	y_array_1 = [(x**2)*(1-2*x**2)*(x**(2*n-3)) for x in x_array_1]
	y_array_2 = [((1-x)**3)*((1-2*(1-x**2))**(n-2))*(x**2) for x in x_array_2]
	return n*(n-1)*(2**n)*np.trapz(y=y_array_1, x=x_array_1) + 8*n*(n-1)*np.trapz(y=y_array_2, x=x_array_2)

################# EXPONENTIAL (lambda=1)
def fp_exp(v, n):
	if v == 0:
		return 0
	x_array = np.arange(0, v+step, step)
	y_array = [Fn1_exp(x, n) for x in x_array]
	return v - np.trapz(y=y_array, x=x_array)/Fn1_exp(v, n)

def Fn1_exp(x, n):
	return ((1.0-np.exp(-1.0*x))**(n-1))

def sp_exp(v, n):
	return v

# Revenue
def fp_rev_exp(n):
	x_array = np.arange(0, 1+step, step) # should not be 1 I think
	y_array = [fp_exp(x, n)*x for x in x_array]
	return np.trapz(y=y_array, x=x_array)

def sp_rev_exp(n):
	x_array = np.arange(0, 1+step, step)
	y_array = [np.exp(-2*1.0*x)*(1-np.exp(-1.0*x))**(n-2)*x for x in x_array]
	return n*(n-1)*1*np.trapz(y=y_array, x=x_array)

# St. Dev
def fp_stdev_exp(n):
	x_array = np.arange(0, 1+step, step) # should not be 1 I think
	y_array = [fp_exp(x, n)*(x**2) for x in x_array]
	return np.sqrt(np.trapz(y=y_array, x=x_array) - fp_rev_exp(n)**2)

def sp_stdev_exp(n):
	x_array = np.arange(0, 1+step, step)
	y_array = [np.exp(-2*1.0*x)*(1-np.exp(-1.0*x))**(n-2)*(x**2) for x in x_array]
	return n*(n-1)*1*np.trapz(y=y_array, x=x_array)


################# MAIN
# 1.a)
## U(0,1)
plot_functions(fp_Uni_0_1, [2, 5, 10], [0, 1], "First price bid function, U(0,1)")
plot_functions(sp_Uni_0_1, [2, 5, 10], [0, 1], "Second price bid function, U(0,1)")

## U(5,10)
plot_functions(fp_Uni_5_10, [2, 5, 10], [5, 10], "First price bid function, U(5,10)")
plot_functions(sp_Uni_5_10, [2, 5, 10], [5, 10], "Second price bid function, U(5,10)")

## triangle
plot_functions(fp_triangle, [2, 5, 10], [0, 1], "First price bid function, Triangular")
plot_functions(sp_triange, [2, 5, 10], [0, 1], "Second price bid function, Triangular")

## exponential
plot_functions(fp_exp, [2, 5, 10], [0, 1], "First price bid function, Exp (lambda=1)")
plot_functions(sp_exp, [2, 5, 10], [0, 1], "Second price bid function, Exp (lambda=1)")

# 1.b)
d = {
	"U(0,1)": {"First Price": {"Revenue": fp_rev_Uni_0_1,
	                           "St. Dev": fp_stdev_Uni_0_1},
	           "Second Price": {"Revenue": sp_rev_Uni_0_1,
	                            "St. Dev":  sp_stdev_Uni_0_1}},
	"U(5,10)": {"First Price": {"Revenue": fp_rev_Uni_5_10,
	                            "St. Dev": fp_stdev_Uni_5_10},
	            "Second Price": {"Revenue": sp_rev_Uni_5_10,
	                             "St. Dev": sp_stdev_Uni_5_10}},
	"Triangular": {"First Price": {"Revenue": fp_rev_triangle,
	                               "St. Dev": fp_stdev_triangle},
	               "Second Price": {"Revenue": sp_rev_triangle,
	                                "St. Dev": sp_stdev_triangle}},
	"Exp (lambda=1)": {"First Price": {"Revenue": fp_rev_exp,
	                                   "St. Dev": fp_stdev_exp},
	                   "Second Price": {"Revenue": sp_rev_exp,
	                                    "St. Dev": sp_stdev_exp}}
}
d = OrderedDict(sorted(d.items(), key=lambda t: t[0]))

for dist in d:
	print "{}".format(dist)
	for auc_type in d[dist]:
		print "  {}".format(auc_type)
		for n in [2, 5, 10]:
			for f in d[dist][auc_type]:
				print "    n={:2d}: {:.4f} ({})".format(n, d[dist][auc_type][f](n), f)


# 1.c)

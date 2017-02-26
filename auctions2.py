import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import triang
from collections import OrderedDict

step = 0.001

################# PLOT
def plot_functions(func, list_n, rng, auc_type, dist):
	x = np.arange(rng[0], rng[1], step)
	for n in list_n:
		plt.plot(x, [func(v, n) for v in x], label="n={}".format(n))
	plt.legend(loc=2)
	plt.xlabel('v')
	plt.ylabel('b(v)')
	plt.title("{} bid function, {}".format(auc_type, dist))
	plt.show()

def plot_reserves(func, list_n, auc_form):
	for n in list_n:
		vals, reserves = func(n)
		plt.plot(vals, reserves, label="n={}".format(n))
	plt.legend(loc=2)
	plt.xlabel('Seller Values')
	plt.ylabel('Optimal Reserve Price')
	plt.title('Seller Values v Optimal Reserve Price for {}'.format(auc_form))
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
	return 5+5*fp_rev_Uni_0_1(n)

def sp_rev_Uni_5_10(n):
	x_array = np.arange(5, 10+step, step)
	y_array = [(1-(x-5)/5.)*(((x-5)/5.)**(n-2))*x for x in x_array]
	return n*(n-1)*(1/5.)*np.trapz(y=y_array, x=x_array)

# St. Dev
def fp_stdev_Uni_5_10(n): #TODO
	# x_array = np.arange(5, 5+(((n-1)*np.power(10, 1./n)+5)/n)+step, step)
	# y_array = [(((n*x-5)/(n-1))**(n-1))*(x**2) for x in x_array]
	# return np.sqrt(((n**2)/(n-1))*np.trapz(y=y_array, x=x_array) - fp_rev_Uni_5_10(n)**2)
	# #return np.sqrt((5**2)*(((n-1.)**2)/(n*(n+2)))-fp_rev_Uni_5_10(n)**2) #not sure
	return -1

def sp_stdev_Uni_5_10(n):
	x_array = np.arange(5, 10+step, step)
	y_array = [(1-(x-5)/5.)*(((x-5)/5.)**(n-2))*(x**2) for x in x_array]
	return np.sqrt(n*(n-1)*(1/5.)*np.trapz(y=y_array, x=x_array) - sp_rev_Uni_5_10(n)**2)

################# TRIANGLE [0, 1]
def fp_triangle(v, n):
	if v == 0:
		return 0.0
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
	return np.sqrt(n*(n-1)*(2**n)*np.trapz(y=y_array_1, x=x_array_1) + 8*n*(n-1)*np.trapz(y=y_array_2, x=x_array_2)-sp_rev_triangle(n)**2)

################# EXPONENTIAL (lambda=1)
def fp_exp(v, n):
	if v == 0:
		return 0.0
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
	return np.sqrt(n*(n-1)*1*np.trapz(y=y_array, x=x_array) - sp_rev_exp(n)**2)


################# RESERVE OPTIMIZATION
################# UNIFORM [0,1]
def run_u_0_1_auction(n):
	bidder_vals = [sp_Uni_0_1(np.random.uniform(0, 1), n) for cnt in range(n)]
	return sorted(bidder_vals)[-2]


def get_reserve_profit_u_0_1(n, seller_val, reserve, num_sims=500):
	profits = []
	for i in range(num_sims):
		win_bid = run_u_0_1_auction(n)
		if win_bid > reserve:
			profits.append(win_bid-seller_val)
		else:
			profits.append(0)
	return np.mean(profits)


def get_optimal_reserve_u_0_1(n, n_sims=500):
	optimal_reserves = []
	seller_vals = np.arange(0.0, 1.0, 0.1)
	reserves = np.arange(0.0, 1.1, 0.1)
	for seller_v in seller_vals:
		tmp = [get_reserve_profit_u_0_1(n, seller_v, reserve, num_sims=n_sims) for reserve in reserves]
		optimal_reserves.append(reserves[tmp.index(max(tmp))])
	return seller_vals, optimal_reserves

################# UNIFORM [5,10]
def run_u_5_10_auction(n):
	bidder_vals = [sp_Uni_5_10(np.random.uniform(5, 10), n) for cnt in range(n)]
	return sorted(bidder_vals)[-2]


def get_reserve_profit_u_5_10(n, seller_val, reserve, num_sims=500):
	profits = []
	for i in range(num_sims):
		win_bid = run_u_5_10_auction(n)
		if win_bid > reserve:
			profits.append(win_bid-seller_val)
		else:
			profits.append(0)
	return np.mean(profits)


def get_optimal_reserve_u_5_10(n, n_sims=500):
	optimal_reserves = []
	seller_vals = np.arange(5.0, 10.0, 0.1)
	reserves = np.arange(5.0, 10.1, 0.1)
	for seller_v in seller_vals:
		tmp = [get_reserve_profit_u_5_10(n, seller_v, reserve, num_sims=n_sims) for reserve in reserves]
		optimal_reserves.append(reserves[tmp.index(max(tmp))])
	return seller_vals, optimal_reserves
	

################# MAIN
# 1.a)
d = {
	"U(0,1)": {"First Price": fp_Uni_0_1,
			   "Second Price": sp_Uni_0_1,
			   "Range": [0, 1]},
	"U(5,10)": {"First Price": fp_Uni_5_10,
				"Second Price": sp_Uni_5_10,
				"Range": [5, 10]},
	"Triangular": {"First Price": fp_triangle,
				   "Second Price": sp_triange,
				   "Range": [0, 1]},
	"Exp (lambda=1)": {"First Price": fp_exp,
					   "Second Price": sp_exp,
					   "Range": [0, 1]},
}
d = OrderedDict(sorted(d.items(), key=lambda t: t[0]))

for dist in d:
	for auc_type in d[dist]:
		if auc_type != "Range":
			plot_functions(d[dist][auc_type], [2, 5, 10], d[dist]["Range"], auc_type, dist)


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
				print "	n={:2d}: {:.4f} ({})".format(n, d[dist][auc_type][f](n), f)


# 1.c)
d = {
	"U(0,1)": get_optimal_reserve_u_0_1,
	"U(5,10)": get_optimal_reserve_u_5_10#,
	# "Triangular": get_optimal_reserve__triangle,
	# "Exp (lambda=1)": get_optimal_reserve_exp
}
d = OrderedDict(sorted(d.items(), key=lambda t: t[0]))

for dist in d:
	plot_reserves(d[dist], [2, 5, 10], dist)
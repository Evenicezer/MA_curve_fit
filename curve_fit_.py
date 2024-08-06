import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from iprocessor import add_day_name_column, add_date_name_column, smooth_sundays_rolling_w7_l


# Sample parameters,
contacts = 2.0
transmission_prob = 0.3649
total_population = 84000000
reducing_transmission = 0.764
exposed_period = 5.2  #
asymptomatic_period = 7
infectious_period = 3.7
isolated_period = 11  # 11,23
prob_asymptomatic = 0.2
prob_quarant_inf = 0.05
test_asy = 0.171
dev_symp = 0.125
mortality_isolated = 0.002
mortality_infected = 0.01

# Sample initial conditions-------------------------------------------------------
total_population = 82000000  # Total number of individuals 2020 may
E0 = 18633.333333333332
A0 = 5064
I0 = 136.89
F0 = 1451
R0 = 2065
D0 = 47
S0 = total_population - E0 - A0 - I0 - F0 - R0 - D0
initial_conditions = [S0, E0, A0, I0, F0, R0, D0]

# Dataframe-------------------------------------------------
df = pd.read_csv(r'C:\Users\Evenezer kidane\PycharmProjects\MA\Ma\German_case_.csv')

#Convert 'Date' column to datetime format----------------------------------------------------------------------
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')

# Modification------------------------------------------------------------------------------
# Add the 'days' column
df = add_day_name_column(df)
df = add_date_name_column(df)
# second modification with w7_l-----------------------------------------------------------------------------
df_observed = smooth_sundays_rolling_w7_l(df)
# -----------------------------------------------------------------------------------------------------
# Taking 'days' time column from dataframe
t_fit_base = np.array(df_observed['days'])
tmax_base = len(t_fit_base)
#time
t_fit_0 = np.zeros((tmax_base,2))
t_fit_0[:,0] = t_fit_base
t_fit_0[:,1] = 0
#
t_fit_1 = np.zeros((tmax_base,2))
t_fit_1[:,0] = t_fit_base
t_fit_1[:,1] = 1
#
t_fit = np.r_[t_fit_0,t_fit_1]
len(t_fit)
#
tmax = len(t_fit)

def derivative_rhs(t, X, contacts, transmission_prob, total_population, reducing_transmission,
                   exposed_period, asymptomatic_period, infectious_period, isolated_period,
                   prob_asymptomatic, prob_quarant_inf, test_asy, dev_symp, mortality_isolated, mortality_infected):
    S, E, A, I, F, R, D = X
    derivS = - contacts * transmission_prob * S * (I + reducing_transmission * A) / total_population
    derivE = contacts * transmission_prob * S * (I + reducing_transmission * A) / total_population - E / exposed_period
    derivA = prob_asymptomatic * E / exposed_period - A / asymptomatic_period
    derivI = (
                         1 - prob_asymptomatic) * E / exposed_period + dev_symp * A / asymptomatic_period - I / infectious_period  # +
    derivF = prob_quarant_inf * I / infectious_period - F / isolated_period + test_asy * A / asymptomatic_period  # prob_isolated_asy*A/asymptomatic_period
    derivR = (1 - prob_quarant_inf - mortality_infected) * I / infectious_period + (
                1 - mortality_isolated) * F / isolated_period + (
                         1 - dev_symp - test_asy) * A / asymptomatic_period  # (1-prob_isolated_asy)*A / asymptomatic_period
    derivD = (mortality_infected) * I / infectious_period + mortality_isolated * F / isolated_period
    return np.array([derivS, derivE, derivA, derivI, derivF, derivR, derivD])



def seaifrd_model(t, contacts, initial_conditions, transmission_prob, total_population, reducing_transmission,
                  exposed_period, asymptomatic_period, infectious_period, isolated_period,
                  prob_asymptomatic, prob_quarant_inf, test_asy, dev_symp, mortality_isolated, mortality_infected):
    def derivative(t, initial_conditions):
        return derivative_rhs(t, initial_conditions, contacts, transmission_prob, total_population,
                              reducing_transmission,
                              exposed_period, asymptomatic_period, infectious_period,
                              isolated_period, prob_asymptomatic,
                              prob_quarant_inf, test_asy, dev_symp, mortality_isolated, mortality_infected)

    solution = solve_ivp(derivative, [0, tmax], initial_conditions, t_eval=t, method='RK45')
    print(solution)
    return solution  # .y.flatten()




def objective(t, contacts, initial_conditions, transmission_prob, total_population, reducing_transmission,
              exposed_period, asymptomatic_period, infectious_period, isolated_period,
              prob_asymptomatic, prob_quarant_inf, test_asy, dev_symp, mortality_isolated, mortality_infected):
    temp = seaifrd_model(t, contacts, initial_conditions, transmission_prob, total_population, reducing_transmission,
                         exposed_period, asymptomatic_period, infectious_period,
                         isolated_period, prob_asymptomatic,
                         prob_quarant_inf, test_asy, dev_symp, mortality_isolated, mortality_infected)
    # print(f' the name of the columns is: {type(temp)}')
    return temp  # [temp.y[5],temp.y[6]]


#----------------------------------------------------------------------------------------------------------------
# df_observed/ dataframe that have the cumulative as well as daily observed data(n_) for dead and recovered
# here we will combine the recovered and dead in one matrix with 2d
#['Date', 'Confirmed', 'Deaths', 'Recovered', 'n_confirmed', 'n_death','n_recovered', 'Infection_case', 'date_name', 'days', 'rolling_mean_r','rolling_mean_c', 'rolling_mean_d']
data_recovered = np.ones((df_observed['n_recovered'].size,2))
data_recovered[:,0] = df_observed['n_recovered']
#
data_dead = np.zeros((df_observed['n_death'].size,2))
data_dead[:,0] = df_observed['n_death']
recovered_dead = np.r_[data_recovered,data_dead]



#-----------------------------------------------------------------------------------------------------------------
def objective_function_recoverd_dead(X, contacts):
    t = X[:, 0]
    mode = X[:, 1]

    # Model parameters (could be passed as arguments or fixed here)
    transmission_prob = 0.3
    total_population = 82000000
    reducing_transmission = 0.55
    exposed_period = 5.2
    asymptomatic_period = 7
    infectious_period = 3.7
    isolated_period = 12
    prob_asymptomatic = 0.34
    prob_quarant_inf = 0.9303
    test_asy = 0.271
    dev_symp = 0.125
    mortality_isolated = 0.02
    mortality_infected = 0.1

    # Solve the model
    solution = seaifrd_model(t, contacts, initial_conditions, transmission_prob, total_population, reducing_transmission,
                             exposed_period, asymptomatic_period, infectious_period, isolated_period,
                             prob_asymptomatic, prob_quarant_inf, test_asy, dev_symp, mortality_isolated, mortality_infected)

    recovered = solution.y[5]
    dead = solution.y[6]

    return np.where(mode, recovered, dead)


t_end = df_observed['days'].iloc[-1]
# Create a sequence from 0 to t_end
#t_fit = np.arange(0, 2*tmax, 1)

params_r_d, _ = curve_fit(objective_function_recoverd_dead, t_fit, recovered_dead[:,0])



# assigning back
# List of names corresponding to each value in params
param_names_r_d = [ 'contacts']

param_dict_r_d = {}

for name, value in zip(param_names_r_d, params_r_d):
    param_dict_r_d[name] = value

print(param_dict_r_d)
print(f'length of the time t_fit_{len(t_fit)}, tmax{(tmax)}, time t {len(t_fit_base)},time t_fit_{len(t_fit)}, recovered_dead{len(recovered_dead)}')

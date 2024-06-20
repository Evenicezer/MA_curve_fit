#------------------------------- Effects of social distancing

# which is a list of time-dependent values for 𝛽 that work as follows:
'''When time t <= 22 days:

betta= betta [0]
Afterwards, when time t <= 35 days:

betta_a = betta [1]
After time t exceeds 35 days,

betta_b= betta [2]'''
#----------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy import integrate
from iprocessor import add_day_name_column,smooth_sundays_rolling_w7_l,add_date_name_column
from utility_intilization import get_value_by_date
#from numerical_simulation import param_dict_r as param_dict,t_fit

#---------------------------------------------
df = pd.read_csv(r'C:\Users\Evenezer kidane\PycharmProjects\MA\Ma\German_case_period_1st.csv')

# -------------------------------------------------------------------------------Convert 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')

# ------------------------------------------------------------------------------  Modification
# Add the 'days' column
df = add_day_name_column(df)
df = add_date_name_column(df)
print(df['Confirmed'])
# ----------------------------------------------------------------------------- second modification with w7_l
df_observed = smooth_sundays_rolling_w7_l(df)
# 2020.03.02 initialization
#exposed_value: 18633.333333333332, asymptomatic_value: 5064.0,
# infection_value: 136.89000000000001, recovered:2065, death:47, isolated:1451, susceptible:81972602.77666667
total_population = 82000000  # Total number of individuals
E0 = 18633.333333333332
A0 = 5064
I0 = 136.89
F0 = 1451
R0 = 2065
D0 = 47
S0 = total_population - E0 - A0 - I0 - F0 - R0 - D0 # Susceptible individuals
initial_conditions = [S0, E0, A0, I0, F0, R0, D0]
X = S0, E0, A0, I0, F0, R0, D0
#--------------------------------------------
def derivative_rhs(t, X, contacts, transmission_prob, total_population, reducing_transmission,
                   exposed_period, asymptomatic_period, infectious_period, isolated_period,
                   prob_asymptomatic, prob_quarant_inf, test_asy, dev_symp, mortality_isolated, mortality_infected):
    S0, E0, A0, I0, F0, R0, D0 = X
    derivS = - contacts * transmission_prob * S0 * (I0 + reducing_transmission * A0) / total_population
    derivE = contacts * transmission_prob * S0 * (I0 + reducing_transmission * A0) / total_population - E0 / exposed_period
    derivA = prob_asymptomatic * E0 / exposed_period - A0 / asymptomatic_period
    derivI = (1 - prob_asymptomatic) * E0 / exposed_period + dev_symp * A0 / asymptomatic_period - I0 / infectious_period  # +
    derivF = prob_quarant_inf * I0 / infectious_period - F0 / isolated_period + test_asy * A0 / asymptomatic_period  # prob_isolated_asy*A/asymptomatic_period
    derivR = (1 - prob_quarant_inf - mortality_infected) * I0 / infectious_period + (1 - mortality_isolated) * F0 / isolated_period + (
                         1 - dev_symp - test_asy) * A0 / asymptomatic_period  # (1-prob_isolated_asy)*A / asymptomatic_period
    derivD = (mortality_infected) * I0 / infectious_period + mortality_isolated * F0 / isolated_period
    return np.array([derivS, derivE, derivA, derivI, derivF, derivR, derivD])



'''contacts = param_dict['contacts']
transmission_prob = param_dict['transmission_prob']
reducing_transmission = param_dict['reducing_transmission']
exposed_period = param_dict['exposed_period']
asymptomatic_period = param_dict['asymptomatic_period']
infectious_period = param_dict['infectious_period']
isolated_period = param_dict['isolated_period']
prob_asymptomatic = param_dict['prob_asymptomatic']
prob_quarant_inf = param_dict['prob_quarant_inf']
test_asy = param_dict['test_asy']
dev_symp = param_dict['dev_symp']
mortality_isolated = param_dict['mortality_isolated']
mortality_infected = param_dict['mortality_infected']'''
# ------------------------------------------------------replacing values
contacts = 3.9 # mean value, from the covimod
transmission_prob = 0.31
total_population = 82000000
reducing_transmission = 0.859
exposed_period = 5.2  # this is the incubation period
asymptomatic_period = 6  #[3, 14]
infectious_period = 7.05 # [5.6, 8.5]
isolated_period = 7 #21
prob_asymptomatic = 0.25
prob_quarant_inf = 0.81 #0.5
test_asy = 0.2 # [0.171, 0.2]
dev_symp = 0.19 # [0.1, 0.85]
mortality_isolated = 0.02
mortality_infected = 0.056

#tmax = t_fit  # maximum simulation day
tmax = 24#269
fslarge = 20
fssmall = 16


t_fit = df_observed['days'].max()
# Plot simulation results
t = np.linspace(0, t_fit, (t_fit + 1) * 10)
fig, ax = plt.subplots(figsize=[12, 7])

# Overlay the graph of observed confirmed cases
ax.plot(df_observed['days'], df_observed['n_confirmed'], label='Confirmed Cases (Observed)', linestyle='-', color='blue')

# Plot your simulated data
solution_seaifrd = integrate.solve_ivp(
    lambda t, X: derivative_rhs(t, X, contacts, transmission_prob, total_population, reducing_transmission,
                                 exposed_period, asymptomatic_period, infectious_period, isolated_period,
                                 prob_asymptomatic, prob_quarant_inf, test_asy, dev_symp, mortality_isolated,
                                 mortality_infected),
    [0, tmax], initial_conditions, t_eval=t, method='RK45')
ax.plot(solution_seaifrd.t, solution_seaifrd.y[3], label='Confirmed Cases (Simulated)', linestyle='--', marker='<',
        color='red', markersize=4)

# Set labels and title
ax.set_xlabel('Days', fontsize=fslarge)
ax.set_ylabel('Number of individuals', fontsize=fslarge)
ax.set_title('Simulation of an ODE-SEAIFRD model', fontsize=fslarge)

# Set legend
ax.legend(fontsize=fssmall, loc='center left', bbox_to_anchor=(1, 0.5))

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()

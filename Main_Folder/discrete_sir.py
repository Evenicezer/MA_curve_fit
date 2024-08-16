import numpy as np
import matplotlib.pyplot as plt
from gen_based_deterministic import dynamics_deterministic_generations


def discrete_sir_generations(population, contacts, transmission_prob, gen=0, max_gen=10):
    """! Computes the number of infected individuals up to a given number of
    generations based on a deterministic discrete SIR model using an iterative
    function.

    @param population Array for number of susceptible, infected and recovered individuals
        at first generation.
    @param contacts Number of contacts per generation.
    @param transmission_prob Transmission probability per contact.
    @param gen Initial generation number.
    @param max_gen Maximum number of generations to be computed.
    @return Number of infected individuals up to a given number of generations.
    """
    if gen < max_gen:
        N = sum(population)
        result = np.zeros((3, max_gen - gen + 1))
        result[:,0] = population
        for i in range(gen,max_gen):
            Si = result[0, i]
            Ii = result[1, i]
            Ri = result[2, i]
            result[0, i+1] = Si - contacts * transmission_prob * Ii / N * Si
            result[1, i+1] = contacts * transmission_prob * Ii / N * Si
            result[2, i+1] = Ri + Ii

        return result
    else:
        return None

def discrete_sir_days(population, contacts, transmission_prob, time_infected, days=10):
    """! Computes the number of infected individuals up to a given number of
    generations based on a deterministic discrete SIR model using an iterative
    function.

    @param population Array for number of susceptible, infected and recovered individuals
        at first generation.
    @param contacts Number of contacts per generation.
    @param transmission_prob Transmission probability per contact.
    @param days Maximum number of days to be computed.
    @return Number of infected individuals up to a given number of generations.
    """
    if days > 0:
        N = sum(population)
        result = np.zeros((3, days + 1))
        result[:,0] = population
        for i in range(days):
            Si = result[0, i]
            Ii = result[1, i]
            Ri = result[2, i]
            result[0, i+1] = Si - contacts * transmission_prob * Ii / N * Si
            result[1, i+1] = contacts * transmission_prob * Ii / N * Si - 1 / time_infected * Ii
            result[2, i+1] = Ri + 1 / time_infected * Ii

        return result
    else:
        return None        


if __name__ == "__main__":
    infected = 1
    contacts = 12
    transmission_prob = 1/6
    max_gen = 15
    total_population = 10000
    recovered = 0

    fslarge = 20
    fssmall = 16    

    exercise = 3

    if exercise == 3:

        population_sir = discrete_sir_generations(np.array([total_population - infected - recovered,
                                    infected, recovered]), contacts,
                                    transmission_prob, max_gen=max_gen)

        infected_simple_genbased = dynamics_deterministic_generations(infected, contacts,
                                    transmission_prob, max_gen=max_gen)


        x_generations = np.linspace(1, max_gen, max_gen)
        plt.figure(figsize=(9,7))
        plt.plot(x_generations, infected_simple_genbased, color="red", label='New infections (Ex. 1)', linewidth=2)
        plt.plot(x_generations, population_sir[1,1:], color="green", label='New infections (Difference equations-SIR)', linewidth=2)
        plt.plot(x_generations, population_sir[2,1:], color="black", label='Recovered (Difference equations-SIR)', linewidth=2)
        plt.plot(x_generations, [total_population for i in range(max_gen)],
                color="blue", label='Total population size', linewidth=2)
        plt.xlabel('Generations', fontsize=fslarge)
        plt.ylabel('Individuals', fontsize=fslarge)
        plt.title('New infections and the effect of non-susceptibles', fontsize=fslarge)
        plt.legend(fontsize=fssmall)
        plt.show()
    
    elif exercise == 4:

        days = 10
        time_infected = [0.5, 1, 1.5, 2]

        results = []
        for tI in time_infected:
            result_new = discrete_sir_days(np.array([total_population - infected - recovered,
                                    infected, recovered]), contacts,
                                    transmission_prob, tI, days=days)
            results.append(result_new)

        plt.figure(figsize=(9,7))
        x_days = np.linspace(0, days, days)
        for i in range(len(time_infected)):
            plt.plot(x_days, results[i][1,1:], label='New infections (T_I=' + str(time_infected[i]) +")", linewidth=2)
        plt.xlabel('Days', fontsize=fslarge)
        plt.ylabel('Individuals', fontsize=fslarge)
        plt.title('New infections for different infectious periods', fontsize=fslarge)
        plt.legend(fontsize=fssmall)
        plt.show()
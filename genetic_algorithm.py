from mutation_funcs import swap_weights, replace_weights
import numpy as np
from neural_network import *
class GA():
    """
    Class encapsulating everything about genetic algorithm.
    It has population of Neural Networks that it tries to optimize
    through fitness, selection, crossover and mutation.
    It takes in population length, the preprocessed training data and
    the sizes of the neural networks in the population.

    len_population : int
    train_x : np.ndarray
    train_y : np.ndarray
    input_layer_len : int
    hidden_layer_len : int
    output_layer_len : int

    """
    def __init__(self, len_population,
                 train_x,
                 train_y,
                 input_layer_len,
                 hidden_layer_len,
                 output_layer_len):
        
        self.len_population = len_population
        self.input_layer_len = input_layer_len
        self.hidden_layer_len = hidden_layer_len
        self.output_layer_len = output_layer_len
        
        # Already preprocessed
        self.train_x = train_x
        self.train_y = train_y
        
        self.population_outputs = np.zeros((self.len_population, 
                                            self.train_x.shape[0], 
                                            self.output_layer_len))
        
        self.population_fitness_values = np.zeros((self.len_population, 1))
        
        self.parent_selection_rate = 0.5
        self.crossover_rate = 0.5
        
        self.population = self.init_population()
        self.new_gen_population = None
        
        # Stagnation settings:
        self.len_stagnation = 500
        self.stagnation = False
        self.normal_mutation_rate = (0.85, 1.25)
        self.stagnation_mutation_rate = (-1., 1.80)
        self.current_mutation_rate = self.normal_mutation_rate
        
        # Fitness sharing:
        self.min_shared_fitness_dist = 10
        
        # Informal things:
        self.fitness_over_time = []
        self.accuracy_over_time = []

    def test_accuracy(self):
        """
        Function to test accuracy of the population on dataset.
        Counts how many right answers each individual got and
        makes percentage rate out of it.
        """
        # Test Population:
        population_correct_ratio = []
        for i in range(len(self.population)):
            output = self.population[i].forward_propagate(self.train_x)
            
            answered_right = 0
            for k in range(len(output)):
                current_y_index = self.train_y[k].argmax()
                yhat = output[k][current_y_index]
                if yhat == max(output[k]):
                    answered_right += 1
            population_correct_ratio.append( (answered_right / output.shape[1] ) * 100)

        return population_correct_ratio

    def init_population(self):
        """
        Initializes population of neural network
        with the size parameters given.
        """
        pop = [None] * self.len_population
        for i in range(self.len_population):
            pop[i] = NN(self.input_layer_len, 
                                    self.hidden_layer_len, 
                                    self.output_layer_len)
        return pop
    
    def forward_propagate_population(self):
        """
        For each individual of population forward_propagation is called
        and then the output is saved to further work with.
        """
        for i in range(self.len_population):
            self.population_outputs[i] = self.population[i].forward_propagate(self.train_x) # Returns (26, 26)
            
    def calculate_shared_fitness(self, individual_index):
        """
        Although I don't use this function in the end, I still thought
        it is okay to leave here.
        Type of niching. If two individuals of population are really
        close to each other in terms of same weights and outputs, it
        punishes them by making their fitness worse.
        """
        denominator = 1
        share_param = 100
        
        curr_ih = self.population[individual_index].weights_ih
        curr_ho = self.population[individual_index].weights_ho
        
        for i in range(self.len_population):
            
            dist1 = np.sum((curr_ih - self.population[i].weights_ih)**2)
            dist2 = np.sum((curr_ho - self.population[i].weights_ho)**2)
            
            if dist1 < self.min_shared_fitness_dist and \
            dist2 < self.min_shared_fitness_dist:
                denominator += (1-( (dist1+dist2)/share_param))
        return denominator
        

    def calculate_fitness(self):
        """
        Calculates fitness of all the individuals in population.
        Given the output of the feed forward function, it calculates the
        fitness of each of them.
        For each image, find the answer in train_y and takes it's index.
        Use it's index to find the answer in the output.
        The closer the value is to 0 the better.
        """
        #self.population_outputs = (10, 26, 26)
        for i in range(self.len_population): # 10
            individual_fitness = 0
            for k in range(len(self.population_outputs[i])): # 26 (pismen)
                current_y_index = self.train_y[k].argmax()
                yhat = self.population_outputs[i][k][current_y_index]
                individual_fitness += (1 - yhat)
            self.population_fitness_values[i] = individual_fitness

            # In case I wanted to use shared fitness, This would be uncommented
            #self.population_fitness_values[i] += self.calculate_shared_fitness(i)
    def select(self):
        """
        Sorts the population by it's fitness and then selects the best 50%
        of the population to be parents.
        """
        index_fitness = [ (i, a[0]) for i, a in enumerate(self.population_fitness_values)]
        sorted_index_fitness = sorted(index_fitness, key=lambda tup: tup[1])
        
        selection_boundary = int(self.len_population * self.parent_selection_rate)
        parents_indexes = [i for (i, a) in sorted_index_fitness[:selection_boundary]]
        
        self.new_gen_population = np.array([self.population[i] for i in parents_indexes])
    
    def crossover(self):
        """
        Randomly creates pairs out of the parents
        then takes the half of the weights of the first parent
        and concatenate it with second half of weights of second parent.
        This creates the offsprings that are added to the new generation
        population.
        """
        import random
        a = [i for i in range(len(self.new_gen_population))]
        last_one = random.choice(a)
        b = []
        for _ in range(len(a)//2):
            first_half = random.choice(a)
            a.remove(first_half)
            second_half = random.choice(a)
            a.remove(second_half)
            b.append((first_half, second_half))
        b.append((a[0], last_one))
        
        pairs = []
        for p in b:
            pairs.append( (self.new_gen_population[p[0]], self.new_gen_population[p[1]]) )
        
        children = []
        for pair in pairs:
            # First child
            new_child = NN()
            new_weights_ih = np.concatenate((
                pair[0].weights_ih[:self.input_layer_len//2],
                pair[1].weights_ih[self.input_layer_len//2:]
            ))
            new_weights_ho = np.concatenate((
                pair[0].weights_ho[:self.hidden_layer_len//2],
                pair[1].weights_ho[self.hidden_layer_len//2:]
            ))
            new_child.from_weights(new_weights_ih, new_weights_ho)
            children.append(new_child)
            
            # Second child
            new_child = NN()
            new_weights_ih = np.concatenate((
                pair[1].weights_ih[:self.input_layer_len//2],
                pair[0].weights_ih[self.input_layer_len//2:]
            ))
            new_weights_ho = np.concatenate((
                pair[1].weights_ho[:self.hidden_layer_len//2],
                pair[0].weights_ho[self.hidden_layer_len//2:]
            ))
            new_child.from_weights(new_weights_ih, new_weights_ho)
            children.append(new_child)
        
        if self.new_gen_population.shape[0] + len(children) != self.len_population:
            children = children[:self.new_gen_population.shape[0]]
        self.new_gen_population = np.concatenate((self.new_gen_population, children))
    
    def stagnation_mutation(self, individual):
        """
        If there is a point where population is stagnating, this function
        is called.
        It mutates the weights more agressively and also more of the
        weights are mutated.
        Randomly chooses atleast 3 columns of the weights to be mutated
        and then randomly chooses of the mutations and mutates.
        """
        mutations = [
            lambda dim : dim * np.random.uniform(self.current_mutation_rate[0], 
                                                 self.current_mutation_rate[1],
                                                 dim.shape),
            lambda a : a * np.random.choice([-1, 1], a.shape),
            swap_weights,
            replace_weights,
                    ]
        if self.stagnation:
            dims_to_be_mutated_len = np.random.randint(3,individual.weights_ih.T.shape[0])
            dims_to_mutate = np.random.choice(individual.weights_ih.T.shape[0],
                                             dims_to_be_mutated_len)
            for dim in dims_to_mutate:
                individual.weights_ih = individual.weights_ih.T
                mutations_index = np.random.randint(0, 2)
                individual.weights_ih[dim] = mutations[mutations_index]\
                                            (individual.weights_ih[dim])

                individual.weights_ih = individual.weights_ih.T

            dims_to_be_mutated_len = np.random.randint(3,individual.weights_ho.shape[0])
            dims_to_mutate = np.random.choice(individual.weights_ho.shape[0],
                                             dims_to_be_mutated_len)
    
            for dim in dims_to_mutate:
                individual.weights_ho = individual.weights_ho.T
                mutations_index = np.random.randint(0,2)
                individual.weights_ho[dim] = mutations[mutations_index]\
                                            (individual.weights_ho[dim])
                    
                individual.weights_ho = individual.weights_ho.T
    def mutate(self):
        """
        All of the population except one individual is mutated.
        It chooses between the first two mutations in the mutations
        array and mutates one randomly chosen column.

        List of possible mutations:
            - Multiply some weights by random val close to 1 (eg 0.8 - 1.25).
            - Replace some weights completely.
            - Negate some weights.
            - swap some weights with other weights.
        Plan:
            1) 50% chance for each individual to mutate.
            2) 50% chance for weights_ih and 50% chance for weights_ho. <---
            3) Choose which dimension of the 2-dim matrix to mutate.
        """
        
        mutations = [
            lambda dim : dim * np.random.uniform(self.current_mutation_rate[0], 
                                                 self.current_mutation_rate[1],
                                                 dim.shape),
            lambda a : a * np.random.choice([-1, 1], a.shape),
            swap_weights,
            replace_weights,
                    ]
        
        for individual in self.new_gen_population[2:]:
            self.stagnation_mutation(individual)
            if not np.random.randint(0,2) == 1:
                continue
            
            # weights_ih mutating:
            if np.random.randint(0,2) == 1:
                individual.weights_ih = individual.weights_ih.T
                dim_to_mutate = np.random.randint(0,individual.weights_ih.shape[0])
                mutations_index = np.random.randint(0, 2)

                individual.weights_ih[dim_to_mutate] = mutations[mutations_index]\
                                            (individual.weights_ih[dim_to_mutate])

                individual.weights_ih = individual.weights_ih.T
            
            # weights_ho mutating:
            if np.random.randint(0,2) == 1:
                individual.weights_ho = individual.weights_ho.T
                dim_to_mutate = np.random.randint(0,individual.weights_ho.shape[0])
                mutations_index = np.random.randint(0, 2)

                individual.weights_ho[dim_to_mutate] = mutations[mutations_index]\
                                            (individual.weights_ho[dim_to_mutate])
                individual.weights_ho = individual.weights_ho.T
    
    def is_stagnating(self, gen):
        """
        If stagnation happenned in the last x generations -->
        make the mutation numbers higher, more agressive and hits more
        columns.
        """
        if gen <= self.len_stagnation:
            return
        current_fitness = self.fitness_over_time[-1]
        last_x_generations = self.fitness_over_time[gen-self.len_stagnation:]
        
        if np.all(last_x_generations == current_fitness):
            self.stagnation = True
            self.current_mutation_rate = self.stagnation_mutation_rate
        else:
            self.stagnation = False
            self.current_mutation_rate = self.normal_mutation_rate
    
    def update_population(self):
        """
        Updates the population so the new generation is now
        the population.
        """
        self.population = self.new_gen_population
    
    def evolve(self, num_of_generations):
        """
        All of the methods above connected and called for an x amount
        of times.
        For each generation it saves the best fitness value and the
        best accuracy value.
        """
        for _ in range(num_of_generations):
            self.forward_propagate_population()
            self.calculate_fitness()
            self.select()
            self.crossover()
            self.is_stagnating(_)
            self.mutate()
            self.update_population()

            self.fitness_over_time.append(self.population_fitness_values.min())
            self.accuracy_over_time.append(max(self.test_accuracy()))
            if _ % 1000 == 0:
                print(f'Generation {_}: Done')
            
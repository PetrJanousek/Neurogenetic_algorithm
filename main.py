import load_dataset
import numpy as np

from genetic_algorithm import *

x, y = load_dataset.load_dataset('dataset')

res = load_dataset.extract_frequent_pixels(x)

train_x, train_y = load_dataset.normalize_data(res, y)

# #TRAINING ALGO START
ga = GA(10, train_x, train_y, 
        input_layer_len=train_x.shape[1], 
        hidden_layer_len=150, 
        output_layer_len=26)

num_generations = 5000
ga.evolve(num_generations)

import matplotlib.pyplot as plt

plt.plot(list(range(num_generations)), ga.fitness_over_time, label = 'Fitness')
plt.plot(list(range(num_generations)), ga.accuracy_over_time, label = 'Accuracy')
plt.xlabel('number of generations')
plt.ylabel('fitness score')
plt.legend(['fitness', 'accuracy'])
plt.show()

population_accuracy = ga.test_accuracy()


for i in population_accuracy:
    print(i)
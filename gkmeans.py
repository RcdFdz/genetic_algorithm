import pandas as pd
import numpy as np
import random
from sklearn.cluster import KMeans
from sklearn import preprocessing


class GKmeans:

    def __init__(self, data, number_chromosomes):
        self.data = data
        self.data_scaled = None

        self.num_nc = number_chromosomes #to be set random

        self.d = np.ndim(self.data)
        self.ch_dim = self.d * 2 + 2

        self.preserved_chromosome = []
        self.mutated_chromosome = []
        self.crossover_chromosome = []

    def normalize(self):
        min_max_scaler = preprocessing.MinMaxScaler()
        data_scaled = pd.DataFrame(min_max_scaler.fit_transform(self.data))
        self.data_scaled = data_scaled

    def get_crossover_chromosome(self):
        # Crossover Chromosomes
        ch_crossover_str_list = []
        for i in range(Nclusters):
            ch1 = self.preserved_chromosome[random.choices(range(0,self.num_nc))[0]]
            ch2 = self.preserved_chromosome[random.choices(range(0,self.num_nc))[0]]
            ms = ch1 + ch2

            for idx in range(0, round(len(ms)/2)):
                idx = random.choices(range(0,len(ms)))[0]
                ms= ms[:idx] + ms[idx+1:]

            ms_neg = ''.join([str(int(not int(i))) for i in ms])

            #ch = (ch1 & Ms) | ( ch2 ∧ Ms )
            ch_ = int(ch1,2)&int(ms,2)
            ch__ = int(ch2,2)&int(ms_neg,2)
            ch_crossover_str_list.append(format(ch_ | ch__, '0'+str(self.ch_dim**2)+'b'))
        self.crossover_chromosome = ch_crossover_str_list
        print(ch_crossover_str_list)

    def get_hyperparameters(self, ch_str_list):
        # integer to binary string format(int, '02b')
        # binary string to integer int(binary_string, 2)
        v = [int(x, 2) for x in list(map(''.join, zip(*[iter(ch_str_list[0])]*self.ch_dim)))]

        # tri-lavel
        wi = [v[i]/sum(v[0:d]) for i in range(0,self.d)]
        ri = [v[i]*v[-1]/sum(v[2:-1]) for i in range(self.d,2*self.d)]
        return wi, ri

    # The following calcul return a NxM Matrix for each point in datase
    # where:
    # N is the number of centers
    # M number of dimensions defining the point

    @staticmethod
    def get_distance(data, centers, wi, ri):
        return [np.multiply(wi, np.power(abs(np.subtract(centers, v)),ri)) for i,v in enumerate(np.matrix(data))]

    #Nclusters = 3
    #kmeans = KMeans(n_clusters=3, random_state=0).fit(self.data)
    #centers = kmeans.cluster_centers_

    #kmeans.cluster_centers_.round(4)

dataset = np.array([[1,1], [1,2], [2,1], [2,2], [1,4], [1,5], [1,6], [5,6], [5,7], [6,6], [6,7]])

a = GKmeans(dataset,3)
a.get_preserved_chromosome()
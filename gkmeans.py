import pandas as pd
import numpy as np
import random
from sklearn import preprocessing
from chromosome import Chromosome as ch


class GKmeans:

    def __init__(self, data, number_chromosomes):
        self.data = data
        self.data_scaled = None

        self.num_nc = number_chromosomes #to be set random

        self.d = np.ndim(self.data)

    def normalize(self):
        min_max_scaler = preprocessing.MinMaxScaler()
        data_scaled = pd.DataFrame(min_max_scaler.fit_transform(self.data))
        self.data_scaled = data_scaled

    def get_hyperparameters(self, ch_str_array):

        v = [int(x, 2) for x in ch_str_array]

        # tri-level
        wi = [v[i]/sum(v[0:self.d]) for i in range(0, self.d)]
        ri = [(v[i]*v[-1]/sum(v[2:-1])) + v[-1] for i in range(self.d, 2*self.d)]
        return wi, ri

    def run(self):
        self.normalize()

        chromosomes = []
        for nc in range(0, 3):
            c = ch(self.d)
            c.generate_chromosomes()
            chromosomes.append(c)

        for chrom in chromosomes:
            self.get_hyperparameters(chrom.chromosome_array)

    @staticmethod
    def get_distance(data, centers, wi, ri):
        return [np.multiply(wi, np.power(abs(np.subtract(centers, v)), ri)) for i, v in enumerate(np.matrix(data))]


dataset = np.array([[1,1], [1,2], [2,1], [2,2], [1,4], [1,5], [1,6], [5,6], [5,7], [6,6], [6,7]])

a = GKmeans(dataset, 3)
a.run()
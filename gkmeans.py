import numpy as np
import random
from operator import itemgetter
from sklearn import preprocessing
from chromosome import Chromosome as ch
from generic_kmeans import GenericKmeans as gk
from plotting import Plotting as pt


class GKmeans:

    def __init__(self, data, number_chromosomes):
        self.data = data
        self.data_scaled = None

        self.num_nc = number_chromosomes #to be set random

        self.d = np.ndim(self.data)

    def normalize(self):
        min_max_scaler = preprocessing.MinMaxScaler()
        data_scaled = min_max_scaler.fit_transform(self.data)
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
        for nc in range(0, self.num_nc):
            c = ch(self.d)
            c.generate_chromosomes_sample()
            chromosomes.append(c)

        for chrom in chromosomes:
            chrom.crossover_chromosome(random.choice([e for i, e in enumerate(chromosomes) if e != chrom]).chromosome_array)

        chromosomes_candidates = []
        for chrom in chromosomes:
            chromosomes_candidates.append(chrom.chromosome_array)
            chromosomes_candidates.append(chrom.mutated_chromosome_array)
            chromosomes_candidates.append(chrom.crossed_chromosome_array)

        results = []
        for chrom_str_array in chromosomes_candidates:
            w, r = self.get_hyperparameters(chrom_str_array)
            it = gk(self.data_scaled, 3, w, r)
            cent = it.new_centroids()
            score = it.get_chromosome_euclidian_score()

            results.append([chrom_str_array, cent, score])

        sorted_candidates = sorted(results, key=itemgetter(2))

        centers = [[5.55111512e-17, 6.66666667e-01], [9.00000000e-01, 9.16666667e-01], [1.00000000e-01, 8.33333333e-02]]
        labels = [2, 2, 2, 2, 0, 0, 0, 1, 1, 1, 1]
        pt(self.data_scaled,  centers, labels, sorted_candidates)


dataset = np.array([[1.0, 1.0], [1.0, 2.0], [2.0, 1.0], [2.0, 2.0], [1.0, 4.0], [1.0, 5.0], [1.0, 6.0], [5.0, 6.0], [5.0, 7.0], [6.0, 6.0], [6.0, 7.0]])

a = GKmeans(dataset, 3)
a.run()

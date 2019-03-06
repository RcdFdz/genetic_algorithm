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
        self.data_scaled = self.normalize()

        self.num_nc = number_chromosomes #to be set random

        self.d = np.ndim(self.data)

    def normalize(self):
        min_max_scaler = preprocessing.MinMaxScaler()
        data_scaled = min_max_scaler.fit_transform(self.data)
        return data_scaled

    def get_hyperparameters(self, ch_str_array):

        v = [int(x, 2) for x in ch_str_array]
        # tri-level
        wi = [v[i]/sum(v[0:self.d]) for i in range(0, self.d)]
        ri = [(v[i]*v[-1]/sum(v[2:-1])) + v[-1] for i in range(self.d, 2*self.d)]
        return wi, ri

    def initialize_chromosomes(self):
        chromosomes = []
        for nc in range(0, self.num_nc):
            c = ch(self.d)
            c.generate_chromosomes_sample()
            chromosomes.append(c)
        return chromosomes

    def generate_descendence(self, chromosomes, num_candidates):
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
        return np.array(sorted_candidates[:num_candidates])

    def plotting(self, sorted_candidates, single_cluster=False):
        pt(self.data_scaled,  sorted_candidates, single_cluster)

    def run(self, max_iter=100):

        candidates = self.generate_descendence(self.initialize_chromosomes(), self.num_nc)
        self.plotting(candidates)

        best_candidate = candidates[0]
        score = min(candidates[:, -1])
        prev_score = np.Inf

        for iteration in range(0, max_iter):
            if prev_score > score:
                candidates = self.generate_descendence(self.initialize_chromosomes(), self.num_nc)
                self.plotting(candidates)
                prev_score = score
                score = min(candidates[:, -1])
                if prev_score > score:
                    best_candidate = candidates[0]
            else:
                print('Convergence at iteration {}.\nCandidate Chromosome: {}.\nCenters at: {}.\nScore: {}'.format(
                    iteration, best_candidate[0], best_candidate[1], best_candidate[2]
                ))
                self.plotting(best_candidate, single_cluster=True)
                break

        w, r = self.get_hyperparameters(best_candidate[0])
        it_best = gk(self.data_scaled, 3, w, r)
        points_closet_centroid = it_best.get_points_closet_centroid(best_candidate[1])

        import matplotlib.pyplot as plt
        plt.title('Score: {}'.format(best_candidate[-1]))
        plt.scatter(points_closet_centroid[:, 0], points_closet_centroid[:, 1], c=points_closet_centroid[:, -1])

        _, idx = np.unique(points_closet_centroid[:, -1], return_index=True)
        plt.scatter(np.array(best_candidate[1])[:, 0], np.array(best_candidate[1])[:, 1],
                    c=np.unique(points_closet_centroid[np.sort(idx), -1]), s=200, alpha=0.5)

        #for el in clusters[1]:
        #    plt.scatter(el[0], el[1], marker='X', s=200)

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()

dataset = np.vstack([10*np.random.randn(100, 2)+10, 10*np.random.randn(100, 2)+5, 10*np.random.randn(100, 2)])

a = GKmeans(dataset, 3)
a.run()

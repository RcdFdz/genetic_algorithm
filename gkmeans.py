import numpy as np
import random
from operator import itemgetter
from sklearn import preprocessing
from chromosome import Chromosome as ch
from generic_kmeans import GenericKmeans as gk


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
            results.append([chrom_str_array, it.new_centroids(), it.get_chromosome_score()])

        sorted_candidates = sorted(results, key=itemgetter(2))

        for candidate in sorted_candidates:
            print("plt.title('Score {}')".format(candidate[-1]))
            print("""
plt.scatter(data_scaled[0], data_scaled[1], c=kmeans.labels_)
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c=np.unique(kmeans.labels_), s=200, alpha=0.5)
""")
            for x_cord, y_cord in candidate[1]:
                print("""plt.scatter({}, {}, marker='X', s=200)""".format(x_cord, y_cord))
            print("""
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
            """)
dataset = np.array([[1.0, 1.0], [1.0, 2.0], [2.0, 1.0], [2.0, 2.0], [1.0, 4.0], [1.0, 5.0], [1.0, 6.0], [5.0, 6.0], [5.0, 7.0], [6.0, 6.0], [6.0, 7.0]])

a = GKmeans(dataset, 3)
a.run()

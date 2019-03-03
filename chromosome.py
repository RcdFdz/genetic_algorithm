import random


class Chromosome:

    def __init__(self, data_dimension):
        self.s_length = data_dimension * 2 + 2

        self.l_length = []

        self.chromosome_array = []
        self.chromosome_str = ''
        self.chromosome_int = 0

        self.mutated_chromosome_array = []
        self.mutated_chromosome_str = ''
        self.mutated_chromosome_int = 0

        self.crossed_chromosome_array = []
        self.crossed_chromosome_str = ''
        self.crossed_chromosome_int = 0

    def generate_chromosome(self, l_min_length, l_max_length):
        for l in range(0, self.s_length):
            self.l_length.append(random.choices(range(l_min_length, l_max_length))[0])
            ch = random.getrandbits(self.l_length[-1])
            self.chromosome_array.append(format(ch, '0' + str(self.l_length[-1]) + 'b'))
        self.chromosome_str = ''.join(self.chromosome_array)
        self.chromosome_int = int(self.chromosome_str, 2)
        return self.chromosome_int

    def mutate_chromosome(self):
        for s in self.chromosome_array:
            idx = random.choices(range(0, len(s)))[0]
            seg_mutated = s[:idx] + str(int(not int(s[idx]))) + s[idx+1:]
            self.mutated_chromosome_array.append(seg_mutated)
        self.mutated_chromosome_str = ''.join(self.mutated_chromosome_array)
        self.mutated_chromosome_int = int(self.mutated_chromosome_str, 2)
        return self.mutated_chromosome_int

    def crossover_chromosome(self, chromosome_array):
        ch1 = chromosome_array
        ch2 = self.chromosome_array

        param_sel = random.choice(range(0, len(ch1)))

        ch = ''.join(ch2[:param_sel]) + ch1[param_sel] + ''.join(ch2[param_sel+1:])

        self.crossed_chromosome_int = int(ch, 2)
        self.crossed_chromosome_str = ch
        self.crossed_chromosome_array = [y for x in [ch2[:param_sel], [ch1[param_sel]], ch2[param_sel+1:]] for y in x]

    def generate_chromosomes_sample(self):
        choices = random.choices(range(3, 10), k=10)
        self.generate_chromosome(min(choices), max(choices))
        self.mutate_chromosome()

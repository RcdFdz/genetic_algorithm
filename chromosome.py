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

    def crossover_chromosome(self, chromosome):
        ch1 = chromosome[:round(len(chromosome) / 2)], chromosome[round(len(chromosome) / 2):]
        ch2 = self.chromosome_str

        len_ch1 = len(ch1[0])

        ms = ch1[0] + self.chromosome_str[len_ch1:]

        for idx in range(0, round(len(ms)/2)):
            idx = random.choices(range(0, len(ms)))[0]
            ms = ms[:idx] + ms[idx+1:]

        ms_neg = ''.join([str(int(not int(i))) for i in ms])

        # ch = (ch1 & Ms) | ( ch2 ∧ Ms )
        ch_ = int(''.join(ch1), 2) & int(ms, 2)
        ch__ = int(''.join(ch2), 2) & int(ms_neg, 2)
        ch___ = ch_ | ch__
        self.crossed_chromosome_str = format(ch___, '0' + str(len(ch2)) + 'b')
        len_start = 0
        for l_len in self.l_length:
            self.crossed_chromosome_array.append(self.crossed_chromosome_str[len_start:len_start + l_len])
            len_start += l_len

        return self.crossed_chromosome_int

    def generate_chromosomes(self):
        choices = random.choices(range(3, 20), k=10)
        self.generate_chromosome(min(choices), max(choices))
        self.mutate_chromosome()

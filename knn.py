import numpy as np

def euclidian_distance(a, b):
    return np.sqrt(np.sum((a-b)**2, axis=1))

class KNN:
    def __init__(self, descriptors_1, descriptors_2, k):
        self.descriptors_1 = descriptors_1
        self.descriptors_2 = descriptors_2
        self.k = k

    def ratio_test(self, match):
        return match.get_distance(0) < 0.5 * match.get_distance(1)

    def solve(self):
        matches = []

        for i, desc in enumerate(self.descriptors_1):

            distances = euclidian_distance(self.descriptors_2, desc)
            distances_sorted = np.argsort(distances)

            match = Match(self, i, distances_sorted[0:self.k])
            if self.ratio_test(match):
                print("Match accepted")
                matches.append(match)
        
        return matches

class Match:
    def __init__(self, my_KNN, index_1, indices_2):
        self.my_KNN = my_KNN
        self.index_1 = index_1
        self.indices_2 = indices_2

    def get_distance(self, ind):
        desc_1 = self.my_KNN.descriptors_1[self.index_1]
        desc_2 = self.my_KNN.descriptors_2[self.indices_2[ind]]
        return np.sqrt(np.sum((desc_1-desc_2)**2))
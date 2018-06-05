import numpy as np
from math import sqrt
from collections import Counter

def Euclidean_Distance(features, labels):
    return np.linalg.norm(np.array(features) - np.array(labels))


def knn_classify(data, predict, k):
    distances = []
    for group in data:
        for features in data[group]:
            euclidean_dis = Euclidean_Distance(features, predict)
            distances.append([euclidean_dis, group])
    knn = [i[1] for i in sorted(distances)][:k]
    print(Counter(knn).most_common(1))
    vote_result = Counter(knn).most_common(1)[0][0]
    return vote_result


dataset = {'k':[[6,5], [1,2],[3,2],[2,3], [4,5]], 'r':[[6,4],[1,3],[3,5],[2,4],[4,6]]}
new_feature = [3,6]
print(knn_classify(dataset, new_feature, 3))
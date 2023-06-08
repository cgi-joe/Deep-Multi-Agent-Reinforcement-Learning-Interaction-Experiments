import numpy as np
from minepy import MINE
from scipy import stats
from collections import Counter

def mic(x, y):
    mine = MINE(alpha=0.6, c=15)
    mine.compute_score(x, y)
    mic = mine.mic()
    return mic

# Example data sets
data_set_1 = [1, 2, 3, 4, 5]
# data_set_2 = [2, 4, 6, 8, 10]
data_set_2 = [1, 2, 3, 4, 5]

# Create 2D arrays from data sets
arr1 = np.array(data_set_1).reshape(-1, 1)
arr2 = np.array(data_set_2).reshape(-1, 1)

# Calculate joint probability distribution
joint_probs = np.concatenate((arr1, arr2), axis=1)

print(joint_probs)

# Calculate joint entropy
joint_entropy = stats.entropy(joint_probs)

# Calculate individual entropies
entropy_1 = stats.entropy(data_set_1)
entropy_2 = stats.entropy(data_set_2)

# Calculate mutual information
mutual_info = entropy_1 + entropy_2 - joint_entropy

print("Joint entropy:", joint_entropy)
print("Entropy of data set 1:", entropy_1)
print("Entropy of data set 2:", entropy_2)
print("Mutual information:", mutual_info)

def normalized_entropy(x, y):
    data = x + y
    unique, counts = np.unique(data, return_counts=True)
    probs = counts / len(data)
    max_entropy = -(probs * np.log2(probs)).sum()
    norm_entropy = 0.0 if max_entropy == 0 else (max_entropy / np.log2(len(unique)))
    return norm_entropy

# Example usage
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]
print(normalized_entropy(x, x))


"""Module for calculating Gini impurity and entropy for decision trees."""
from __future__ import division
import numpy as np

pre_elements = ['A', 'A', 'A', 'C', 'B', 'C']

def counts(elements):
    """Count occurrences of each unique element in the list."""
    classes = {}
    for element in elements:
        if element in classes:
            classes[element] += 1
        else:
            classes[element] = 1
    return [count for _, count in classes.items()]

element_sets = counts(pre_elements)
print(element_sets)

def gini_1(elements):
    """Calculate Gini impurity for a list of elements."""
    total = len(elements)
    impurity = 1.0
    for count in element_sets:
        prob = count / total
        impurity -= prob ** 2
    return impurity

def gini(elements):
    """cts - count per class, n - total number of samples"""
    cts = counts(elements)
    n = sum(cts)
    ### this function returns the total Gini impurity of all the elements, grouped by classes
    return 1 - sum(p_i ** 2 / n ** 2 for p_i in cts)

print("Gini Impurity:", gini(pre_elements))

def entropy_1(elements):
    """Calculate entropy for a list of elements."""
    total = len(elements)
    ent = 0.0
    for count in element_sets:
        prob = count / total
        ent -= prob * np.log2(prob)
    return ent


def entropy(elements):
    """calculates the total entropy of all the elements"""
    if len(elements)==0:
        return 0
    cts = counts(elements)
    n = sum(cts)
    props = 1/n*np.array(cts)
    return -np.dot(np.log2(props), props)

print("Entropy:", entropy(pre_elements))

###above is the code calculating the original dataset's Gini impurity and entropy

for i in range(len(pre_elements)):
    print("========================================")
    left = pre_elements[:i]
    right = pre_elements[i:]
    print(left, right)
    weighted_gini = 1/len(pre_elements)*(gini(left)*len(left)+gini(right)*len(right))
    weighted_entropy = 1/len(pre_elements)*(entropy(left)*len(left)+entropy(right)*len(right))
    print("Weighted Gini Impurity:", weighted_gini)
    print("Weighted Entropy:", weighted_entropy)

###above is the code calculating the weighted Gini impurity and entropy for each possible split

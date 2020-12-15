import os
import abc
import numpy as np

from math import sqrt
from builtins import isinstance

def crawling(filePath, inv = False):
    data = {}
    try:
        with open(filePath) as file:
            for line in file:
                line = line.replace("\n", "")
                tokens = line.split("\t")
                if len(tokens) < 2:
                    continue
                elif len(tokens) == 2:
                    user = tokens[0]
                    item = tokens[1]
                    rating = 0.5
                else:
                    user = tokens[0]
                    item = tokens[1]
                    rating = tokens[2]
                if inv == False:
                    data.setdefault(user, {})
                    data[user][item] = float(rating)
                else:
                    data.setdefault(item, {})
                    data[item][user] = float(rating)
            file.close()
    except IOError as e:
        print(e)
    return data

def cosine(dataA, dataB):
    if type(dataA) is list and type(dataB) is list:
        if len(dataA) != len(dataB):
            return -1
        AB = sum([dataA[i] * dataB[i] for i in range(len(dataA))])
        normA = sqrt(sum([dataA[i] ** 2 for i in range(len(dataA))]))
        normB = sqrt(sum([dataB[i] ** 2 for i in range(len(dataB))]))
        denominator = normA * normB
        if denominator == 0:
            return 0
        return AB / denominator
    elif type(dataA) is dict and type(dataB) is dict:
        interSet = [obj for obj in dataA if obj in dataB]
        if len(interSet) == 0:
            return 0
        AB = sum([dataA[obj] * dataB[obj] for obj in interSet])
        normA = sqrt(sum([dataA[obj] ** 2 for obj in dataA]))
        normB = sqrt(sum([dataB[obj] ** 2 for obj in dataB]))
        denominator = normA * normB
        if denominator == 0:
            return -1
        return AB / denominator
    else:
        return -1
    
def pearson(dataA, dataB, significanceWeighting = False):
    if type(dataA) is list and type(dataB) is list:
        if len(dataA) != len(dataB):
            return -1
        length = len(dataA)
        intersection = [i for i in range(length) if dataA[i] != 0 and dataB[i] != 0]
        if len(intersection) == 0:
            return 0
        meanA = np.mean([dataA[i] for i in range(length) if dataA[i] != 0])
        meanB = np.mean([dataB[i] for i in range(length) if dataB[i] != 0])
        numerator = sum([(dataA[i] - meanA) * (dataB[i] - meanB) for i in intersection])
        deviationA = sqrt(sum([(dataA[i] - meanA) ** 2 for i in intersection]))
        deviationB = sqrt(sum([(dataB[i] - meanB) ** 2 for i in intersection]))
        if (deviationA * deviationB) == 0:
            return 0
        correlation = numerator / (deviationA * deviationB)
    elif type(dataA) is dict and type(dataB) is dict:
        intersection = [obj for obj in dataA if obj in dataB]
        if len(intersection) == 0:
            return 0
        meanA = np.mean([dataA[obj] for obj in dataA.keys()])
        meanB = np.mean([dataB[obj] for obj in dataB.keys()])
        numerator = sum([(dataA[obj] - meanA) * (dataB[obj] - meanB) for obj in intersection])
        deviationA = sqrt(sum([(dataA[obj] - meanA) ** 2 for obj in intersection]))
        deviationB = sqrt(sum([(dataB[obj] - meanB) ** 2 for obj in intersection]))
        if (deviationA * deviationB) == 0:
            return 0
        correlation = numerator / (deviationA * deviationB)
    else:
        return -1
    if significanceWeighting == True:
        if len(intersection) < 50:
            correlation *= (len(intersection) / 50)
    return correlation
    
def jaccard(dataA, dataB):
    nIntersection = sum([1 for obj in dataA if obj in dataB])
    nUnion = len(dataA) + len(dataB) - nIntersection
    if nUnion == 0:
        return -1
    return nIntersection / nUnion

class CollaborativeFiltering(object):
    __metaclass__ = abc.ABCMeta
    
    def __init__(self):
        self.prefs = None
        self.list = None
    
    @classmethod
    @abc.abstractmethod
    def building(cls):
        raise NotImplementedError
    
    @classmethod
    @abc.abstractmethod
    def recommend(cls):
        raise NotImplementedError
    
    def knn(self, target, similarity, k = None):
        similarities = [(similarity(self.prefs[target], self.prefs[other]), other) for other in self.prefs if target != other]
        similarities.sort(reverse = True)
        if k != None:
            similarities = similarities[0:k]
        return similarities

class UserBased(CollaborativeFiltering):        
    def modeling(self, data):
        self.prefs = data
        self.list = {}
        for user in self.prefs:
            for item in self.prefs[user]:
                self.list[item] = None
    
    def building(self, similarity = jaccard, k = None):
        model = {}
        for user in self.prefs:
            model[user] = self.knn(user, similarity, k)
        return model
    
    def predict(self, user, item, nearestNeighbors):
        if item in self.prefs[user]:
            return self.prefs[user][item]
        meanRating = np.mean([score for score in self.prefs[user].values()])
        weightedSum = 0
        normalizingFactor = 0
        for neighbor, similarity in nearestNeighbors.items():
            if item not in self.prefs[neighbor]:
                continue
            meanRatingOfNeighbor = np.mean([r for r in self.prefs[neighbor].values()])
            weightedSum += similarity * (self.prefs[neighbor][item] - meanRatingOfNeighbor)
            normalizingFactor += np.abs(similarity)
        if normalizingFactor == 0:
            return 0
        return meanRating + (weightedSum / normalizingFactor)
    
    def recommend(self, user, similarity = jaccard, model = None, n = None):
        candidateItems = {}
        nearestNeighbors = {}
        for similarity, neighbor in model[user]:
            if similarity <= 0:
                break
            nearestNeighbors[neighbor] = similarity
            for item in self.prefs[neighbor]:
                candidateItems[item] = None
        predicted = [(self.predict(user, item, nearestNeighbors), item)
                            for item in candidateItems if item not in self.prefs[user]]
        predicted.sort(reverse = True)
        recommend = [item for similarity, item in predicted]
        if n != None:
            recommend = recommend[0:n]
        return recommend

def evaluate(testSet, recommender, similarity = None, model = None, n = None):
    totalPrecision = 0
    totalRecall = 0
    
    for user in testSet:
        recommend = recommender.recommend(user, similarity = similarity, model = model, n = n)
        hit = sum([1 for item in testSet[user] if item in recommend])
        precision = hit / n
        recall = hit / len(testSet[user])
        totalPrecision += precision
        totalRecall += recall
    
    result = {}
    result["precision"] = totalPrecision / len(testSet)
    result["recall"] = totalRecall / len(testSet)
    return result

trainSet = crawling("u1.base")
testSet = crawling("u1.test")

ubcf = UserBased()
ubcf.modeling(trainSet)
model = ubcf.building(k=1)

result = evaluate(testSet, ubcf, similarity=jaccard, model=model, n=39)
print(result)

'''
data = crawling("data.dat")

ubcf = UserBased()
ubcf.modeling(data)
model = ubcf.building(k=1)

for user in data.keys():
    recommend = ubcf.recommend(user, model=model, n=39)
    print(user)
    print(recommend)
    print("")
'''
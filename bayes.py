from numpy import mean,std,prod,linalg
from scipy.stats import norm,mode
from sklearn import neighbors


class GaussianNB:
    def __init__(self,features,target):
        self.features = features
        self.target = target
        self.classes = set(target)
        self.model = {i:[] for i in self.classes}

        for f,c in zip(self.features,self.target):
            self.model[c].append(f)

        for c in self.model:
            self.model[c] = [mean(self.model[c],0),std(self.model[c],0),len(self.model[c])/len(self.target)]

    def predict(self,data):
        pred = []
        for d in data:
            probability = []
            for c in self.model:
                probability.append(prod(norm(self.model[c][0],self.model[c][1]).pdf(d)) * self.model[c][-1])
            
            pred.append(probability.index(max(probability)))

        return pred


class Knn:
    def __init__(self,features,target):
        self.features = features
        self.target = target

    def query(self,data,k):
        pred = []
        for d in data:
            dist = linalg.norm(self.features - d,axis=1)
            pred.append(mode(self.target[dist.argsort()[:k]])[0][0])

        return pred


class RandomForest:
    pass
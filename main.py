from numpy import mean,std,prod,linalg
from scipy.stats import norm,mode
import numpy as np
from math import floor,log2


class GaussianNB():
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


class Knn():
    def __init__(self,features,target):
        self.features = features
        self.target = target

    def query(self,data,k):
        pred = []
        for d in data:
            dist = linalg.norm(self.features - d,axis=1)
            pred.append(mode(self.target[dist.argsort()[:k]])[0][0])

        return pred


class Node:
    def __init__(self,feature,threshold,is_root=False):
        self.feature = feature
        self.threshold = threshold
        self.is_root = is_root
        self.left = None
        self.right = None
        self.dataset = None
        self.group = None


class Tree():
    def __init__(self,features,target,max_depth):
        self.nodes = []
        self.max_depth = max_depth
        self.features = features
        self.feature_dim = len(self.features[0])
        self.features_num  = floor(log2(abs(self.feature_dim)) + 1)
        self.target = target.reshape(len(target),1)
        self.dataset = np.concatenate((features, self.target), axis=1)
        self.predictions = []

    def add_node(self,node):
        self.nodes.append(node)

    def create_tree(self):
        # bootstrap dataset
        data = self.dataset[np.random.choice(len(self.dataset),len(self.dataset),replace=True),:]

        for d in range(self.max_depth):
            if d == 0:
                best_feature,threshold = self.best_feature(data)
                node = Node(best_feature,threshold,False if d else True)
                node.dataset = data
                node.group = int(mode(node.dataset[:,-1])[0])
                self.add_node(node)
            else:
                for n in range(2**(d-1)):
                    node = self.nodes[2**(d-1)-1+n]
                    best_feature,threshold = self.best_feature(node.dataset[node.dataset[:,node.feature] < node.threshold])
                    node.left = Node(best_feature,threshold,False if d else True)
                    node.left.dataset = node.dataset[node.dataset[:,node.feature] < node.threshold]
                    node.left.group = int(mode(node.left.dataset[:,-1])[0])
                    self.add_node(node.left)
                    best_feature,threshold = self.best_feature(node.dataset[node.dataset[:,node.feature] > node.threshold])
                    node.right = Node(best_feature,threshold,False if d else True)
                    node.right.dataset = node.dataset[node.dataset[:,node.feature] > node.threshold] 
                    node.right.group = int(mode(node.right.dataset[:,-1])[0])          
                    self.add_node(node.right)

        return self.nodes

    def query(self,data,node=0):
        if node == 0: node = self.nodes[0]
        if node.left or node.right:
            node = node.left if data[node.feature] < node.threshold else node.right
            return self.query(data,node)
        else:
            return node.group

    def predict(self,data):
        for d in data:
            self.predictions.append(self.query(d))

        return self.predictions


    def best_feature(self,dataset):
        selected_features = [int(item) for item in np.linspace(0,self.feature_dim-1,self.feature_dim)[np.random.choice(self.feature_dim,self.features_num,replace=False)]]
        selected_features.sort()
        best_feature_index,threshold = self.info(dataset,selected_features)
        best_feature = selected_features[best_feature_index]
        return (best_feature,threshold)



    def info(self,data,sel_features):
        thresholds,target = ([data.mean(axis=0)[item] for item in sel_features],int(mode(data[:,-1])[0]))
        data = data[:,np.append(sel_features,self.feature_dim)]
        targets = set(data[:,-1])
        total_entropy = 0
        for t in targets:
            p = (data[:,-1] == t).sum()/len(data)
            total_entropy += p * log2(1/p)

        info = []
        for i in range(len(thresholds)):
            tru = (data[:,i] > thresholds[i]).sum()
            fls = (data[:,i] <= thresholds[i]).sum()

            TP = np.logical_and(data[:,i] > thresholds[i],data[:,-1] == target).sum() / tru if tru else 1
            FP = np.logical_and(data[:,i] > thresholds[i],data[:,-1] != target).sum() / tru if tru else 1

            FN = np.logical_and(data[:,i] <= thresholds[i],data[:,-1] == target).sum() / fls if fls else 1
            TN = np.logical_and(data[:,i] <= thresholds[i],data[:,-1] != target).sum() / fls if fls else 1

            E_tru = -TP * (log2(TP) if TP else 0) + -FP * (log2(FP) if FP else 0)
            E_fls = -FN * (log2(FN) if FN else 0) + -TN * (log2(TN) if TN else 0)

            info.append(total_entropy - tru/(tru+fls)*E_tru - fls/(tru+fls)*E_fls)

        ret = info.index(max(info))
        return (ret,thresholds[ret])

class RandomForest:
    def __init__(self,feature,target,trees_no,max_depth):
        self.feature = feature
        self.target = target
        self.trees_no = trees_no
        self.max_depth = max_depth
        self.trees = []

    def fit(self):
        for i in range(self.trees_no):
            tree = Tree(self.feature,self.target,self.max_depth)
            tree.create_tree()
            self.trees.append(tree)

    def predict(self,data):
        predictions = []
        for t in self.trees:
            predictions.append(t.predict(data))

        predictions = np.array(predictions)

        prediction = []
        for i in range(len(predictions[0])):
            prediction.append(int(mode(predictions[:,i])[0]))

        return np.array(prediction)
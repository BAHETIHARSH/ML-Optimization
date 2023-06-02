import random
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from sklearn.model_selection import train_test_split,cross_val_score
import cv2 
import os 
from tqdm import tqdm 
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model


train = "C:/Users/omdod/Desktop/Tomato Leaf Diseases/Training Set"
test = "C:/Users/omdod/Desktop/Tomato Leaf Diseases/Validation Set"

image_size=128

def extract_color_histogram(image, bins=(8, 8, 8)):
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
		[0, 180, 0, 256, 0, 256])
	cv2.normalize(hist, hist)
	return hist.flatten()


def train_data():
    np_img = []
    for directory in tqdm(os.listdir(train)): 
        directory_path = os.path.join(train, directory)
        for image in os.listdir(directory_path):
            path = os.path.join(directory_path, image)
            img = cv2.imread(path) 
            img = cv2.resize(img, (image_size, image_size))
            img = extract_color_histogram(img)
            np_img.append(img)
    return np_img

def test_data():
    np_img = []
    for directory in tqdm(os.listdir(test)): 
        directory_path = os.path.join(test, directory)
        for image in os.listdir(directory_path):
            path = os.path.join(directory_path, image)
            img = cv2.imread(path) 
            img = cv2.resize(img, (image_size, image_size))
            img = extract_color_histogram(img)
            np_img.append(img)
    return np_img


def train_output():
    i = []
    ans = 0
    it = 0
    for directory in os.listdir(train): 
        directory_path = os.path.join(train, directory)
        cur = np.ones((len(os.listdir(directory_path)), 1))*it
        i = np.append(i, cur)
        ans+=len(os.listdir(directory_path))
        it+=1
    return i


def test_output():
    i = []
    ans = 0
    it = 0
    for directory in os.listdir(test): 
        directory_path = os.path.join(test, directory)
        cur = np.ones((len(os.listdir(directory_path)), 1))*it
        i = np.append(i, cur)
        ans+=len(os.listdir(directory_path))
        it+=1
    return i


x_train = train_data() 
x_test = test_data()

x_train = (x_train-np.min(x_train))/(np.max(x_train)-np.min(x_train))
x_test = (x_test-np.min(x_test))/(np.max(x_test)-np.min(x_test))
y_train = train_output()
y_test = test_output()

x_data=np.concatenate((x_train,x_test),axis=0)

y_data=np.concatenate((y_train,y_test),axis=0)


class GeneticOptimizer(ABC):
    def __init__(self, features):
        self.features = features

    # Generate population set of 10 individuals
    def generate_population(self):
        population = []
        for i in range(10):
            tempDict = {}
            for gene in self.features.keys():
                tempDict[gene] = random.choice(self.features[gene])
            population.append(tempDict)
        return population

    # Genrate offsprings from parent1 and parent2 using one point crossover
    def crossing(self, parent1, parent2):
        #   dictonary to array conversion
        genes = list(parent1.keys())
        childs = []
        set1 = []
        set2 = []
        for i in genes:
            set1.append(parent1[i])
            set2.append(parent2[i])
        numberOfFeatures = len(set1)

        # Crossing of two parents

        crossingPT = random.randint(1, numberOfFeatures - 1)

        set1[:crossingPT], set2[:crossingPT] = set2[:crossingPT], set1[:crossingPT]

        #     set to dictonary conversion
        tempchild = {}
        for i in (set1, set2):
            for j in range(len(genes)):
                tempchild[genes[j]] = i[j]
            childs.append(tempchild)
        return childs

    # Generate the mutated offspring using random resetting
    def mutation(self, offsprings):
        gene = random.choice(list(offsprings.keys()))
        Mutated = offsprings.copy()
        Mutated[gene] = random.choice(self.features[gene])

        return Mutated

    def generations(self, n):
        population = self.generate_population()
        #     print(population)
        SelectionQueue = []

        for i in population:
            SelectionQueue.append((self.fitnessFunction(i), i))

        SelectionQueue.sort(reverse=True, key=lambda x: x[0])

        for i in range(n):
            print("Generation ", i + 1)
            for m in range(len(SelectionQueue)):
                print(m + 1, SelectionQueue[m])
            SelectionQueue.sort(reverse=True, key=lambda x: x[0])

            print(f"Crossing Breeding Genertion {i + 1} : ")
            for n in range(0, len(SelectionQueue), 2):
                var = random.randint(1, 100)
                if var == 100:
                    mutated = self.mutation(SelectionQueue[n][1])
                    SelectionQueue.append((self.fitnessFunction(mutated), mutated))
                    print("Mutation Occurs", mutated)
                else:
                    crossBreed = self.crossing(SelectionQueue[n][1], SelectionQueue[n + 1][1])
                    print(f"crossBreed of {n / 2 + 1} parent pair")

                    for j in range(len(crossBreed)):
                        print(j + 1, crossBreed[j])
                        SelectionQueue.append((self.fitnessFunction(crossBreed[j]), crossBreed[j]))

            SelectionQueue.sort(reverse=True, key=lambda x: x[0])
            SelectionQueue = SelectionQueue[:10]
        return SelectionQueue

    @abstractmethod
    def fitnessFunction(self, variable):
        raise NotImplementedError("Must override methodB")


# Random Forest Classifier optimised with genetic optimiser
class GeneticRandomForestClassifierAlgorithm(GeneticOptimizer):
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.features = {
            "n_estimators": [80, 90, 100, 110, 120],
            "criterion": ["gini", "entropy"],
            "max_features": ["sqrt", "log2", None],
            "warm_start": [True, False]
        }
        self.fitnessvalues = {}

        super(GeneticRandomForestClassifierAlgorithm, self).__init__(self.features)

    def fitnessFunction(self, variable):
        if str(variable) in list(self.fitnessvalues.keys()):
            return self.fitnessvalues[str(variable)]
        else:
            clf = RandomForestClassifier(n_estimators=variable["n_estimators"], max_features=variable["max_features"],
                                         warm_start=variable["warm_start"])
            clf.fit(self.x_train, self.y_train)
            val = clf.score(self.x_test, self.y_test)
        self.fitnessvalues[str(variable)] = val
        return val


class GeneticLogesticAlgorithm(GeneticOptimizer):
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.features = {
            "penalty": ['l1', 'l2', 'elasticnet', 'none'],
            "solver": ['lbfgs', 'liblinear', 'newton-cg'],
            "fit_intercept": [True, False],
            "warm_start": [True, False]
        }
        self.fitnessvalues = {}

        super(GeneticLogesticAlgorithm, self).__init__(self.features)

    def fitnessFunction(self, variable):
        if str(variable) in list(self.fitnessvalues.keys()):
            return self.fitnessvalues[str(variable)]
        else:

            clf = linear_model.LogisticRegression(penalty = variable["penalty"],solver = variable["solver"],fit_intercept = variable["fit_intercept"],warm_start = variable["warm_start"])
            try:
                clf.fit(self.x_train, self.y_train)
                val = clf.score(self.x_test, self.y_test)
            except:
                val = -1
        self.fitnessvalues[str(variable)] = val
        return val


class GeneticMLPCAlgorithm(GeneticOptimizer):
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.features = {
            "activation": ['identity', 'logistic', 'tanh', 'relu'],
            "solver": ['lbfgs', 'sgd', 'adam'],
            "learning_rate": ['constant', 'invscaling', 'adaptive'],
            "shuffle": [True, False]
        }
        self.fitnessvalues = {}

        super(GeneticMLPCAlgorithm, self).__init__(self.features)

    def fitnessFunction(self, variable):
        if str(variable) in list(self.fitnessvalues.keys()):
            return self.fitnessvalues[str(variable)]
        else:
            clf = MLPClassifier(activation=variable["activation"], solver=variable["solver"],
                                learning_rate=variable["learning_rate"], shuffle=variable["shuffle"],
                                hidden_layer_sizes=(10, 10), verbose=False)
            try:
                clf.fit(self.x_train, self.y_train)
                val = clf.score(self.x_test, self.y_test)
            except:
                val = -1
            self.fitnessvalues[str(variable)] = val
            return val


class GeneticSVMAlgorithm(GeneticOptimizer):
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.features = {
            "kernel": ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
            "gamma": ['scale', 'auto'],
            "probability": [True, False]
        }
        self.fitnessvalues = {}

        super(GeneticSVMAlgorithm, self).__init__(self.features)

    def fitnessFunction(self, variable):
        if str(variable) in list(self.fitnessvalues.keys()):
            return self.fitnessvalues[str(variable)]
        else:
            clf = svm.SVC(kernel=variable["kernel"], gamma=variable["gamma"], probability=variable["probability"])
            try:
                clf.fit(self.x_train, self.y_train)
                val = clf.score(self.x_test, self.y_test)
            except:
                val = -1
        self.fitnessvalues[str(variable)] = val
        return val


class GeneticSGDAlgorithm(GeneticOptimizer):
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.features = {
            "penalty": ['l2', 'l1', 'elasticnet', None],
            "loss": ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'],
            "shuffle": [True, False]
        }
        self.fitnessvalues = {}

        super(GeneticSGDAlgorithm, self).__init__(self.features)

    def fitnessFunction(self, variable):
        if str(variable) in list(self.fitnessvalues.keys()):
            return self.fitnessvalues[str(variable)]
        else:
            SGD = SGDClassifier(penalty=variable["penalty"], loss=variable["loss"], shuffle=variable["shuffle"])
            SGD.fit(self.x_train, self.y_train)
            val = SGD.score(self.x_test, self.y_test)
            self.fitnessvalues[str(variable)] = val
            return val


class GeneticKNNAlgorithm(GeneticOptimizer):
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.features = {
            "algorithm": ['auto', 'ball_tree', 'kd_tree', 'brute'],
            "weights": ['uniform', 'distance'],
            "n_neighbors": [5, 6, 7, 8, 9, 10]
        }
        self.fitnessvalues = {}

        super(GeneticKNNAlgorithm, self).__init__(self.features)

    def fitnessFunction(self, variable):
        if str(variable) in list(self.fitnessvalues.keys()):
            return self.fitnessvalues[str(variable)]
        else:
            knn = KNeighborsClassifier(algorithm=variable["algorithm"], weights=variable["weights"],
                                       n_neighbors=variable["n_neighbors"], n_jobs=-1)
            knn.fit(self.x_train, self.y_train)
            val = knn.score(self.x_test, self.y_test)
            self.fitnessvalues[str(variable)] = val
            return val

model = GeneticSGDAlgorithm(x_train, y_train, x_test, y_test)
model.generations(4)
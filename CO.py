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



#SVM
def constraint_validation(C,gamma,degree):  # weigths = [0,2] , metric = [0,3] , n_neighbours = [0,15]
    if C<0.001:
        C=0.001
    elif C>100:
        C = 100
        
    if gamma<0.001:
        gamma=0.001
    elif gamma>1:
        gamma = 1
        
    if degree < 1:
        degree = 1
    elif degree > 6:
        degree = 6
        
    return C,gamma,degree

#KNN
def constraint_validation(n_neighbors,weights,metric):  # weigths = [0,2] , metric = [0,3] , n_neighbours = [0,15]
    if weights<0:
        weights=0
    elif weights>2:
        weights = 2
    if metric<0:
        metric=0
    elif metric>3:
        metric = 3
    if n_neighbors < 2:
        n_neighbors = 2
    elif n_neighbors > 15:
        n_neighbors=15
    return n_neighbors,weights,metric    

#MLPC
def constraint_validation(activation,solver,learning_rate,shuffle): # weigths = [0,2] , metric = [0,3] , n_neighbours = [0,15]
    if activation<0:
        activation=0
    elif activation>4:
        activation = 4
        
    if solver<0:
        solver=0
    elif solver>3:
        solver = 3
        
    if learning_rate < 0:
        learning_rate = 0
    elif learning_rate > 3:
        learning_rate = 3
       
    if shuffle < 0:
        shuffle = 0
    elif shuffle > 2:
        shuffle = 2
        
    return activation,solver,learning_rate,shuffle

#SGD
def constraint_validation(penalty,loss,shuffle):  # weigths = [0,2] , metric = [0,3] , n_neighbours = [0,15]
    if penalty<0:
        penalty=0
    elif penalty>4:
        penalty = 4
        
    if loss<0:
        loss=0
    elif loss>5:
        loss = 5
        
    if shuffle < 0:
        shuffle = 0
    elif shuffle > 2:
        shuffle = 2
        
    return penalty,loss,shuffle

# Logistic Regression
def constraint_validation(solver,penalty,C):  # weigths = [0,2] , metric = [0,3] , n_neighbours = [0,15]
    if solver<0:
        solver=0
    elif solver>5:
        solver = 5
    if penalty<0:
        penalty=0
    elif penalty>4:
        penalty = 4
    if C < 0.01:
        C = 0.01
    elif C > 100:
        C = 100
    return solver,penalty,C 

#Random Forest
def constraint_validation(n_estimators,max_features,max_depth,min_samples_split,min_samples_leaf):  # weigths = [0,2] , metric = [0,3] , n_neighbours = [0,15]
    if n_estimators<10:
        n_estimators=10
    elif n_estimators>100:
        n_estimators = 100
        
    if max_features<1:
        max_features=1
    elif max_features>13:
        max_features = 13
        
    if max_depth < 5:
        max_depth = 5
    elif max_depth > 50:
        max_depth = 50
        
    if min_samples_split < 2:
        min_samples_split = 2
    elif min_samples_split > 11:
        min_samples_split = 11
        
    if min_samples_leaf < 1:
        min_samples_leaf = 1
    elif min_samples_leaf > 11:
        min_samples_leaf = 11
        
    return n_estimators,max_features,max_depth,min_samples_split,min_samples_leaf

#SVM
def fitness_function(C,gamma,degree):
      
    model = svm.SVC(C=float(C),gamma=float(gamma),degree=int(degree))
                                   
    scores=np.mean(cross_val_score(model, x_data, y_data, cv=3, n_jobs=-1,scoring="accuracy"))
    return scores

#KNN
def fitness_function(n_neighbors,weights,metric):

    if weights<=1:
        weights='uniform'
    else:
        weights='distance'
    if metric<=1:
        metric='minkowski'
    elif metric<=2:
        metric='euclidean'
    else:
        metric='manhattan'
    model = KNeighborsClassifier(n_neighbors=int(n_neighbors) , weights=weights , metric=metric)

    scores=np.mean(cross_val_score(model, x_data, y_data, cv=3, n_jobs=-1,scoring="accuracy"))
    return scores 

#MLPC
def fitness_function(activation,solver,learning_rate,shuffle):
    
    if activation<=1:
        activation='identity'
    elif activation<=2:
        activation='logistic'
    elif activation<=3:
        activation='tanh'
    else:
        activation='relu'
        
        
    if solver<=1:
        solver='lbfgs'
    elif solver<=2:
        solver='sgd'
    else:
        solver='adam'
        
        
    if learning_rate<=1:
        learning_rate='constant'
    elif learning_rate <=2:
        learning_rate='invscaling'        
    else:
        learning_rate='adaptive'
        
    if shuffle<=1:
        shuffle=True    
    else:
        shuffle=False 
    
    model = MLPClassifier(activation=activation,solver=solver,learning_rate=learning_rate,shuffle=shuffle)
                                   
    scores=np.mean(cross_val_score(model, x_data, y_data, cv=3, n_jobs=-1,scoring="accuracy"))
    return scores

#SGD
def fitness_function(penalty,loss,shuffle):
    
    if penalty<=1:
        penalty='none'
    elif penalty<=2:
        penalty='l1'
    elif penalty<=3:
        penalty='l2'
    else:
        penalty='elasticnet'
    if loss<=1:
        loss='hinge'
    elif loss<=2:
        loss='log'
    elif loss<=3:
        loss='modified_huber'
    elif loss<=4:
        loss='squared_hinge'
    else:
        loss= 'perceptron'
    if shuffle<=1:
        shuffle=True
    else:
        shuffle=False
    model = SGDClassifier(penalty=penalty,loss=loss,shuffle=shuffle)
                                   
    scores=np.mean(cross_val_score(model, x_data, y_data, cv=3, n_jobs=-1,scoring="accuracy"))
    return scores

# Logistic Regression
def fitness_function(solver,penalty,C):
 
    if solver<1:
        solver='newton-cg'
    elif solver<2:
        solver='lbfgs'
    elif solver<3:
        solver='liblinear'
    elif solver<4:
        solver='sag'
    else:
        solver='saga'
    if penalty<1:
        penalty='none'
    elif penalty<2:
        penalty='l1'
    elif penalty<3:
        penalty='l2'
    else:
        penalty='elasticnet'
    
    model = LogisticRegression(solver=solver,penalty=penalty,C=C)

    scores=np.mean(cross_val_score(model, x_data, y_data, cv=3, n_jobs=-1,scoring="accuracy"))
    print(scores)
    return scores

#Random Forest
def fitness_function(n_estimators,max_features,max_depth,min_samples_split,min_samples_leaf):
    
    model = RandomForestClassifier(n_estimators = int (n_estimators) ,max_features = int (max_features),
            max_depth = int(max_depth),min_samples_split = int(min_samples_split),min_samples_leaf = int(min_samples_leaf))
                                   
    scores=np.mean(cross_val_score(model, x_data, y_data, cv=3, n_jobs=-1,scoring="accuracy"))
    return scores 

def co(population,dimension,iteration):
    #initializing Population
    chetta = [[random.uniform(0.001,100) ,random.uniform(0.001,1),random.uniform(1,6)] for i in range(population)] 
    
    lb = [0,0,0] 
    ub = [15,2,3]
    # home territory
    home_territory = chetta
    print(chetta)
    #calculating fitness Value
    chetta_fitness = [fitness_function(c[0],c[1],c[2]) for c in chetta]
    print(chetta_fitness)
    
    #initializing variables
    best_fitness = 0
    best_position = [0,0,0]
    T = 60*(dimension/10)
    best_cost=[]                                        # best_fitness of each iteration
    
    for i in range(population):
        if chetta_fitness[i]>best_fitness:
            best_finess = chetta_fitness[i]            # prey_solution 
            best_position = chetta[i]                 # leader
    best_cost.append(best_fitness)
    t = 0
    it = 0
    while it <= iteration:
        random_chetta = []
        no_of_chetta = math.ceil(random.uniform(2,population))
        print('no of chetta',no_of_chetta)
        n=0
        while(n != no_of_chetta):
            pos = random.randint(0,population-1)
            print(i,pos)
            if pos not in random_chetta:
                n+=1
                random_chetta.append(pos)
        print(random_chetta)
        xb=best_position
        xbest=best_fitness
        for i in range(no_of_chetta): 
            print('no of chetta',no_of_chetta)
            if i < len(random_chetta)-1 :
                neighbor = random_chetta[i+1]
            else:
                neighbor = random_chetta[i-1]
            print(i)
            temp=chetta[random_chetta[i]]
            for j in range(dimension):

                #initializing random constants
                
                r1 = random.uniform(0,1)
                r_hat = random.uniform(-1,1)
                r2 = random.uniform(0,1)
                r3 = random.uniform(0,1)
                c1 = math.exp( 2 * ( 1 - ( t/T )))
                H = c1*(2*r1 - 1)
                alpha = 0.0001 * (t/T) * abs(xb[j] - chetta[random_chetta[i]][j]) 
                beta = chetta[neighbor][j]-chetta[random_chetta[i]][j]
                r = np.random.randn()
                r_check = (abs(r)**(math.exp(r/2))) * math.sin(2*math.pi*r)
                
                
                if r2 < r3:
                    r4 = random.uniform(0,3)
                    if H >= r4:
                        temp[j] = chetta[random_chetta[i]][j] + r_hat**(-1) * alpha   # searching
                    else:
                        temp[j] = xb[j] + r_check * beta      # attcking
                else:
                    temp[j] = chetta[random_chetta[i]][j]
                    
            temp[0],temp[1],temp[2] = constraint_validation(temp[0],temp[1],temp[2])               # validating the new values
            
            temp_fitness = fitness_function(temp[0],temp[1],temp[2])
            
            if temp_fitness > chetta_fitness[random_chetta[i]]:
                chetta[i] = temp
                chetta_fitness[random_chetta[i]] = temp_fitness                         # updating values
                if best_fitness < temp_fitness:
                    best_fitness = temp_fitness
                    best_position = temp                            # leader update
                        
        t += 1
        best_cost.append(best_fitness) 
                
        it+=1          
        if best_fitness < xbest:
            best_fitness = xbest
            best_position = xb
        print('Cost : ',best_fitness)
        print('Position : ',best_position)
                        
dimension=3
population = 10
iteration=10
co(population,dimension,iteration)


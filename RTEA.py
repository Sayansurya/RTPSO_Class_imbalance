#RTEA
import io
import numpy as np
import pandas as pd
import random
import math,time,sys,os
from matplotlib import pyplot
from datetime import datetime
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, roc_auc_score

popSize = 20
maxIter = 100
c1 = 2
c2 = 2
WMAX = 0.9
WMIN = 0.4
Pm = 0.2

df = pd.read_csv('cancer_classification.csv')
# df=pd.read_csv("liver-disorders_csv.csv")
a,b = np.shape(df)
# print(a,b)
data = df.values[:,0:b-1]
label = df.values[:,b-1]
dimension = data.shape[1]

for s in range(1):
    cross = 5
    test_size = (1/cross)
    trainX, testX, trainy, testy = train_test_split(data, label,stratify=label ,test_size=test_size,random_state=(7+17*int(time.time()%1000)))
    # trainX, testX, trainy, testy = train_test_split(data,label,test_size=0.1, random_state=42)
    clf=KNeighborsClassifier(n_neighbors=5)
    # print(np.shape(trainX))
    clf.fit(trainX,trainy)
    val=clf.score(testX,testy)
    #print("Acc: ", val)


    trainX_i, testX_i, trainy_i, testy_i = train_test_split(trainX, trainy,stratify=trainy ,test_size=test_size,random_state=(7+17*int(time.time()%1000)))

    # def initialise(dimension):
    #     population=np.zeros((popSize,dimension))
    #     minn = 1
    #     maxx = math.floor(0.5*dimension)

    #     if maxx<minn:
    #         maxx = minn + 1
    #         #not(c[i].all())

    #     for i in range(popSize):
    #         random.seed(i**3 + 10 + time.time() ) 
    #         no = random.randint(minn,maxx)
    #         if no == 0:
    #             no = 1
    #         random.seed(time.time()+ 100)
    #         pos = random.sample(range(0,dimension-1),no)
    #         for j in pos:
    #             population[i][j]=1

    #     return population

    def initialise(dimension, select):
        population = np.zeros((popSize, dimension))
        random.seed(time.time() + 1000)
        pos = random.sample(range(0,dimension-1), select)
        for i in range(popSize):
            for j in pos:
                population[i][j] = 1

        return population

    def fitness(agent, majority_index, minority_index, trainX, testX, trainy, testy):
        clf=KNeighborsClassifier(n_neighbors=5)
        rows1 = []
        for i in range(len(agent)):
            if(agent[i] == 1):
                rows1.append(majority_index[i])

        rows2 = minority_index.copy()

        rows = rows1+rows2

        train_data=[trainX[i,:] for i in rows]
        test_data=testX.copy()

        model=clf.fit(train_data,trainy[rows])

        false_positive_rate,true_positive_rate,thresholds=roc_curve(testy,model.predict_proba(test_data)[:,1])
        return auc(false_positive_rate, true_positive_rate)


    def allfit(population, majority_index, minority_index, trainX, testX, trainy, testy):
        fit = []
        for i in range(len(population)):
            fit.append(fitness(population[i], majority_index, minority_index, trainX, testX, trainy, testy))

        return fit



    def SMO(x):
        for i in range(len(x)):
            random.seed(i**3 + 10 + time.time() ) 
            rnd = random.random()
            if (rnd <= Pm):
                x[i] = 1 - x[i]

        return x


    def RTEA():
        count_0, count_1 = 0, 0
        index_0 = []
        index_1 = []
        for i in range(len(trainy_i)):
            if(trainy_i[i] == 1):
                count_0 += 1
                index_0.append(i)
            else:
                count_1 += 1
                index_1.append(i)

        if count_0 > count_1:
            majority_index = index_0.copy()
            minority_index = index_1.copy()
        else:
            majority_index = index_1.copy()
            minority_index = index_0.copy()



        population = initialise(len(majority_index), len(minority_index))
        popfit = allfit(population, majority_index, minority_index, trainX_i, testX_i, trainy_i, testy_i)
        for i in range(maxIter):
            for j in range(popSize):
                random.seed(j**3 + 10 + time.time() )
                one = random.randint(0,popSize-1)
                two = random.randint(0,popSize-1)
                three = random.randint(0,popSize-1)
                four = random.randint(0, popSize-1)

                One, Two, Three, Four = population[one], population[two], population[three], population[four]


                y,z = np.array([]), np.array([])

                random.seed(j**4 + 40 + time.time()*500)
                r = random.random()
                if (r <= 0.5):
                    y = np.append(y, np.add(One, np.multiply(Four, np.add(Two, Three)))%2)
                else:
                    y = np.append(y, np.add(One, np.add(Two, Three))%2)


                z = np.append(z,SMO(y))

                if(fitness(z, majority_index, minority_index, trainX_i, testX_i, trainy_i, testy_i) < fitness(population[j], majority_index, minority_index, trainX_i, testX_i, trainy_i, testy_i)): 
                    population[j] = z.copy()
                    popfit[j] = fitness(z, majority_index, minority_index, trainX_i, testX_i, trainy_i, testy_i).copy()
                    gbestVal=popfit[j]
                    gbestVec=population[j].copy()
            print("maxiter",i)  
            print(gbestVal)          
                
                
        fitsrt = np.argsort(popfit)
        gbestVal = popfit[fitsrt[-1]]
        gbestVec = population[fitsrt[-1]]
        rows1 = []
        for i in range(len(gbestVec)):
            if(gbestVec[i] == 1):
                rows1.append(majority_index[i])

        rows2 = minority_index.copy()

        rows = rows1+rows2

        train_data=[trainX_i[i,:] for i in rows]
        train_y=trainy_i[rows]

        model=clf.fit(train_data,train_y)
        val=model.score(testX,testy)
        #print("Test_val: ",val)
#         return val
        """
        count_index = []
        for j in range(len(majority_index)):
            count_index.append(0)

        count_majority = 0
        for j in range(len(population)):
            for k in range(len(population[0])):
                if (population[j][k] == 1):
                    count_index[k] += 1

        for j in range(len(count_index)):
            if(count_index[j]!=0):
                count_majority += 1

        for j in range(len(majority_index)):
            for k in range(j+1,len(majority_index)):
                if (count_index[j]<count_index[k]):
                    count_index[j], count_index[k] = count_index[k], count_index[j]
                    majority_index[j], majority_index[k] = majority_index[k], majority_index[j]
    #                 swap(count_index[j], count_index[k])
    #                 swap(majority_index[j], majority_index[k])

        rows1 = []
        if (count_majority > len(minority_index)):
            for j in range(len(minority_index)):
                rows1.append(majority_index[j])

        else:
            for j in range(majority_index):
                rows1.append(majority_index[j])

        rows2 = minority_index.copy()

        rows = rows1+rows2

        train_data=[trainX_i[i,:] for i in rows]
        train_y=trainy_i[rows]

        model=clf.fit(train_data,train_y)
        val=model.score(testX,testy)
        print("Test_val: ",val)
        print()
        """






    RTEA()
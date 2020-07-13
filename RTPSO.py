#datasets
#Abalone19, Yeast5, Kddcup-guess_passwd_vs_satan, poker-9_vs_7, Abalone9-18, Led7digit-0-2-4-5-6-7-8-9_vs_1, Page-blocks

from google.colab import files
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
from sklearn.ensemble import AdaBoostClassifier
from imblearn.metrics import geometric_mean_score

uploaded = files.upload()
df = pd.read_csv(io.BytesIO(uploaded['file_name']))

#RTEA
pop = [5,10,30,40,50]
maxIter = 50
c1 = 2
c2 = 2
WMAX = 0.9
WMIN = 0.4
Pm = 0.2

#df = pd.read_csv('cancer_classification.csv')
# df=pd.read_csv("liver-disorders_csv.csv")
a,b = np.shape(df)
# print(a,b)
data = df.values[:,0:b-1]
label = df.values[:,b-1]
dimension = data.shape[1]
for j in range(5):
  print("popsize",pop[j])
  for s in range(2):
      popSize=pop[j]
      print("popsize",popSize)
      cross = 5
      test_size = (1/cross)
      trainX, testX, trainy, testy = train_test_split(data, label,stratify=label ,test_size=test_size,random_state=(7+17*int(time.time()%1000)))
      #trainX, testX, trainy, testy = train_test_split(data,label,test_size=0.2, random_state=42)
      #clf=KNeighborsClassifier(n_neighbors=5)
      # print(np.shape(trainX))
      #clf.fit(trainX,trainy)
      abc = AdaBoostClassifier(n_estimators=50,
                              learning_rate=1)
      model=abc.fit(trainX,trainy)
      false_positive_rate,true_positive_rate,thresholds=roc_curve(testy,model.predict_proba(testX)[:,1])
      val=auc(false_positive_rate, true_positive_rate)
      #val=clf.score(testX,testy)
      print("initial Acc: ", val)


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
          #clf=KNeighborsClassifier(n_neighbors=5)
          rows1 = []
          for i in range(len(agent)):
              if(agent[i] == 1):
                  rows1.append(majority_index[i])

          rows2 = minority_index.copy()

          rows = rows1+rows2

          train_data=[trainX[i,:] for i in rows]
          test_data=testX.copy()

          #model=clf.fit(train_data,trainy[rows])
          abc = AdaBoostClassifier(n_estimators=50,
                              learning_rate=1)
          model=abc.fit(train_data,trainy[rows])
          false_positive_rate,true_positive_rate,thresholds=roc_curve(testy,model.predict_proba(test_data)[:,1])
          #val=auc(false_positive_rate, true_positive_rate)
          #false_positive_rate,true_positive_rate,thresholds=roc_curve(testy,model.predict_proba(test_data)[:,1])
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

          #model=clf.fit(train_data,train_y)
          #val=model.score(testX,testy)
          abc = AdaBoostClassifier(n_estimators=50,
                              learning_rate=1)
          model=abc.fit(train_data,train_y)
          false_positive_rate,true_positive_rate,thresholds=roc_curve(testy,model.predict_proba(testX)[:,1])
          val1=auc(false_positive_rate, true_positive_rate)
          print("Test_val: ",val1)
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
              for j in (majority_index):
                  rows1.append(majority_index[j])

          rows2 = minority_index.copy()

          rows = rows1+rows2

          train_data=[trainX_i[i,:] for i in rows]
          train_y=trainy_i[rows]
          clf=KNeighborsClassifier(n_neighbors=5)
          model=clf.fit(train_data,train_y)
          val=model.score(testX,testy)
          abc = AdaBoostClassifier(n_estimators=50,
                              learning_rate=1)
          #model=abc.fit(train_data,train_y)
          #false_positive_rate,true_positive_rate,thresholds=roc_curve(testy,model.predict_proba(testX)[:,1])
          #val2=auc(false_positive_rate, true_positive_rate)
          print("Test_val2: ",val)
          print()
          """
          






      RTEA()

# pso

pop = [5,10,30,40,50] #50 #40 #30 #10 # 5 #20
maxIter = 50
c1 = 2
c2 = 2
WMAX = 0.9
WMIN = 0.4
w=0.5
#df = pd.read_csv('cancer_classification.csv')
# df=pd.read_csv("liver-disorders_csv.csv")
a,b = np.shape(df)
# print(a,b)
data = df.values[:,0:b-1]
label = df.values[:,b-1]
dimension = data.shape[1]
for j in range(5):
  print("popsize",pop[j])
  for i in range(0,2):
    popSize=pop[j]
    print("popsize",popSize)
    cross = 5
    test_size = (1/cross)
    trainX, testX, trainy, testy = train_test_split(data, label,stratify=label ,test_size=test_size,random_state=(7+17*int(time.time()%1000)))
    # trainX, testX, trainy, testy = train_test_split(data,label,test_size=0.1, random_state=42)
    abc = AdaBoostClassifier(n_estimators=50,
                                learning_rate=1)
        
    model=abc.fit(trainX,trainy)
    false_positive_rate,true_positive_rate,thresholds=roc_curve(testy,model.predict_proba(testX)[:,1])
      
    val1=auc(false_positive_rate, true_positive_rate)
    print("Acc: ", val1)


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
        #clf=KNeighborsClassifier(n_neighbors=5)
        rows1 = []
        for i in range(len(agent)):
            if(agent[i] == 1):
                rows1.append(majority_index[i])
        
        rows2 = minority_index.copy()
        
        rows = rows1+rows2
        
        train_data=[trainX[i,:] for i in rows]
        test_data=testX.copy()
        abc = AdaBoostClassifier(n_estimators=50,
                                learning_rate=1)
        model=abc.fit(train_data,trainy[rows])

        false_positive_rate,true_positive_rate,thresholds=roc_curve(testy,model.predict_proba(test_data)[:,1])
        #m=geometric_mean_score(testy, model.predict(testX), average='macro')
        #score = f1_score(testy,model.predict(testX))
        return auc(false_positive_rate, true_positive_rate)

      
    def allfit(population, majority_index, minority_index, trainX, testX, trainy, testy):
        fit = []
        for i in range(len(population)):
            fit.append(fitness(population[i], majority_index, minority_index, trainX, testX, trainy, testy))
            
        return fit

    def sigmoid(gamma):
        if gamma < 0:
            return 1 - 1/(1 + math.exp(gamma))
        else:
            return 1/(1 + math.exp(-gamma))
        
        
    def PSO():
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
        
        velocity=np.zeros((np.shape(population)[0],len(majority_index)))
        pbestVec = population.copy()
        pbestVal = popfit.copy()

        fitsrt = np.argsort(popfit)
        gbestVal = popfit[fitsrt[-1]]
        gbestVec = population[fitsrt[-1]]
        
        for curIter in range(maxIter):
    #       popnew=np.zeros((np.shape(population)[0],len(majority_index)))
    #       fitList = allfit(population,train_majority_X,train_majority_y,test_majority_X,test_majority_y)
            for i in range(np.shape(population)[0]):
                if(popfit[i]>pbestVal[i]):
                  pbestVal[i]=popfit[i]
                  pbestVec[i]=population[i].copy()
                #gbest
            for i in range(np.shape(population)[0]):
                if(popfit[i]>(gbestVal)):
                  gbestVal=popfit[i]
                  gbestVec=population[i].copy()
          
          
    #         print("gbest: ",gbestVal)


            W = WMAX - (curIter/maxIter)*(WMAX - WMIN )
    #         print("W : ",W)
        
            for inx in range(np.shape(population)[0]):
              random.seed(time.time()+10)
              r1=c1*random.random()
              random.seed(time.time()+19)
              r2=c2*random.random()
            
              x = np.subtract(pbestVec[inx] , population[inx])
              y=np.subtract(gbestVec,population[inx])
            
              velocity[inx]=np.multiply(W,velocity[inx])+np.multiply(r1,x)+np.multiply(r2,y)
    #           population[inx] = np.add(population[inx],velocity[inx])
              s = np.add(population[inx],velocity[inx])
            
              for j in range(len(s)):
                  if(random.random() < sigmoid(s[j])):
                      s[j] = 1
                  else:
                      s[j] = 0
                    
              fits = fitness(s, majority_index, minority_index, trainX_i, testX_i, trainy_i, testy_i)
              if(fits>popfit[inx]):
                    population[inx] = s.copy()
                    popfit[inx] = fits
    #           popfit[inx] = fitness(population[inx], majority_index, minority_index, trainX_i, testX_i, trainy_i, testy_i)

          
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

        #model=clf.fit(train_data,train_y)
        #val=model.score(testX,testy)
        abc = AdaBoostClassifier(n_estimators=50,
                                learning_rate=1)
        # print(np.shape(trainX))
        model=abc.fit(train_data,train_y)
        false_positive_rate,true_positive_rate,thresholds=roc_curve(testy,model.predict_proba(testX)[:,1])
      #print(auc(false_positive_rate, true_positive_rate))
        val=auc(false_positive_rate, true_positive_rate)
        print("Test_val: ",val)
    #     return val
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

        #model=clf.fit(train_data,train_y)
        #val=model.score(testX,testy)
        abc = AdaBoostClassifier(n_estimators=50,
                                learning_rate=1)
        # print(np.shape(trainX))
        model=abc.fit(train_data,train_y)
        false_positive_rate,true_positive_rate,thresholds=roc_curve(testy,model.predict_proba(testX)[:,1])
      #print(auc(false_positive_rate, true_positive_rate))
        val2=auc(false_positive_rate, true_positive_rate)
        print("Test_val2: ",val2)
        """
        
        
        
    PSO()



#RTPSO
pop = [5,10,20,30,40,50]
maxIter = 50
c1 = 2
c2 = 2
WMAX = 0.9
WMIN = 0.4
Pm = 0.2
initial_max=0
w=1/3
#df = pd.read_csv('cancer_classification.csv')
# df=pd.read_csv("liver-disorders_csv.csv")
a,b = np.shape(df)
# print(a,b)
data = df.values[:,0:b-1]
label = df.values[:,b-1]
dimension = data.shape[1]
for j in range(6):
  print("popsize",pop[j])
  for s in range(2):
      popSize=pop[j]
      print("popsize",popSize)
      cross = 5
      test_size = (1/cross)
      trainX, testX, trainy, testy = train_test_split(data, label,stratify=label ,test_size=test_size,random_state=(7+17*int(time.time()%1000)))
      # trainX, testX, trainy, testy = train_test_split(data,label,test_size=0.1, random_state=42)
      #clf=KNeighborsClassifier(n_neighbors=5)
      abc = AdaBoostClassifier(n_estimators=50,
                              learning_rate=1)
      # print(np.shape(trainX))
      model=abc.fit(trainX,trainy)
      false_positive_rate,true_positive_rate,thresholds=roc_curve(testy,model.predict_proba(testX)[:,1])
    #print(auc(false_positive_rate, true_positive_rate))
      val1=auc(false_positive_rate, true_positive_rate)
    #val=clf.score(testX,testy)
      if(val1>initial_max):
        initial_max=val1
      print("AUC initial: ", val1)
      #val=clf.score(testX,testy)

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
          #clf=KNeighborsClassifier(n_neighbors=5)
          rows1 = []
          for i in range(len(agent)):
              if(agent[i] == 1):
                  rows1.append(majority_index[i])

          rows2 = minority_index.copy()

          rows = rows1+rows2

          train_data=[trainX[i,:] for i in rows]
          test_data=testX.copy()
          abc = AdaBoostClassifier(n_estimators=50,
                              learning_rate=1)
          model=abc.fit(train_data,trainy[rows])
          #m=geometric_mean_score(testy, model.predict(test_data), average='macro')
          #score = f1_score(testy,model.predict(test_data))
          false_positive_rate,true_positive_rate,thresholds=roc_curve(testy,model.predict_proba(test_data)[:,1])
          return (auc(false_positive_rate, true_positive_rate)) 


      def allfit(population, majority_index, minority_index, trainX, testX, trainy, testy):
          fit = []
          for i in range(len(population)):
              fit.append(fitness(population[i], majority_index, minority_index, trainX, testX, trainy, testy))

          return fit


      def sigmoid(gamma):
          if gamma < 0:
              return 1 - 1/(1 + math.exp(gamma))
          else:
              return 1/(1 + math.exp(-gamma))


      def SMO(x):
          for i in range(len(x)):
              random.seed(i**3 + 10 + time.time() ) 
              rnd = random.random()
              if (rnd <= Pm):
                  x[i] = 1 - x[i]

          return x


      def RTEA(population, popfit, majority_index, minority_index, trainX_i, testX_i, trainy_i, testy_i):
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

          return population, popfit


      def RTPSO():
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

          velocity=np.zeros((np.shape(population)[0],len(majority_index)))
          pbestVec = population.copy()
          pbestVal = popfit.copy()

          fitsrt = np.argsort(popfit)
          gbestVal = popfit[fitsrt[-1]]
          gbestVec = population[fitsrt[-1]]

          for curIter in range(maxIter):
      #       popnew=np.zeros((np.shape(population)[0],len(majority_index)))
      #       fitList = allfit(population,train_majority_X,train_majority_y,test_majority_X,test_majority_y)
              for i in range(np.shape(population)[0]):
                  if(popfit[i]>pbestVal[i]):
                    pbestVal[i]=popfit[i]
                    pbestVec[i]=population[i].copy()
                  #gbest
              for i in range(np.shape(population)[0]):
                  if(popfit[i]>(gbestVal)):
                    gbestVal=popfit[i]
                    gbestVec=population[i].copy()


      #         print("gbest: ",gbestVal)


              W = WMAX - (curIter/maxIter)*(WMAX - WMIN )
      #         print("W : ",W)

              for inx in range(np.shape(population)[0]):
                random.seed(time.time()+10)
                r1=c1*random.random()
                random.seed(time.time()+19)
                r2=c2*random.random()

                x = np.subtract(pbestVec[inx] , population[inx])
                y=np.subtract(gbestVec,population[inx])

                velocity[inx]=np.multiply(W,velocity[inx])+np.multiply(r1,x)+np.multiply(r2,y)
      #           population[inx] = np.add(population[inx],velocity[inx])
      #           popfit[inx] = fitness(population[inx], majority_index, minority_index, trainX_i, testX_i, trainy_i, testy_i)
                s = np.add(population[inx],velocity[inx])

                for j in range(len(s)):
                    if(random.random() < sigmoid(s[j])):
                        s[j] = 1
                    else:
                        s[j] = 0

                fits = fitness(s, majority_index, minority_index, trainX_i, testX_i, trainy_i, testy_i)
                if(fits>popfit[inx]):
                      population[inx] = s.copy()
                      popfit[inx] = fits
      #           popfit[inx] = fitness(population[inx], majority_index, minority_index, trainX_i, testX_i, trainy_i, testy_i)
              population, popfit = RTEA(population, popfit, majority_index, minority_index, trainX_i, testX_i, trainy_i, testy_i)

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
          abc = AdaBoostClassifier(n_estimators=50,
                              learning_rate=1)
          model=abc.fit(train_data,train_y)
          #val=model.score(testX,testy)
          false_positive_rate,true_positive_rate,thresholds=roc_curve(testy,model.predict_proba(testX)[:,1])
          val=auc(false_positive_rate, true_positive_rate)
          print("Test_val: ",val)
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

          #model=clf.fit(train_data,train_y)
          abc = AdaBoostClassifier(n_estimators=50,
                              learning_rate=1)
      # print(np.shape(trainX))
          model=abc.fit(train_data,train_y)
          false_positive_rate,true_positive_rate,thresholds=roc_curve(testy,model.predict_proba(testX)[:,1])
    #print(auc(false_positive_rate, true_positive_rate))
          val2=auc(false_positive_rate, true_positive_rate)
    #val=clf.score(testX,testy)
          #val=model.score(testX,testy)
          print("Test2_val: ",val2)
          print()
          """



      RTPSO()
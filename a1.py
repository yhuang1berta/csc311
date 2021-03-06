# -*- coding: utf-8 -*-
"""csc311_A1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1BmCgUTnUIAjM-NZ47tsFFKIXnQ9LkOHA
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
import sklearn.linear_model as lin
import sklearn.neighbors as ngh

# In the functions below,
# X = input data
# T = data labels
# w = weight vector for decision boundary
# b = bias term for decision boundary
# elevation and azimuth are angles describing the 3D viewing direction

import numpy.random as rnd
rnd.seed(3)
print('\n\nQuestion 1')
print('----------')
print('\nQuestion 1(a):')
B = np.random.rand(4,5)
print(B)
print('\nQuestion 1(b):')
y = np.random.rand(4,1)
print(y)
print('\nQuestion 1(c):')
C = B.reshape((2,10))
print(C)
print('\nQuestion 1(d):')
D = B - y
print(D)
print('\nQuestion 1(e):')
z = y.reshape(4)
print(z)
print('\nQuestion 1(f):')
B[:,3] = z
print(B)
print('\nQuestion 1(g):')
D[:,0] = B[:,2] + z
print(D)
print('\nQuestion 1(h):')
print(B[:3])
print('\nQuestion 1(i):')
print(B[:,[1,3]])
print('\nQuestion 1(j):')
print(np.log(B))
print('\nQuestion 1(k):')
print(np.sum(B))
print('\nQuestion 1(l):')
print(np.amax(B, axis=0))
print('\nQuestion 1(m):')
print(np.max(B.sum(axis=1)))
print('\nQuestion 1(n):')
print(np.matmul(B.transpose(), D))
print('\nQuestion 1(j):')
print(y.transpose()@D@D.transpose()@y)

print('\n\nQuestion 2')
print('----------')

# Q2(a)
def matrix_poly(A):
  #helper
  def mat_mul(X,Y):
    # calculate X * Y
    mat = np.zeros(X.shape)
    elem_sum = 0
    for i in range(X.shape[0]):
      for j in range(Y.shape[1]):
        for k in range(Y.shape[0]):
          elem_sum += X[i,k] * Y[k,j]
        mat[i,j] = elem_sum
        elem_sum = 0
    return mat

  # find A*A
  final = mat_mul(A,A)
  # find A + A*A
  for i in range(A.shape[0]):
    for j in range(A.shape[1]):
      final[i,j] += A[i,j]
  # find A*(A + A*A)
  final = mat_mul(A,final)
  # find A + (A*(A + A*A))
  for i in range(A.shape[0]):
    for j in range(A.shape[1]):
      final[i,j] += A[i,j]

  return final

# Q2(b)
def timing(N):
  A = np.random.rand(N,N)
  loop_start = time.time()
  B1 = matrix_poly(A)
  loop_end = time.time()
  np_start = time.time()
  B2 = A + (A@(A+(A@A)))
  np_end = time.time()
  print("Magnitude of B1-B2: " + str(np.linalg.norm(B1-B2, 2)))
  print("Execution time for naive iterative method with N = " + str(N) + " is " + str(loop_end - loop_start))
  print("Execution time for vectorized method with N = " + str(N) + " is " + str(np_end - np_start))


# test = np.arange(9).reshape(3,3)
# print(matrix_poly(test))
# print(test + (test@(test + (test @ test))))

print("\nQuestion 2(c):")
print("N = 100:")
timing(100)
print("N = 300:")
timing(300)
print("N = 1000:")
timing(1000)

# Q3(a)
def least_squares(x,t):
  X = np.ones((x.shape[0], 2))
  X[:,1] = x
  w = np.linalg.inv(X.transpose()@X) @ X.transpose() @ t
  return w

# print(least_squares(dataTrain[0],dataTrain[1]))

# Q3(b)
def plot_data(x,t):
  b, a = least_squares(x,t)
  min_x, max_x = np.min(x), np.max(x)
  pt1 = [min_x, max_x]
  pt2 = [a*min_x+b, a*max_x+b]
  plt.scatter(x,t)
  plt.plot(pt1,pt2,color="r")
  plt.title("Question 3(b): the fitted line")
  plt.show()
  return a,b

# plot_data(dataTrain[0],dataTrain[1])

# Q3(c)
def error(a,b,X,T):
  est_mat = a*X+b
  mse = np.mean(np.square(T-est_mat))
  return mse

# a,b = least_squares(dataTrain[0],dataTrain[1])
# error(a,b,dataTrain[0],dataTrain[1])

print('\n\nQuestion 3')
print('----------')

# Q3(d)
# Read the training and test data from the file dataA1Q3.pickle
with open('dataA1Q3.pickle','rb') as f:
  dataTrain, dataTest = pickle.load(f)

# Call plot_data to fit a line to the training data
train_a,train_b = plot_data(dataTrain[0],dataTrain[1])

print("\nQuestion 3(d):")
# Print the values of a and b for the fitted line
print("a: "+str(train_a))
print("b: "+str(train_b))

# Compute and print the training error
print("Mean Square Error of training data: " + str(error(train_a,train_b,dataTrain[0],dataTrain[1])))

# Compute and print the test error
print("Mean Square Error of test data: " + str(error(train_a, train_b, dataTest[0],dataTest[1])))

def boundary_mesh(X,w,w0):
    # decision boundary
    X = X.T
    xmin = np.min(X[0])
    xmax = np.max(X[0])
    zmin = np.min(X[2])
    zmax = np.max(X[2])
    x = np.linspace(xmin,xmax,2)
    z = np.linspace(zmin,zmax,2)
    xx,zz = np.meshgrid(x,z)
    yy = -(xx*w[0] + zz*w[2] + w0)/w[1]
    return xx,yy,zz


def plot_data(X,T,elevation=30,azimuth=30):
    colors = np.array(['r','b'])    # red for class 0 , blue for class 1
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = np.array(['r','b'])    # red for class 0 , blue for class 1
    X = X.T
    ax.scatter(X[0],X[1],X[2],color=colors[T],s=1)
    ax.view_init(elevation,azimuth)
    plt.draw()
    return ax,fig
    

def plot_db(X,T,w,w0,elevation=30,azimuth=30):
    xx,yy,zz, = boundary_mesh(X,w,w0)
    ax,fig = plot_data(X,T,elevation,azimuth)
    ax.plot_surface(xx,yy,zz,alpha=0.5,color='green')
    return ax,fig


def plot_db3(X,T,w,w0):
    _,fig1 = plot_db(X,T,w,w0,30,0)
    _,fig2 = plot_db(X,T,w,w0,30,45)
    _,fig3 = plot_db(X,T,w,w0,30,175)
    return fig1,fig2,fig3
    

def movie_data(X,T):
    ax,fig = plot_data(X,T,30,-20)
    plt.pause(1)
    for angle in range(-20,200):
        ax.view_init(30, angle)
        plt.draw()
        plt.pause(0.0001)
    return ax
        

def movie_db(X,T,w,w0):
    xx,yy,zz,= boundary_mesh(X,w,w0)
    ax,fig = plot_data(X,T,30,-20)
    ax.plot_surface(xx,yy,zz,alpha=0.3,color='green')
    plt.pause(1)
    for angle in range(-20,200):
        ax.view_init(30, angle)
        plt.draw()
        plt.pause(0.0001)
    return ax

with open("dataA1Q4v2.pickle","rb") as f:
  Xtrain,Ttrain,Xtest,Ttest = pickle.load(f)

clf = lin.LogisticRegression()
clf.fit(Xtrain, Ttrain)
w = clf.coef_[0]
bias = clf.intercept_[0]

print("\nQuestion 4")
print("----------")

print('\nQuestion 4(a):')
print("Weight: " + str(w))
print("Bias: " + str(bias))

print('\nQuestion 4(b):')
accuracy1 = clf.score(Xtest,Ttest)
comparison = np.equal(clf.predict(Xtest), Ttest)
accuracy2 = np.count_nonzero(comparison == True) / Ttest.shape[0]
print("accuracy1: " + str(accuracy1))
print("accuracy2: " + str(accuracy2))
print("accuracy1 - accuracy2: " + str(accuracy1 - accuracy2))

# Q4(c).
ax,fig = plot_db(Xtrain,Ttrain,w,bias,30,5)
fig.suptitle("Question 4(c): Training data and decision boundary")

# Q4(d).
ax,fig = plot_db(Xtrain,Ttrain,w,bias,30,20)
fig.suptitle("Question 4(d): Training data and decision boundary")

# plot_data(Xtrain, Ttrain,30,10)

print('\n\nQuestion 6')
print('----------')

# Q5 (a)-(k)
def gd_logreg(lrate):
  # Q5(a). initialize weight
  np.random.seed(3)
  # Q5(b).
  w0 = np.random.randn(Xtrain.shape[1]+1)/1000
  w1 = w0.copy()
  # add x0=1 to Xtrain and Ttrain
  unbiased_train = np.ones((Xtrain.shape[0],Xtrain.shape[1]+1))
  unbiased_train[:,1:] = Xtrain
  unbiased_test = np.ones((Xtest.shape[0],Xtest.shape[1]+1))
  unbiased_test[:,1:] = Xtest
  # Q5(c). all helper functions below are needed
  def sigma(z):
    return 1/(1+np.exp(-z))
  def z(x,w):
    return x@w
  def h(x,w):
    return sigma(z(x,w))
  def gd(x,t,w):
    # gradient of L_ce = [X^T(y-t)]
    return 1/(Ttrain.shape[0]) * x.transpose()@(h(x,w)-t)
  def E(x,t,w):
    # logistic-cross-entropy
    return (t@np.logaddexp(0,-z(x,w))+(1-t)@np.logaddexp(0,z(x,w)))/t.shape[0]
  train_CE = []
  test_CE = []
  train_acc = []
  test_acc = []
  E0 = E(unbiased_train,Ttrain,w0)
  E1 = 1
  
  # Q5(d).
  while abs(E0-E1) >= np.float64(10**-10):
  # for i in range(200):
    E0 = E1
    w0 = w1.copy()
    weight_update = gd(unbiased_train,Ttrain,w1)
    w1 -= lrate * weight_update
    train_est_mat = np.where(z(unbiased_train,w1)>=0,1,0)
    test_est_mat = np.where(z(unbiased_test,w1)>=0,1,0)
    train_compare = np.equal(train_est_mat,Ttrain)
    train_acc.append(np.count_nonzero(train_compare==True)/Ttrain.shape[0])
    test_compare = np.equal(test_est_mat,Ttest)
    test_acc.append(np.count_nonzero(test_compare==True)/Ttest.shape[0])
    E1 = E(unbiased_train,Ttrain,w1)
    train_CE.append(E1)
    test_CE.append(E(unbiased_test,Ttest,w1))

  # Q5(e).
  print("Q4 outputs:")
  print("Weight: " + str(w))
  print("Bias: " + str(bias))
  print("Q5 outputs:")
  print("Bias: "+str(w1[0]))
  print("final weight vector = "+str(w1[1:]))
  print("learning rate: " + str(lrate))

  # Q5(f).
  plt.plot(train_CE)
  plt.plot(test_CE,color="r")
  plt.suptitle("Question 5: Training and test loss v.s. iterations")
  plt.xlabel("Iteration number")
  plt.ylabel("Cross entropy")
  plt.show()

  # Q5(g)
  plt.semilogx(train_CE)
  plt.semilogx(test_CE,color="r")
  plt.suptitle("Question 5: Training and test loss v.s. iterations (log scale)")
  plt.xlabel("Iteration number")
  plt.ylabel("Cross entropy")
  plt.show()

  # Q5(h)
  plt.semilogx(train_acc)
  plt.semilogx(test_acc,color="r")
  plt.suptitle("Question 5: Training and test accuracy v.s. iterations (log scale)")
  plt.xlabel("Iteration number")
  plt.ylabel("Accuracy")
  plt.show()

  # Q5(i).
  plt.plot(train_CE[-100:])
  plt.suptitle("Question 5: last 100 training cross entropies")
  plt.xlabel("Iteration number")
  plt.ylabel("Cross entropy")
  plt.show()

  # Q5(j).
  plt.semilogx(test_CE[50:],color="r")
  plt.suptitle("Question 5: test loss from iteration 50 on (log scale)")
  plt.xlabel("Iteration number")
  plt.ylabel("Cross entropy")
  plt.show()

  # Q5(k).
  ax,fig = plot_db(unbiased_train,Ttrain,w1[1:],w1[0],30,5)
  fig.suptitle("Question 5: Training data and decision boundary")


  return w1

# print("lrate = 10")
# print(gd_logreg(10))
# print("lrate = 3")
# print(gd_logreg(3))
print("\nQuestion 5(e):")
print(gd_logreg(1))
# print("lrate = 0.3")
# print(gd_logreg(0.3))
# print("lrate = 0.1")
# print(gd_logreg(0.1))

with open('mnistTVT.pickle','rb') as f:
  Xtrain,Ttrain,Xval,Tval,Xtest,Ttest = pickle.load(f)

# Q6(a).
def reduce_train(Xtrain,Ttrain):
  reduced_Ttrain_index = np.where((Ttrain == 5) | (Ttrain == 6), True, False)
  full_reduced_Xtrain = Xtrain[reduced_Ttrain_index]
  full_reduced_Ttrain = Ttrain[reduced_Ttrain_index]
  return full_reduced_Xtrain, full_reduced_Ttrain

# Q6(b).
def plot_first_16():
  full_reduced_Xtrain, full_reduced_Ttrain = reduce_train(Xtrain,Ttrain)
  for i in range(16):
    plt.subplot(4,4,i+1)
    plt.axis(False)
    plt.imshow(full_reduced_Xtrain[i].reshape((28,28)),cmap="Greys",interpolation="nearest")
  plt.suptitle("Question 6(b): 16 MNIST training images.")
  plt.plot()

plot_first_16()

def train_with(target1,target2,Xtrain,Ttrain,Xval,Tval,Xtest,Ttest):
  # Note: the reason why I'm including the data-reduction and ploting part in 
  # here is because if I modify "reduce_train" function from pervious, and call 
  # it in this function, the one(occasional several) of the return numpy arrays
  # will become a tuple, and will even fail to be converted to a numpy array
  # using np.array(). I do believe it is a problem caused by the machine, and
  # I'm unable to solve it within the time this assignment is due.

  # reducing training data
  reduced_Ttrain_index = np.where((Ttrain == target1) | (Ttrain == target2), True, False)
  full_reduced_Xtrain = Xtrain[reduced_Ttrain_index]
  full_reduced_Ttrain = Ttrain[reduced_Ttrain_index]
  small_reduced_Xtrain = full_reduced_Xtrain[:2000]
  small_reduced_Ttrain = full_reduced_Ttrain[:2000]
  # reducing validation data
  reduced_Tval_index = np.where((Tval == target1) | (Tval == target2), True, False)
  reduced_Xval = Xval[reduced_Tval_index]
  reduced_Tval = Tval[reduced_Tval_index]
  # reducing testing data
  reduced_Ttest_index = np.where((Ttest == target1) | (Ttest == target2), True, False)
  reduced_Xtest = Xtest[reduced_Ttest_index]
  reduced_Ttest = Ttest[reduced_Ttest_index]
  
  # print("Done reducing data!")

  # fit each k into model
  val_acc = []
  train_acc = []
  best_val_acc, best_k = -1, None
  # Q6(c). step i: loop through odd k [1,19] to find best k
  for k in range(1,20,2):
    knn = ngh.KNeighborsClassifier(k)
    knn.fit(full_reduced_Xtrain,full_reduced_Ttrain)
    val_acc.append(knn.score(reduced_Xval, reduced_Tval))
    train_acc.append(knn.score(small_reduced_Xtrain,small_reduced_Ttrain))
    # Q6(c). step iii
    if best_val_acc < val_acc[-1]:
      best_val_acc = val_acc[-1]
      best_k = k
    # print("k = " + str(k) + " Done!")

  # Q6(c). step ii: plot all k
  plt.plot(train_acc)
  plt.plot(val_acc,color="r")
  plt.xticks([x for x in range(10)],labels=[i for i in range(1,20,2)])
  plt.suptitle("Question 6(c): Training and Validation Accuracy for KNN, digits "+str(target1)+" and "+str(target2))
  plt.xlabel("Number of Neighbours, K")
  plt.ylabel("Accuracy")

  # Q6(c). step iv: print out best k output
  knn_best = ngh.KNeighborsClassifier(best_k)
  knn_best.fit(full_reduced_Xtrain,full_reduced_Ttrain)
  knn_best_acc = knn_best.score(reduced_Xtest, reduced_Ttest)
  # Q6(c). step v,vi:
  print("best k value: " + str(best_k))
  print("best k validation accuracy: " + str(val_acc[best_k//2]))
  print("best k test accuracy" + str(knn_best_acc))

# train models with 5,6 as target
print("Question 6")
print("----------")
print("\nQuestion 6(c):")
train_with(5,6,Xtrain,Ttrain,Xval,Tval,Xtest,Ttest)

# Q6(d). train models with 4,7 as target
print("\nQuestion 6(d):")
train_with(4,7,Xtrain,Ttrain,Xval,Tval,Xtest,Ttest)




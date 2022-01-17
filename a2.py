import numpy as np
import matplotlib.pyplot as plt
import random
import time

### all of our data sets
data = []
crime_train_df = np.genfromtxt("pp2data/train-crime.csv", delimiter=',', dtype='float')
crime_train_labels = np.genfromtxt("pp2data/trainR-crime.csv", delimiter=',', dtype='float')
crime_test_df = np.genfromtxt("pp2data/test-crime.csv", delimiter=',', dtype='float')
crime_test_labels = np.genfromtxt("pp2data/testR-crime.csv", delimiter=',', dtype='float')
data.append([crime_train_df, crime_train_labels, crime_test_df, crime_test_labels])

wine_train_df = np.genfromtxt("pp2data/train-wine.csv", delimiter=',', dtype='float')
wine_train_labels = np.genfromtxt("pp2data/trainR-wine.csv", delimiter=',', dtype='float')
wine_test_df = np.genfromtxt("pp2data/test-wine.csv", delimiter=',', dtype='float')
wine_test_labels = np.genfromtxt("pp2data/testR-wine.csv", delimiter=',', dtype='float')
data.append([wine_train_df, wine_train_labels, wine_test_df, wine_test_labels])

artsmall_train_df = np.genfromtxt("pp2data/train-artsmall.csv", delimiter=',', dtype='float')
artsmall_train_labels = np.genfromtxt("pp2data/trainR-artsmall.csv", delimiter=',', dtype='float')
artsmall_test_df = np.genfromtxt("pp2data/test-artsmall.csv", delimiter=',', dtype='float')
artsmall_test_labels = np.genfromtxt("pp2data/testR-artsmall.csv", delimiter=',', dtype='float')
data.append([artsmall_train_df, artsmall_train_labels, artsmall_test_df, artsmall_test_labels])

artlarge_train_df = np.genfromtxt("pp2data/train-artlarge.csv", delimiter=',', dtype='float')
artlarge_train_labels = np.genfromtxt("pp2data/trainR-artlarge.csv", delimiter=',', dtype='float')
artlarge_test_df = np.genfromtxt("pp2data/test-artlarge.csv", delimiter=',', dtype='float')
artlarge_test_labels = np.genfromtxt("pp2data/testR-artlarge.csv", delimiter=',', dtype='float')
data.append([artlarge_train_df, artlarge_train_labels, artlarge_test_df, artlarge_test_labels])

# task 1
# function for calculating correct weights
def regularization(lamb, phi, labels):
    return np.matmul(np.linalg.inv(lamb*np.identity(phi.shape[1]) + np.matmul(np.transpose(phi), phi)), np.transpose(phi)).dot(labels)

# mean squared error 
def MSE(phi, w, t):
    return (((phi).dot(w) - t)**2).mean()


titles = ['crime', 'wine', 'artsmall', 'artlarge']
lambdas = list(range(0, 151))


train_err_by_set = []
test_err_by_set = []

# for each dataset
for dataset in data:
    train_df = dataset[0]
    train_labels = dataset[1]
    test_df = dataset[2]
    test_labels = dataset[3]


    train_err = []
    test_err = []
    # for each individual lambda find the weights and testing MSE, store appropriately
    for lamb in lambdas:
        w = regularization(lamb, train_df, train_labels)    

        train_mse = MSE(train_df, w, train_labels) 
        test_mse = MSE(test_df, w, test_labels)
        train_err.append(train_mse)
        test_err.append(test_mse)

    train_err_by_set.append(train_err)
    test_err_by_set.append(test_err)


# graphing our results
for i in range(len(train_err_by_set)):
    plt.plot(lambdas, train_err_by_set[i], label='training error')
    plt.plot(lambdas, test_err_by_set[i], label='testing error')
    plt.xlabel('regularization parameter lambda')
    plt.ylabel('MSE')
    plt.legend()
    plt.title(titles[i])
    plt.show()




print()
print('Part 2')

# part 2

# this is my function for correctly splitting training set into k folds
def get_indexes(labels, k):
    indexes = list(range(len(labels))) # might need to subtract or add 1 here
    random.shuffle(indexes)

    folds = [[] for _ in range(k)]

    j=0
    for i in range(len(indexes)):
        if j == k: # once j hits 10 we'll be out of bounds and need to append to fold[0]
            j=0
        folds[j].append(indexes[i])

        j += 1

    return folds


results_by_dataset = {} # for storing final results
for x in range(len(data)):
    start = time.time()
    train_df = data[x][0]
    train_labels = data[x][1]
    test_df = data[x][2]
    test_labels = data[x][3]

    folds = get_indexes(train_labels, 10) # get a list of list of indexes representing our folds

    avg_mse_by_lambda = []
    # for every lambda we'll train on 9 folds and test on 1, 10 separate times and store the average
    for lamb in lambdas:
        results_by_fold = []
        for i in range(len(folds)):
            train_i = []
            test_i = []

            # setting up training and testing indexes based on folds
            for j in range(len(folds)):
                if j==i:
                    test_i = folds[j]
                else:
                    train_i += folds[j]
                
            train_X = train_df[train_i]
            train_Y = train_labels[train_i]
            test_X = train_df[test_i]
            test_Y = train_labels[test_i]

            # our results
            w = regularization(lamb, train_X, train_Y)
            mse = MSE(test_X, w, test_Y)

            results_by_fold.append(mse)
        
        # stores avg performance of lambdas over 10 folds and stores
        avg_mse_by_lambda.append(sum(results_by_fold)/len(results_by_fold))

    # stores the best lambda out of 150
    lowest_mse = min(avg_mse_by_lambda)
    best_lambda= avg_mse_by_lambda.index(lowest_mse)

    end = time.time()

    # stores best lambda, it's mse, and time to compute
    results_by_dataset[titles[x]] = {
                                "best lambda":best_lambda,
                                "associated mse":lowest_mse, 
                                "associated runtime":end-start}

# crime: middle
# wine: low
# artsmall: 15 
# artlarge: 25

for key in results_by_dataset.keys():
    print(key)
    print(results_by_dataset[key])

print()
print('Part 3')

# part 3
# (3.54) from Bishop
def findSn(alpha, beta, phi):
    return np.linalg.inv(alpha*np.identity(phi.shape[1]) + beta*np.matmul(np.transpose(phi), phi))

# (3.53)
def findMn(beta, Sn, phi, t):
    return beta*np.matmul(np.matmul(Sn, np.transpose(phi)), t)

# (3.91)
def findGamma(alpha, lamb):
    sum=0
    for l in lamb:
        sum += l/(alpha+l)
    
    return sum

# (3.92)
def findAlpha(gamma, m_N):
    return gamma/np.matmul(np.transpose(m_N), m_N)

# (3.95)
def findBeta(N, gamma, t, m_N, phi): # do in a for loop
    step1 = (t - np.matmul(phi, m_N))**2
    step2 = sum(step1)
    step3 = step2/(N-gamma)
    return 1/step3

# (3.87)
def findEigenvalues(beta, phi):
    return np.linalg.eig(beta*np.matmul(np.transpose(phi), phi))[0]

# our check to see if alpha and beta have converged, will return False if they haven't and True if they have
def condition(alpha_old, alpha_new, beta_old, beta_new):
    return abs(alpha_old-alpha_new) < .0001 and abs(beta_old-beta_new) < .0001

results_by_dataset = {}
for i in range(len(data)):
    start = time.time()

    dataset = data[i]
    train_X = dataset[0]
    train_y = dataset[1]
    test_X = dataset[2]
    test_y = dataset[3]

    # randomly initialize alpha and beta, along with variables to store results from previous iteration
    alpha_new = random.uniform(1,10)
    beta_new = random.uniform(1,10)
    alpha_old, beta_old = 0, 0

    while not condition(alpha_old, alpha_new, beta_old, beta_new):
        # iteratively refining alpha and beta
        S_N = findSn(alpha_new, beta_new, train_X)
        m_N = findMn(beta_new, S_N, train_X, train_y)
        eigenvalues = findEigenvalues(beta_new, train_X)
        gamma = findGamma(alpha_new, eigenvalues)

        alpha_old = alpha_new
        beta_old = beta_new

        alpha_new = findAlpha(gamma, m_N)
        beta_new = findBeta(len(train_X), gamma, train_y, m_N, train_X)
    
    end = time.time()


    # storing our results for each dataset
    results_by_dataset[titles[i]] = {
        "alpha": alpha_new,
        "beta": beta_new,
        "lambda": alpha_new/beta_new,
        "MSE": MSE(test_X, m_N, test_y),
        "time": (end-start)
    }





for key in results_by_dataset.keys():
    print(key)
    print(results_by_dataset[key])
import numpy as np
np.random.seed(0)
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.neighbors import KNeighborsClassifier

def test_regression_model(model, trainX, trainY, testX, testY):
    model.fit(trainX, trainY)
    y_pred = model.predict(testX)
    acc = model.score(testY, y_pred)
    print('TEST REGRESSION MODEL: Your accuracy is %s' % str(acc))
    y1 = np.array([1, 2, 3])
    y2 = np.array([1, 2, 3])
    acc1 = model.score(y2, y1)
    y1 = np.array([0.5, 0.5, 0.5, 0.5])
    y2 = np.array([1, 0, 1, 0])
    acc2 = model.score(y2, y1)
    assert (trainX.shape[1] == model.W.shape[0]), 'Number of features (including feature "1") and length of model.W are not the same'
    assert (acc <= 1.0 and acc1 == 1 and abs(acc2) < 0.1), 'Wrong implementation of reg.score()'
    assert (acc >= 0.5), 'Your code or parameters are not good even for a linear dependency'
    return

def generate_regression_data(Nfeat=100, Mtrain=150, Mtest=150):
    W = np.random.rand(Nfeat+1)
    trainX = np.c_[np.random.rand(Mtrain, Nfeat), np.ones(Mtrain)]
    trainY = np.matmul(trainX, W)
    testX = np.c_[np.random.rand(Mtest, Nfeat), np.ones(Mtest)]
    testY = np.matmul(testX, W)
    return trainX, trainY, testX, testY

def generate_data(n_features=100, n_train_samples=150, n_test_samples=150, 
                  is_regression=True, seed=0):
    W = np.random.rand(n_features+1)

    trainX = np.c_[np.random.rand(n_train_samples, n_features), np.ones(n_train_samples)]
    trainY = np.matmul(trainX, W)

    testX = np.c_[np.random.rand(n_test_samples, n_features), np.ones(n_test_samples)]
    testY = np.matmul(testX, W)
    #print(trainX.shape, trainY.shape, testX.shape, testY.shape)
    if not is_regression:
        trainY[trainY > 0.5] = 1
        trainY[trainY <= 0.5] = 0
        testY[testY > 0.5] = 1
        testY[testY <= 0.5] = 0

    return trainX, trainY, testX, testY

def test_score(model):
    is_regression = 'Regressor' in str(type(model))
    if not is_regression:
        #classification accuracy test
        y_pred = [1, 0, 1, 0, 1]
        y_gt = [1, 1, 1, 0, 0]
        assert (model.score(y_pred, y_gt) == accuracy_score(y_gt, y_pred)), 'Wrong output of score method'
        print('score test passed')
    else:
        y_pred = [1., 2., 3., 4., 5.] 
        y_gt = [5.,4., 3., 2. ,1.]
        assert (model.score(y_pred, y_gt) == mean_squared_error(y_gt, y_pred)), 'Wrong output of score method'
        print('score test passed')  

def test_kneighbors(model):
    trainX, trainY, testX, testY = generate_data(is_regression=False)
    is_approx =  'approx' in str(type(model)).lower()
    if not is_approx:
        sklearn_model = KNeighborsClassifier(n_neighbors=1, weights='uniform', algorithm='brute')
        
        sklearn_model.fit(trainX, trainY)
        sk_neigh_dist, sk_neigh_indarray = sklearn_model.kneighbors(trainX[:1], n_neighbors=1)
        
        model.fit(trainX, trainY)
        neigh_dist, neigh_indarray = model.kneighbors(trainX[:1], n_neighbors=1)
        
        condition1 = 'float' in  str(type(neigh_dist[0, 0]))
        condition2 = 'int' in str(type(neigh_indarray[0, 0])) 
        assert (condition1 and condition2), 'Wrong output type'
        condition = (np.sum(np.abs(np.array(neigh_indarray) - sk_neigh_indarray)) == 0)
        message = 'Wrong nearest neighbor search'
        print(neigh_indarray, sk_neigh_indarray)
        assert condition, message
        print('kneighbors test passed')

def accuracy_test(model):
    print('accuracy test passed')
    
def test_fit_predict(model):
    is_regression = 'Regressor' in str(type(model))
    trainX, trainY, testX, testY = generate_data(is_regression=is_regression)
    model = model.fit(trainX, trainY)
    y_pred = model.predict(testX)
    
    assert (y_pred.shape == testY.shape), 'Wrong output shape of predict method'
    assert (type(y_pred[0]) == type(testY[0])), 'Wrong output type of predict method'
    print('fit_predict test passed')
    
    if 'predict_proba' in dir(model):
        y_pred_proba = model.predict_proba(testX)
        
        condition01 = np.mean((np.sum(y_pred_proba, axis=1)==1.0)) == 1.0
        condition02 = np.sum(y_pred_proba < 0.0) == 0.0
        condition03 = np.sum(y_pred_proba > 1.0) == 0.0
        condition = condition01 and condition02 and condition03
        assert condition, 'Wrong output of predict method'
        print('predict_proba test passed')
    
def test_knn_model(model):
    test_fit_predict(model)
    test_score(model)
    test_kneighbors(model)
    accuracy_test(model)
    print('Test passed')

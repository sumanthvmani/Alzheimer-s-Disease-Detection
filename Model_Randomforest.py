import numpy as np
from sklearn.ensemble import RandomForestClassifier
from Evaluation_nrml import evaluation


def Model_Randomforest(train_data, train_target, test_data, test_target, Hidden=None):
    print('Random Forest')
    if Hidden is None:
        Hidden = 125
    IMG_SIZE = 10
    Train_Temp = np.zeros((train_data.shape[0], IMG_SIZE))
    for i in range(train_data.shape[0]):
        Train_Temp[i, :] = np.resize(train_data[i], IMG_SIZE)
    train_data = Train_Temp.reshape(Train_Temp.shape[0], IMG_SIZE)
    Test_Temp = np.zeros((test_data.shape[0], IMG_SIZE))
    for i in range(test_data.shape[0]):
        Test_Temp[i, :] = np.resize(test_data[i], IMG_SIZE)
    test_data = Test_Temp.reshape(Test_Temp.shape[0], IMG_SIZE)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    pred = np.zeros(test_target.shape)
    for i in range(test_target.shape[1]):
        clf.fit(train_data, train_target[:, i])
        Y_pred = clf.predict(test_data)
        pred[:, i] = np.asarray(Y_pred)

    avg = np.mean(pred)
    pred[pred >= avg] = 1
    pred[pred < avg] = 0
    pred = pred.astype('int')
    Eval = evaluation(test_target, pred)
    return Eval, pred
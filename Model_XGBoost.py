import numpy as np
from xgboost import XGBClassifier
from Evaluation_nrml import evaluation


def Model_XGBoost(train_data, train_target, test_data, test_target, HN=None, LR=None, act=None, neighbors=5):
    print('Model_Xgboost')
    IMG_SIZE = 10

    Train_Temp = np.zeros((train_data.shape[0], IMG_SIZE))
    for i in range(train_data.shape[0]):
        Train_Temp[i, :] = np.resize(train_data[i], IMG_SIZE)
    train_data = Train_Temp.reshape(Train_Temp.shape[0], IMG_SIZE)

    Test_Temp = np.zeros((test_data.shape[0], IMG_SIZE))
    for i in range(test_data.shape[0]):
        Test_Temp[i, :] = np.resize(test_data[i], IMG_SIZE)

    test_data = Test_Temp.reshape(Test_Temp.shape[0], IMG_SIZE)
    model = XGBClassifier(learning_rate=0.01, objective="multi:softprob", num_class=test_target.shape[1], random_state=42, eval_metric="mlogloss" )
    model.fit(train_data, np.argmax(train_target, axis=1))
    prob = model.predict_proba(test_data)
    pred = np.zeros_like(prob)
    pred[np.arange(len(prob)), np.argmax(prob, axis=1)] = 1
    avg = np.mean(pred)
    pred[pred >= avg] = 1
    pred[pred < avg] = 0
    pred = pred.astype('int')
    Eval = evaluation(test_target, pred)
    return Eval, pred
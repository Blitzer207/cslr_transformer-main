"""
This file contains the data imputation techniques used in this work.
"""
import numpy as np
from numpy import mean
from numpy import std
from pandas import read_csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def impute(data: np.ndarray, method: str = "knn") -> np.ndarray:
    """
    Predicts missing features value
    :param method: Type of imputation technique to use. knn or iterative
    :param data: input data with missing values
    :return: input with values
    
    缺失的值在numpy中用np.nan表示
    """
    imputer = None
    if method.lower() == "knn": 
        imputer = KNNImputer()
        data = imputer.fit_transform(data)
    elif method.lower() == "iterative":
        imputer = IterativeImputer()
        imputer.fit(data)
        data = imputer.transform(data)
    elif method.lower() == "mean":
        imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
        imputer.fit(data)
        data = imputer.transform(data)
    else:
        raise ValueError("Please provide a valid imputation method. knn or iterative")
    #将特定条件下的缺失值置为0
    #第一段代码遍历数据的每一行（除去第一行和最后一行），如果某一行的上一行和下一行的第一个值都为0（精确到小数点后3位），那么就将该行的第一个值设为0。
    for i in range(1, data.shape[0] - 1, 1):
        if (data[i - 1][0], 3) == 0.000 and round(data[i + 1][0], 3) == 0.000:
            data[i][0] = 0.000
    #如果数据的左上角和右下角的值都为0，那么就将该值设为0,data形状是二维数组 [帧数，特征数]=[n,258 or 398]
    for row in range(1, data.shape[0] - 1, 1): 
        for col in range(1, data.shape[1] - 1, 1):
            if (
                round(data[row - 1][col - 1], 3) == 0.000 
                and round(data[row + 1][col + 1], 3) == 0.000
            ):
                data[row][col] = 0.000

    """
    model = RandomForestClassifier()
    pipeline = Pipeline(steps=[('i', imputer), ('m', model)])
    # define model evaluation
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    # evaluate model
    scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    print('Mean Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
    """
    return data

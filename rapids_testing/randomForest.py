import cupy as cp
from cuml.ensemble import RandomForestClassifier as cuRFC

X = cp.random.normal(size=(10,4)).astype(cp.float32)
y = cp.asarray([0,1]*5, dtype=cp.int32)

cuml_model = cuRFC(max_features=1.0,
                   n_bins=8,
                   n_estimators=40)
cuml_model.fit(X,y)
# RandomForestClassifier()
cuml_predict = cuml_model.predict(X)

print("Predicted labels : ", cuml_predict)
# Predicted labels :  [0. 1. 0. 1. 0. 1. 0. 1. 0. 1.]
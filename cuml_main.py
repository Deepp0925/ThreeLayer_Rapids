from three_layer_classification.three_layer_model import threeLayerHSIClassification
from three_layer_classification.guidedMedianFilter import guidedMedianFilter
# from sklearn.ensemble import RandomForestClassifier
from cuml.ensemble import RandomForestClassifier
        
X_train, y_train = build_dataset(img, train_gt, ignored_labels=IGNORED_LABELS)
# TODO find equivalent of this in cuml
X_train, y_train = sklearn.utils.shuffle(X_train, y_train)
class_weight = "balanced" if CLASS_BALANCING else None
train1 = time.perf_counter()
# Start Training
#Train the first layer of the system.
clf = threeLayerHSIClassification() 
clf.fit(X_train, y_train)
        
#transform training samples for Random Forest (Second Layer)
X_train_transformed = clf.transform(X_train)  
# added np.nan_to_num to allow for Fusion dataset to run
X_train_transformed = np.nan_to_num(X_train_transformed)
y_train = np.nan_to_num(y_train)
rf = RandomForestClassifier(n_estimators=500)
rf.fit(X_train_transformed, y_train)
        
#transform all dataset for generating the whole image
X_transformed = clf.transform(img.reshape(-1, N_BANDS))
# added np.nan_to_num to allow for Fusion dataset to run
X_transformed = np.nan_to_num(X_transformed)
#Stop Training
training_time = time.perf_counter() - train1      
        
#Start Testing
test_time1 = time.perf_counter()        
#save_model(clf, MODEL, DATASET)
prediction = rf.predict(X_transformed)
prediction = prediction.reshape(img.shape[:2])
#Third layer starts here. GMF is applied to prediction image
prediction = guidedMedianFilter(prediction,img)
#prediction = prediction.reshape(img.shape[:2])
#Stop Testing
testing_time = time.perf_counter() - test_time1 
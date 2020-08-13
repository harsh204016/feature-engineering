from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedKFold,GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.metrics import make_scorer

model= MultiOutputClassifier(lgb.LGBMClassifier(random_state=10,
                                               is_unbalance=True,
                                               objective='multiclass',
                                               device= 'gpu',
                                               gpu_platform_id=0,
                                               gpu_device_id=0))

parameters = {
     "estimator__n_estimators": [70, 100,150,200],
     "estimator__max_depth":[8,20,50,-1],
     "estimator__num_leaves":[31,50,75],
     "estimator__bosting":['dart','goss','gbdt'],
     "estimator__bagging_fraction":[0.6,0.5,1],
     "estimator__learning_rate":[0.1,0.005,0.09,0.15],
     "estimator__num_iterations":[100,200,500]
}

rkf = RepeatedKFold(
    n_splits=10,
    n_repeats=2,
    random_state=10
)
score = make_scorer(acc)
cv = GridSearchCV(
    model,
    parameters,
    #cv=rkf,
    scoring= score,
    n_jobs=-1,
    verbose=4)
    
cv.fit(train_data[cols].values,y.values)
cv.best_params_

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import KFold

DEFAULT_CV = KFold(n_splits=5, random_state=42)

def compile_model(model_fn,  input_shape, optimizer='adam',):
    def get_model():
        model = model_fn()
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.build(input_shape=input_shape)
        return model

    return get_model


def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, *args, **kwargs):
    model.fit(X_train, y_train, *args, **kwargs)
    
    keras_eval = model.evaluate(X_test, y_test)
    
    pred = model.predict(X_test).argmax(axis = 1)
    f1_micro = f1_score(pred, y_test.argmax(axis = 1), average = 'micro')
    f1_macro = f1_score(pred, y_test.argmax(axis = 1), average = 'macro')
    accur = accuracy_score(pred, y_test.argmax(axis = 1))
    
    return list(keras_eval) + [f1_micro, f1_macro, accur]
    

def custom_cross_val(cr_f, X, y, cv, *args, **kwargs):
    cr_f().summary()
    eval_res = list()
    for i, (train, test) in enumerate(cv.split(y)):
        model = cr_f()
        print('Running Fold', i+1, '/', cv.n_splits)
        eval1 = train_and_evaluate_model(model, 
                                         [X[j][train] for j in range(len(X))], y[train], 
                                         [X[j][test] for j in range(len(X))], y[test], 
                                         *args, **kwargs)
        
        print()
        print('Fold result: ', eval1)
        eval_res.append(eval1)
    
    return np.array(eval_res)


def describe_cv_result(cv_res):
    print(cv_res)
    mean_cv_res = cv_res.mean(axis = 0)
    std_cv_res = cv_res.std(axis = 0)
    print('Mean')
    print(pd.DataFrame([mean_cv_res], columns = ['loss', 'keras_accur', 'micro_f1', 'macro_f1', 'accur']))
    print('Std')
    print(pd.DataFrame([std_cv_res], columns = ['loss', 'keras_accur', 'micro_f1', 'macro_f1', 'accur']))

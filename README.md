# Alternative-Way-of-performing-BBS

The objective of this project is to use machine learning models to validate and compare their predictions against a professional’s assessment of a health patient using the Berg Balance Scale. This study is to determine whether using subset features from the BBS as training data for ML models can be a standalone method of predicting patient fall risk.

## Project Items

- [All tables can be found in the excel files in "excels" directory (excels/*)](excels)

- [All images and figures files can be found in "figures" directory (figures/*)](figures)

- [All code files in "jupyter_notebooks" directory (jupyter_notebooks/*)](jupyter_notebooks)

## Code Example

The execution of 8192 scenarios of SVM for one subset. lofull = 8192

```py
ero = []
ers = []
ero0 = 0
ers0 = 0
lo = lofull

for i in range(1,lo):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=i)
    
    
    # NO PARAMETER ADJUSTMENT
    model = SVC()
    model.fit(X_train,y_train)
    y_predict = model.predict(X_test)
    ero.append(np.mean(y_predict != y_test))
    if np.mean(y_predict != y_test) <= 0.015:
        ero0 += 1
    
    
    # WITH PARAMETER ADJUSTMENT
    param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001]}
    grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=0)
    grid.fit(X_train,y_train)
    grid_predict = grid.predict(X_test)
    ers.append(np.mean(grid_predict != y_test))
    if np.mean(grid_predict != y_test) <= 0.015:
        ers0 += 1
    
    
plt.figure(figsize=(12,6))
plt.plot(range(1,lo),ero,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='green', markersize=7)
plt.plot(range(1,lo),ers,color='red', linestyle='dashed', marker='o',
         markerfacecolor='yellow', markersize=7)
plt.title('Error Rate vs. Random State')
plt.xlabel('Random State')
plt.ylabel('Error Rate')

print('ero0=',ero0)
print('ers0=',ers0)
```

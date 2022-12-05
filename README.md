# Alternative-Way-of-performing-BBS

The objective of this project is to use machine learning models to validate and compare their predictions against a professional’s assessment of a health patient using the Berg Balance Scale. This study is to determine whether using subset features from the BBS as training data for ML models can be a standalone method of predicting patient fall risk.

## Project Items

- [All tables can be found in the excel files in "excels" directory (excels/*)](excels)

- [All images and figures files can be found in "figures" directory (figures/*)](figures)

- [All code files in "jupyter_notebooks" directory (jupyter_notebooks/*)](jupyter_notebooks)

## Figure Examples

Correlation Heatmap

<img src="figures/Correlation Heatmap.png" width="600em" />

SVM (8192 runs)

<img src="figures/SVM Scaled vs Unscaled 7 High Corr rs101 BF8192.png" width="600em" />

## Code Example

The execution of SVM for one subset.

```py
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=rs)

# NO PARAMETER ADJUSTMENT
model = SVC()
model.fit(X_train,y_train)
y_predict = model.predict(X_test)

# WITH PARAMETER ADJUSTMENT
param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001]}
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=0)
grid.fit(X_train,y_train)
grid_predict = grid.predict(X_test)
```

## Credits

[Christopher Ta](https://github.com/Krunk-Juice)
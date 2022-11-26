import pandas as pd


def pregunta_01():
    # Lea el archivo `insurance.csv` y asignelo al DataFrame `df`
    df = pd.read_csv("insurance.csv")


    # Asigne la columna `charges` a la variable `y`.
    y = df["charges"]

    # Asigne una copia del dataframe `df` a la variable `X`.
    X = df.copy()

    # Remueva la columna `charges` del DataFrame `X`.
    X.drop(["charges"], axis = 1, inplace = True)

    # Retorne `X` y `y`
    return X, y


def pregunta_02():
    from sklearn.model_selection import train_test_split
    X, y = pregunta_01()
    (X_train, X_test, y_train, y_test,) = train_test_split(
        X,
        y,
        test_size = 300,
        random_state = 12345,
    )

    # Retorne `X_train`, `X_test`, `y_train` y `y_test`
    return X_train, X_test, y_train, y_test


def pregunta_03():
    from sklearn.compose import make_column_selector
    from sklearn.compose import make_column_transformer
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import f_regression
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    pipeline = Pipeline(
        steps=[
            (
                "column_transfomer",
                make_column_transformer(
                    (
                        OneHotEncoder(),
                        make_column_selector(dtype_include = object),
                    ),
                    remainder = "passthrough",
                ),
            ),
            (
                "selectKBest",
                SelectKBest(score_func = f_regression),
            ),
            (
                "linearRegression",
                LinearRegression(),
            ),
        ],
    )
    X_train, _, y_train, _ = pregunta_02()
    param_grid = {
        "selectKBest__k": range(1, 11),
    }
    gridSearchCV = GridSearchCV(
        estimator = pipeline,
        param_grid = param_grid,
        cv = 5,
        scoring = "neg_mean_squared_error",
        refit = True,
        return_train_score = False,
    )
    gridSearchCV.fit(X_train, y_train)
    return gridSearchCV


def pregunta_04():
    from sklearn.metrics import mean_squared_error
    gridSearchCV = pregunta_03()
    X_train, X_test, y_train, y_test = pregunta_02()
    y_train_pred = gridSearchCV.predict(X_train)
    y_test_pred = gridSearchCV.predict(X_test)


    mse_train = mean_squared_error(
        y_train,
        y_train_pred,
    ).round(2)

    mse_test = mean_squared_error(
        y_test,
        y_test_pred,
    ).round(2)
    return mse_train, mse_test

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import os

def get_column_from_all_csvdata(all_data, column_name: str):
    """
    all_data: list of pandas dataframe.
    Get the column with column_name from all csv data. 
    flatten it. 
    """
    column = []
    for data in all_data:
        column.append(data[column_name].to_numpy())
    column = np.array(column).flatten()
    return column


def prepare_X_Y_from_csv(paths, X_columns, Y_columns):# fetched_columns need to be implemented later. 
    """
    Read data from csv files and prepare the data for training and testing.
    """
    # Read data from csv files
    all_data = []
    for path in paths:
        data = pd.read_csv(path)
        all_data.append(data)
        # print(data.columns)

    # X
    all_X = []
    for column_name in X_columns:
        all_X.append(get_column_from_all_csvdata(all_data, column_name))

    # Y
    all_Y = []
    for column_name in Y_columns:
        all_Y.append(get_column_from_all_csvdata(all_data, column_name))

    # Prepare the data
    X = np.array(all_X).T
    Y = np.array(all_Y).T

    # print some for debug
    print(X.shape, Y.shape)
    print(X[:10])
    print(Y[:10])
    return X, Y

def train_linear_regression(X, Y, save_folder=None):
    """
    Train the linear regression model.
    """

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

    # Step 4: Create and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Step 5: Evaluate the model
    y_pred = model.predict(X_test)

    # Print model performance
    print(f"Coefficients: {model.coef_}")
    print(f"Intercept: {model.intercept_}")
    print(f"testset Mean squared error (MSE): {np.mean((y_pred - y_test)**2)}")
    print(f"testset Coefficient of determination (R^2): {model.score(X_test, y_test)}")

    with open(f"{save_folder}/model.txt", "w") as f:
        f.write(f"Coefficients: {model.coef_}\n")
        f.write(f"Intercept: {model.intercept_}\n")
        f.write(f"testset Mean squared error (MSE): {np.mean((y_pred - y_test)**2)}\n")
        f.write(f"testset Coefficient of determination (R^2): {model.score(X_test, y_test)}\n")

    return model

def expe1(save_path):
    os.makedirs(save_path, exist_ok=True)

    paths = []
    for i in range(2, 6):
        # list all statistics_{}.csv
        all_in_folder = os.listdir(f"experiments/bench_train_rows{i}")
        all_in_folder.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
        for tmp in all_in_folder:
            if tmp.startswith("statistics_"):
                paths.append(f"experiments/bench_train_rows{i}/{tmp}")
    # print(paths)

    X_columns = [
        # "num_tiles",
        # "num_rendered",
        "global_ave_n_rendered_per_pix",
        "global_ave_n_considered_per_pix",
        "global_ave_n_contrib2loss_per_pix",
    ]
    Y_columns = [
        "70 render time",
        "b10 render time",
    ]
    X, Y = prepare_X_Y_from_csv(paths, X_columns, Y_columns)

    # save X and Y in the same csv with dataframe
    df = pd.DataFrame(X, columns=X_columns)
    df[Y_columns[0]] = Y[:, 0]
    df[Y_columns[1]] = Y[:, 1]    
    df.to_csv(f"{save_path}/X_Y.csv", index=False)
    # return 
    model = train_linear_regression(X, Y, save_folder=save_path)
    # save predict_Y in the same csv with dataframe
    predict_Y = model.predict(X)
    df["predict_70 render time"] = predict_Y[:, 0]
    df["predict_b10 render time"] = predict_Y[:, 1]
    df.to_csv(f"{save_path}/X_Y.csv", index=False)

def expe2(save_path):
    os.makedirs(save_path, exist_ok=True)

    paths = []
    for i in range(2, 6):
        # list all statistics_{}.csv
        all_in_folder = os.listdir(f"experiments/bench_train_rows{i}")
        all_in_folder.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
        for tmp in all_in_folder:
            if tmp.startswith("statistics_"):
                paths.append(f"experiments/bench_train_rows{i}/{tmp}")
    # print(paths)

    X_columns = [
        # "num_tiles",
        # "num_rendered",
        "global_ave_n_rendered_per_pix",
        "global_ave_n_considered_per_pix",
        "global_ave_n_contrib2loss_per_pix",
    ]
    Y_columns = [
        "70 render time",
    ]
    X, Y = prepare_X_Y_from_csv(paths, X_columns, Y_columns)

    # save X and Y in the same csv with dataframe
    df = pd.DataFrame(X, columns=X_columns)
    df[Y_columns[0]] = Y[:, 0]
    df.to_csv(f"{save_path}/X_Y.csv", index=False)
    # return 
    model = train_linear_regression(X, Y, save_folder=save_path)
    # save predict_Y in the same csv with dataframe
    predict_Y = model.predict(X)
    df["predict_70 render time"] = predict_Y[:, 0]
    df["difference"] = df["predict_70 render time"] - df["70 render time"]
    df["difference_ratio"] = abs(df["difference"]) / df["70 render time"]
    
    with open(f"{save_path}/model.txt", "a") as f:
        f.write(f"average abs difference: {abs(df['difference']).mean()}\n")
        f.write(f"average difference_ratio: {df['difference_ratio'].mean()}\n")
    df.to_csv(f"{save_path}/X_Y.csv", index=False)

def expe3(save_path):
    os.makedirs(save_path, exist_ok=True)

    paths = []
    for i in range(2, 6):
        # list all statistics_{}.csv
        all_in_folder = os.listdir(f"experiments/bench_train_rows{i}")
        all_in_folder.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
        for tmp in all_in_folder:
            if tmp.startswith("statistics_"):
                paths.append(f"experiments/bench_train_rows{i}/{tmp}")
    # print(paths)

    X_columns = [
        # "num_tiles",
        # "num_rendered",
        "global_ave_n_rendered_per_pix",
        "global_ave_n_considered_per_pix",
        "global_ave_n_contrib2loss_per_pix",
    ]
    Y_columns = [
        "b10 render time",
    ]
    X, Y = prepare_X_Y_from_csv(paths, X_columns, Y_columns)

    # save X and Y in the same csv with dataframe
    df = pd.DataFrame(X, columns=X_columns)
    df[Y_columns[0]] = Y[:, 0]
    df.to_csv(f"{save_path}/X_Y.csv", index=False)
    # return 
    model = train_linear_regression(X, Y, save_folder=save_path)
    # save predict_Y in the same csv with dataframe
    predict_Y = model.predict(X)
    df["predict_b10 render time"] = predict_Y[:, 0]
    df["difference"] = df["predict_b10 render time"] - df["b10 render time"]
    df["difference_ratio"] = abs(df["difference"]) / df["b10 render time"]
    with open(f"{save_path}/model.txt", "a") as f:
        f.write(f"average abs difference: {abs(df['difference']).mean()}\n")
        f.write(f"average difference_ratio: {df['difference_ratio'].mean()}\n")
    df.to_csv(f"{save_path}/X_Y.csv", index=False)

if __name__ == "__main__":
    # expe1("experiments/sklearn/expe1")
    expe2("experiments/sklearn/expe2")
    expe3("experiments/sklearn/expe3")


    pass
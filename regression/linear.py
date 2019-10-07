# Imports

import pandas
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score


def extract_row(y_in: pandas.DataFrame, x_in: pandas.DataFrame, idx: int):
    y_out = y_in.loc[idx]
    x_out = x_in.loc[idx]
    new_y_in = y_in.drop(y_in.index[idx])
    new_x_in = x_in.drop(x_in.index[idx])
    return y_out, x_out, new_y_in, new_x_in


def linear_prediction(y_real, x_pred, Y_train, X_train):
    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(X_train, Y_train)

    # Make predictions using the testing set
    y_sal_pred = regr.predict([x_pred])
    return y_real - y_sal_pred


def linear_prediction_for_idx(y_df, x_df, idx):
    y_real, x_pred, Y_train, X_train = extract_row(y_in=y_df, x_in=x_df, idx=idx)
    return linear_prediction(y_real, x_pred, Y_train, X_train)


if __name__ == "__main__":
    data = pandas.read_csv("../data/test.csv")

    print(data)

    y_sal = data['sal']
    x_sal = data.loc[:, 'nr':]

    out_data = {'deltas': []}
    for idx in range(len(data.index)):
        y_delta = linear_prediction_for_idx(y_df=y_sal, x_df=x_sal, idx=idx)
        out_data['deltas'].append(y_delta[0])

        #TODO : https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression

    deltas = pandas.DataFrame(out_data)

    print(deltas)
    print(deltas.mean(axis = 0))

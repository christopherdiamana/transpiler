import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

def list_to_string(list_to_convert):
    list_converted = ""
    n_elt = len(list_to_convert)

    for idx, value in enumerate(list_to_convert):
        list_converted += str(value)
        if idx < n_elt - 1:
            list_converted += ", "

    return list_converted



def train_tumors_model():
    df = pd.read_csv('tumors.csv')

    scaler = MinMaxScaler()

    x_train = scaler.fit_transform(df[["size", "p53_concentration"]])
    y_train = df['is_cancerous']

    model = LinearRegression()

    model.fit(x_train, y_train)

    joblib.dump(scaler, "scaler_classification.joblib")
    joblib.dump(model, "linear_regression.joblib")


def produce_linear_prediction_c_code(model):
    bias = model.intercept_
    coefs = model.coef_

    n_thetas = len(coefs)

    thetas = "{"
    thetas += str(bias) + ", "
    thetas += list_to_string(coefs)
    thetas += "}"

    n_thetas += 1 #Add Bias


    prediction_code = "float thetas[{}] = {}".format(n_thetas, thetas)

    code = """

    {};

    float prediction(float* features, int n_features)
    {{
        float result = thetas[0];

        for (int i = 0; i < n_features; i++)
        {{
            result += features[i] * thetas[i + 1];
        }}

        return result;
    }}""".format(prediction_code)

    return code



def produce_main_c_code(data):
    n_data = len(data)
    string_data ="{" + list_to_string(data) + "}"

    code = """

    int main()
    {{
        float prediction_result = 0;
        float data_to_predict[{array_size}] = {array};
        prediction_result = prediction(data_to_predict, {array_size});

        printf("Prediction: %f", prediction_result);

        return 0;
    }}""".format(array=string_data, array_size=n_data)

    return code


def main():

    train_tumors_model()

    data_to_predict = [[0.0199419, -0.0041917]]

    model = joblib.load("linear_regression.joblib")
    scaler = joblib.load("scaler_classification.joblib")

    data_to_predict_normalized = scaler.transform(data_to_predict)
    data_to_predict_normalized = data_to_predict_normalized[0]

    c_code = "#include <stdio.h>\n"
    c_code += produce_linear_prediction_c_code(model)
    c_code += produce_main_c_code(data_to_predict_normalized)

    with open("linear_regression.c", "w+") as f:
        f.write(c_code)
    print("The file linear_regression.c was generated")

    print("""\nTo compile, run the following compilation command:
        gcc linear_regression.c -o linear_regression""")


if __name__ == "__main__":
    main()

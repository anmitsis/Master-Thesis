using Pkg
Pkg.add(["CSV", "DataFrames", "XGBoost", "Dash"])
Pkg.add("DashHtmlComponents")
using CSV
using DataFrames
using XGBoost
using Dash, DashCoreComponents, DashHtmlComponents

# Load data
X_train = CSV.read("X_train.csv", DataFrame)
y_train = CSV.read("y_train.csv", DataFrame)
X_test = CSV.read("X_test.csv", DataFrame)
y_test = CSV.read("y_test.csv", DataFrame)


# Convert y_train and y_test to vectors
y_train = convert(Vector{Float32}, y_train[:, 1])
y_test = convert(Vector{Float32}, y_test[:, 1])

# Convert DataFrames to matrices
X_train_matrix = Matrix(X_train)
X_test_matrix = Matrix(X_test)

# Training
dtrain = DMatrix(X_train_matrix, label=y_train)
param = Dict("max_depth" => 3, "eta" => 0.1, "objective" => "binary:logistic", "eval_metric" => "logloss")
num_round = 100
bst = xgboost(dtrain, num_round=num_round, params=param)


# Prediction function
function predict_diabetes(model, input_data)
    dtest = DMatrix(input_data)
    preds = XGBoost.predict(model, dtest)
    return preds[1] > 0.5  # Convert probabilities to binary output
end




app = dash()

app.layout = html_div() do
    html_h1("Diabetes Predictor"),
    dcc_input(id="input-age", type="number", placeholder="Age"),
    dcc_input(id="input-chol", type="number", placeholder="Cholesterol Level"),
    dcc_input(id="input-glucose", type="number", placeholder="Glucose Level"),
    dcc_input(id="input-time_ppn", type="number", placeholder="Time PPN"),
    dcc_input(id="input-waist", type="number", placeholder="Waist Size"),
    dcc_input(id="input-weight", type="number", placeholder="Weight"),
    dcc_input(id="input-height", type="number", placeholder="Height"),
    html_button("Predict", id="predict-button", n_clicks=0),
    html_div(id="output-prediction")
end

callback!(app, Output("output-prediction", "children"), 
          Input("predict-button", "n_clicks"), 
          State("input-age", "value"), 
          State("input-chol", "value"), 
          State("input-glucose", "value"), 
          State("input-time_ppn", "value"), 
          State("input-waist", "value"), 
          State("input-weight", "value"), 
          State("input-height", "value")) do n_clicks, age, chol, glucose, time_ppn, waist, weight, height
    if !ismissing(age) && !ismissing(chol) && !ismissing(glucose) && !ismissing(time_ppn) && !ismissing(waist) && !ismissing(weight) && !ismissing(height)
        input_data = [age, chol, glucose, time_ppn, waist, weight, height]
        input_data = reshape(input_data, 1, length(input_data))  # Reshape to 2D matrix
        prediction = predict_diabetes(bst, input_data)
        return prediction ? "You may have diabetes." : "You are unlikely to have diabetes."
    else
        return "Please enter all the values."
    end
end


run_server(app, "0.0.0.0", 8080)

#http://localhost:8080/
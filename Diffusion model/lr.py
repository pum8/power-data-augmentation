import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor

data = pd.read_excel("perf_events_pwr.xlsx")
synth_data = pd.read_csv("synthetic_data_ddpm.csv")

column_name = ['occupancy', 'ILP',
       'intensity', 'reuse_ratio', 'ld_coalesce', 'L2_hit_rate',
       'L1_hit_rate', 'branch_eff',  'pwr_avg']

real_data = data[column_name]
X = real_data.drop(columns=['pwr_avg']) 
y = real_data['pwr_avg'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,shuffle=True, random_state=42)

X_train_with_synth = pd.concat([X_train, synth_data.drop(columns=['pwr_avg'])], ignore_index=True)
y_train_with_synth = pd.concat([y_train, synth_data['pwr_avg']], ignore_index=True)

def ml_models(X_train,y_train):

       model = LinearRegression()

       model.fit(X_train, y_train)

       y_pred = model.predict(X_test)
       mse = mean_squared_error(y_test, y_pred)
       print("Mean Squared Error for Linear Regression:", mse)

       r2_linear = r2_score(y_test, y_pred)
       print("R-squared for Linear Regression:", r2_linear)

       DTmodel = DecisionTreeRegressor()
       DTmodel.fit(X_train, y_train)
       y_pred = DTmodel.predict(X_test)
       DTmse = mean_squared_error(y_test, y_pred)
       print("DT Mean Squared Error:", DTmse)
       y_pred_dt = DTmodel.predict(X_test)
       r2_dt = r2_score(y_test, y_pred_dt)
       print("R-squared for Decision Tree Regression:", r2_dt)

       RFmodel = RandomForestRegressor(n_estimators=100, random_state=42)
       RFmodel.fit(X_train, y_train)
       y_pred_rf = RFmodel.predict(X_test)
       RFmse = mean_squared_error(y_test, y_pred_rf)
       print("Mean Squared Error (Random Forest Regression):", RFmse)
       r2_rf = r2_score(y_test, y_pred_rf)
       print("R-squared for Random Forest Regression:", r2_rf)

       GBmodel = GradientBoostingRegressor(n_estimators=100, random_state=42)
       GBmodel.fit(X_train, y_train)
       y_pred_gb = GBmodel.predict(X_test)
       GBmse = mean_squared_error(y_test, y_pred_gb)
       print("Mean Squared Error (Gradient Boosting Regression):", GBmse)
       r2_gb = r2_score(y_test, y_pred_gb)
       print("R-squared for Gradient Boosting Regression:", r2_gb)

       SVRmodel = SVR()
       SVRmodel.fit(X_train, y_train)
       y_pred_svr = SVRmodel.predict(X_test)
       SVRmse = mean_squared_error(y_test, y_pred_svr)
       print("Mean Squared Error (Support Vector Regression):", SVRmse)
       r2_svr = r2_score(y_test, y_pred_svr)
       print("R-squared for Support Vector Regression:", r2_svr)

def mlp_models(X_train, y_train):
    mlp = MLPRegressor(hidden_layer_sizes=(100, 100, 50), 
                       activation='relu', 
                       solver='adam', 
                       alpha=0.0001, 
                       batch_size=64, 
                       learning_rate='adaptive', 
                       max_iter=1000, 
                       random_state=42)
    mlp.fit(X_train, y_train)

    y_pred_mlp = mlp.predict(X_test)
    mse_mlp = mean_squared_error(y_test, y_pred_mlp)
    r2_mlp = r2_score(y_test, y_pred_mlp)
    print("Mean Squared Error MLP Regression:", mse_mlp)
    print("R-squared for MLP Regression:", r2_mlp)



print("Model on Real data:")
ml_models(X_train,y_train)
print("Model on Synthtic data:")
ml_models(X_train_with_synth,y_train_with_synth)

print("MLP Model on Real Data:")
mlp_models(X_train, y_train)
print("MLP Model on Synthetic Data:")
mlp_models(X_train_with_synth, y_train_with_synth)
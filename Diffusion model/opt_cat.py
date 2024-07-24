import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import optuna
from optuna.samplers import TPESampler
from sklearn.utils import shuffle

real_data_cat_path = 'real_data_cat.csv'
real_data_cat = pd.read_csv(real_data_cat_path)

X = real_data_cat.drop(columns=['pwr_avg_category'])
y = real_data_cat['pwr_avg_category']

column_name = ['occupancy', 'ILP',
       'intensity', 'reuse_ratio', 'ld_coalesce', 'L2_hit_rate',
       'L1_hit_rate', 'branch_eff']
scaler = StandardScaler()
X = scaler.fit_transform(X)
X = pd.DataFrame(X, columns=column_name)



synth_data_cat = pd.read_csv("synthetic_data_ddpm_cat.csv")

X_synth = synth_data_cat.drop(columns=['pwr_avg_category'])
X_synth = scaler.transform(X_synth)
X_synth = pd.DataFrame(X_synth, columns=column_name)

le = LabelEncoder()
y = le.fit_transform(y)
y = pd.Series(y)
y_synth = le.transform(synth_data_cat['pwr_avg_category'])
y_synth = pd.Series(y_synth)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

X_train_with_synth = pd.concat([X_train, X_synth], ignore_index=True)
y_train_with_synth = pd.concat([y_train, y_synth], ignore_index=True)

X_train_with_synth, y_train_with_synth = shuffle(X_train_with_synth, y_train_with_synth, random_state=42)
"""
def objective(trial):
    n_layers = trial.suggest_int('n_layers', 1, 5)
    hidden_layer_sizes = tuple(trial.suggest_int(f'n_units_l{i}', 50, 200) for i in range(n_layers))
    activation = trial.suggest_categorical('activation', ['identity', 'logistic', 'tanh', 'relu'])
    solver = trial.suggest_categorical('solver', ['lbfgs', 'sgd', 'adam'])
    alpha = trial.suggest_loguniform('alpha', 1e-5, 1e-1)
    learning_rate = trial.suggest_categorical('learning_rate', ['constant', 'invscaling', 'adaptive'])
    batch_size = trial.suggest_int('batch_size', 32, 256)
    learning_rate_init = trial.suggest_loguniform('learning_rate_init', 1e-5, 1e-1)
    n_iter_no_change = trial.suggest_int('n_iter_no_change', 5, 20)

    mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
                        activation=activation,
                        solver=solver,
                        alpha=alpha,
                        learning_rate=learning_rate,
                        batch_size=batch_size,
                        learning_rate_init=learning_rate_init,
                        n_iter_no_change=n_iter_no_change,
                        early_stopping=True,
                        max_iter=500,
                        random_state=42)

    scores = cross_val_score(mlp, X_train, y_train, n_jobs=-1, cv=3)
    accuracy = scores.mean()
    
    return accuracy

study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
study.optimize(objective, n_trials=50)
"""
#print("Best hyperparameters:", study.best_params)
#best_params ={'n_layers': 4, 'n_units_l0': 144, 'n_units_l1': 64, 'n_units_l2': 160, 'n_units_l3': 70, 'activation': 'relu', 'solver': 'adam', 
 #'alpha': 0.01851818470299336, 'learning_rate': 'constant', 'batch_size': 254, 'learning_rate_init': 0.014988907054177008, 'n_iter_no_change': 17}
best_params = {'n_layers': 2, 'n_units_l0': 91, 'n_units_l1': 189, 'activation': 'logistic', 'solver': 'lbfgs', 'alpha': 0.06148454913654118, 'learning_rate': 'adaptive', 'batch_size': 156, 'learning_rate_init': 0.00018059456149989743, 'n_iter_no_change': 9}#study.best_params
hidden_layer_sizes = tuple(best_params[f'n_units_l{i}'] for i in range(best_params['n_layers']))
mlp_best = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
                         activation=best_params['activation'],
                         solver=best_params['solver'],
                         alpha=best_params['alpha'],
                         learning_rate=best_params['learning_rate'],
                         batch_size=best_params['batch_size'],
                         learning_rate_init=best_params['learning_rate_init'],
                         
                         n_iter_no_change=best_params['n_iter_no_change'],
                         max_iter=500,
                         early_stopping=True,
                         random_state=42)

mlp_best.fit(X_train_with_synth, y_train_with_synth)
y_pred_mlp_best = mlp_best.predict(X_test)
accuracy_mlp_best = accuracy_score(y_test, y_pred_mlp_best)

print(X_train_with_synth.shape)
print("Accuracy for MLP Classifier with best hyperparameters:", accuracy_mlp_best)
print(classification_report(y_test, y_pred_mlp_best, target_names=le.classes_))


#baseline
#Best hyperparameters: {'n_layers': 4, 'n_units_l0': 144, 'n_units_l1': 64, 'n_units_l2': 160, 'n_units_l3': 70, 'activation': 'relu', 'solver': 'adam', 'alpha': 0.01851818Best hyperparameters: {'n_layers': 4, 'n_units_l0': 144, 'n_units_l1': 64, 'n_units_l2': 160, 'n_units_l3': 70, 'activation': 'relu', 'solver': 'adam', 
# 'alpha': 0.01851818470299336, 'learning_rate': 'constant', 'batch_size': 254, 'learning_rate_init': 0.014988907054177008, 'n_iter_no_change': 17}

#whole data
#Best hyperparameters: {'n_layers': 2, 'n_units_l0': 91, 'n_units_l1': 189, 'activation': 'logistic', 'solver': 'lbfgs', 'alpha': 0.06148454913654118, 'learning_rate': 'adaptive', 'batch_size': 156, 'learning_rate_init': 0.00018059456149989743, 'n_iter_no_change': 9}
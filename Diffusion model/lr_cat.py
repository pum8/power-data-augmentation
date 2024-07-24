
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler, LabelEncoder


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


X_train_with_synth =  pd.concat([X_train, X_synth], ignore_index=True)
y_train_with_synth = pd.concat([y_train, y_synth], ignore_index=True)

X_train_with_synth, y_train_with_synth = shuffle(X_train_with_synth, y_train_with_synth, random_state=42)



def mlp_models_baseline(X_train, y_train, X_test, y_test):
    mlp = MLPClassifier(hidden_layer_sizes=(144, 64, 160,70),
                        activation='relu',
                        solver='adam',
                        alpha=0.01851818470299336,
                        batch_size=254,
                        learning_rate_init=0.014988907054177008,
                        max_iter=300,
                        early_stopping=True,
                        random_state=42)
    mlp.fit(X_train, y_train)
    y_pred_mlp = mlp.predict(X_test)
    accuracy_mlp = accuracy_score(y_test, y_pred_mlp)
    print("Accuracy for MLP Classifier:", accuracy_mlp)
    
def mlp_models_aug(X_train, y_train, X_test, y_test):
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

    mlp_best.fit(X_train, y_train)
    y_pred_mlp_best = mlp_best.predict(X_test)
    accuracy_mlp_best = accuracy_score(y_test, y_pred_mlp_best)
    print("Accuracy for MLP Classifier whole:", accuracy_mlp_best)
print("MLP Model on Real Data:")
mlp_models_baseline(X_train, y_train, X_test, y_test)

print("MLP Model on Synthetic Data:")
mlp_models_aug(X_train_with_synth, y_train_with_synth, X_test, y_test)
#Best hyperparameters: {'n_layers': 2, 'n_units_l0': 91, 'n_units_l1': 189, 'activation': 'logistic', 'solver': 'lbfgs', 'alpha': 0.06148454913654118, 'learning_rate': 'adaptive', 'batch_size': 156, 'learning_rate_init': 0.00018059456149989743, 'n_iter_no_change': 9}
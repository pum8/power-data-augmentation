import pandas as pd
from sklearn.model_selection import train_test_split
from synthcity.plugins import Plugins
from synthcity.metrics.eval_statistical import KolmogorovSmirnovTest
from synthcity.plugins.core.dataloader import GenericDataLoader
import optuna
import pickle

data = pd.read_excel("perf_events_pwr.xlsx")

column_name = ['occupancy', 'ILP', 'intensity', 'reuse_ratio', 'ld_coalesce', 
               'st_coalesce', 'L2_hit_rate', 'L1_hit_rate', 'branch_eff', 'pwr_avg']
column_name_wo_target = ['occupancy', 'ILP', 'intensity', 'reuse_ratio', 
                         'ld_coalesce', 'st_coalesce', 'L2_hit_rate', 
                         'L1_hit_rate', 'branch_eff']

real_data = data[column_name].copy()

def objective(trial):
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_int('batch_size', 32, 128, step=32)
    dim_embed = trial.suggest_int('dim_embed', 128, 512, step=128)
    
    syn_model = Plugins().get("ddpm", lr=lr, batch_size=batch_size, dim_embed=dim_embed)
    syn_model.fit(real_data)
    syn_data = syn_model.generate(count=600)
    syn_df = syn_data.dataframe()
    
    real_data_loader = GenericDataLoader(real_data)
    syn_data_loader = GenericDataLoader(syn_df)
    ks_test = KolmogorovSmirnovTest()
    result = ks_test.evaluate(real_data_loader, syn_data_loader)
    
    return result['marginal']

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

best_trial = study.best_trial
print(f"Best trial: {best_trial.params}")

best_params = best_trial.params
syn_model = Plugins().get("ddpm", lr=best_params['lr'], batch_size=best_params['batch_size'], dim_embed=best_params['dim_embed'])
syn_model.fit(real_data)
syn_data = syn_model.generate(count=600)
syn_df = syn_data.dataframe()
syn_df.to_csv("synthetic_data_ddpm_optimized.csv", index=False)

with open('ddpm_model_optimized.pkl', 'wb') as f:
    pickle.dump(syn_model, f)

real_data_loader = GenericDataLoader(real_data)
syn_data_loader = GenericDataLoader(syn_df)
result = KolmogorovSmirnovTest().evaluate(real_data_loader, syn_data_loader)
print("Kolmogorov-Smirnov test results for the best model:")
print(result)
#Best trial: {'lr': 0.0004986080281772112, 'batch_size': 64, 'dim_embed': 384}
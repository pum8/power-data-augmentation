import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df_original = pd.read_excel('perf_events_pwr.xlsx')

column_name = ['occupancy', 'ILP', 'intensity', 'reuse_ratio', 'ld_coalesce',
               'L2_hit_rate', 'L1_hit_rate', 'branch_eff', 'pwr_avg']

df_selected_original = df_original[column_name].copy()
scaler_original = StandardScaler()
df_selected_scaled_original = scaler_original.fit_transform(df_selected_original)

pca_original = PCA(n_components=1)
principal_components_original = pca_original.fit_transform(df_selected_scaled_original)

df_synthetic = pd.read_csv('synthetic_data.csv')

df_selected_synthetic = df_synthetic[column_name].copy()

df_selected_scaled_synthetic = scaler_original.transform(df_selected_synthetic)

principal_components_synthetic = pca_original.transform(df_selected_scaled_synthetic)

df_pca_original = pd.DataFrame()
df_pca_synth = pd.DataFrame()

df_pca_original['PCA'] = pd.Series(principal_components_original.flatten())
df_pca_synth['PCA'] = pd.Series(principal_components_synthetic.flatten())

df_pca_original['PCA'] = pd.Series(df_pca_original['PCA'])
df_pca_synth['PCA'] = pd.Series(df_pca_synth['PCA'])

df_pca_synth.to_csv('pca_synth_results.csv', index=False)
df_pca_original.to_csv('pca_original_results.csv', index=False)
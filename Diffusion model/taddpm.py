import pandas as pd
from sklearn.model_selection import train_test_split
from synthcity.plugins import Plugins
from synthcity.metrics.eval_statistical import KolmogorovSmirnovTest
from synthcity.plugins.core.dataloader import GenericDataLoader
import pickle

data = pd.read_excel("perf_events_pwr.xlsx")

#['graph_name', 'pwr_peak', 'pwr_avg', 'TEPS', 'occupancy', 'ILP',   'intensity', 'reuse_ratio', 'ld_coalesce',
# 'st_coalesce', 'L2_hit_rate',   'L1_hit_rate', 'branch_eff', 'pred_eff', 'performance/watt']
column_name = ['occupancy', 'ILP',
       'intensity', 'reuse_ratio', 'ld_coalesce', 'L2_hit_rate',
       'L1_hit_rate', 'branch_eff',  'pwr_avg']
column_name_wo_target = ['occupancy', 'ILP',
       'intensity', 'reuse_ratio', 'ld_coalesce', 'L2_hit_rate',
       'L1_hit_rate', 'branch_eff']

real_data = data[column_name].copy()

syn_model = Plugins().get("ddpm",lr= 0.0004986080281772112, batch_size=64, dim_embed= 384)
syn_model.fit(real_data)
syn_data= syn_model.generate(count=5000)
print(type(syn_data))
syn_df=syn_data.dataframe()
syn_df.to_csv("synthetic_data_ddpm.csv",index=False)
real_data_loader = GenericDataLoader(real_data)
syn_data_loader = GenericDataLoader(syn_df)
ks_test = KolmogorovSmirnovTest()
result = ks_test.evaluate(real_data_loader, syn_data_loader)
print("Kolmogorov-Smirnov test results:")
print(result)


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

pwr_avg_quantiles = real_data['pwr_avg'].quantile([0.25, 0.5, 0.75])
print(pwr_avg_quantiles)
low_boundary = pwr_avg_quantiles[0.25]
high_boundary = pwr_avg_quantiles[0.75]

# Function to categorize pwr_avg
def categorize_pwr_avg(value):
    if value <= low_boundary:
        return 'low'
    elif value <= high_boundary:
        return 'avg'
    else:
        return 'high'
    
real_data['pwr_avg_category'] = real_data['pwr_avg'].apply(categorize_pwr_avg)

print(real_data[['pwr_avg','pwr_avg_category']])

column_name_new = ['occupancy', 'ILP',
       'intensity', 'reuse_ratio', 'ld_coalesce', 'L2_hit_rate',
       'L1_hit_rate', 'branch_eff',  'pwr_avg_category']


real_data = real_data[column_name_new].copy()
real_data.to_csv("real_data_cat.csv",index=False)
syn_model = Plugins().get("ddpm",lr= 0.0004986080281772112, batch_size=64, dim_embed= 384,is_classification=True)
syn_model.fit(real_data)
syn_data= syn_model.generate(count=5000)
print(type(syn_data))
syn_df=syn_data.dataframe()
syn_df.to_csv("synthetic_data_ddpm_cat.csv",index=False)
real_data_loader = GenericDataLoader(real_data)
syn_data_loader = GenericDataLoader(syn_df)
ks_test = KolmogorovSmirnovTest()
result = ks_test.evaluate(real_data_loader, syn_data_loader)
print("Kolmogorov-Smirnov test results:")
print(result)
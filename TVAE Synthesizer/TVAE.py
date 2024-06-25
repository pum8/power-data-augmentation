import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.single_table import TVAESynthesizer
from sdv.evaluation.single_table import run_diagnostic
from sdv.evaluation.single_table import evaluate_quality
data = pd.read_excel("perf_events_pwr.xlsx")

#['graph_name', 'pwr_peak', 'pwr_avg', 'TEPS', 'occupancy', 'ILP',   'intensity', 'reuse_ratio', 'ld_coalesce', 'st_coalesce', 'L2_hit_rate',   'L1_hit_rate', 'branch_eff', 'pred_eff', 'performance/watt']
column_name = ['occupancy', 'ILP',
       'intensity', 'reuse_ratio', 'ld_coalesce', 'L2_hit_rate',
       'L1_hit_rate', 'branch_eff',  'pwr_avg']
column_name_wo_target = ['occupancy', 'ILP',
       'intensity', 'reuse_ratio', 'ld_coalesce', 'L2_hit_rate',
       'L1_hit_rate', 'branch_eff']

real_data = data[column_name].copy()

print(real_data)
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(real_data)
python_dict = metadata.to_dict()

#print(python_dict)

synthesizer = TVAESynthesizer(
    metadata,
    batch_size=200,#200
    enforce_min_max_values=True,
    compress_dims=(512,512),#512
    decompress_dims=(256,256),#256
    enforce_rounding=False,
    cuda=True,
    l2scale=1e-3,#3
   # loss_factor=3,
    epochs=500 )#500
synthesizer.fit(real_data)
print(synthesizer.get_parameters())

synthetic_data = synthesizer.sample(num_rows=300)
print(synthetic_data.head())

synthesizer.save(
    filepath='model_040324.pkl'
)

synthetic_data.to_csv('synthetic_data.csv', index=False)
print(synthetic_data.head())

diagnostic = run_diagnostic(
    real_data=real_data,
    synthetic_data=synthetic_data,
    metadata=metadata
)
print("Diagnostic Report: ",diagnostic)
quality_report = evaluate_quality(
    real_data,
    synthetic_data,
    metadata)
print("Evaluation Report: ",quality_report)

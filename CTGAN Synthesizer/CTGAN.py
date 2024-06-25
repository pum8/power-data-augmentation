import pandas as pd
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
from sdv.evaluation.single_table import run_diagnostic
from sdv.evaluation.single_table import evaluate_quality

data = pd.read_excel("perf_events_pwr.xlsx")
#['graph_name', 'pwr_peak', 'pwr_avg', 'TEPS', 'occupancy', 'ILP',   'intensity', 'reuse_ratio', 'ld_coalesce', 'st_coalesce', 'L2_hit_rate',   'L1_hit_rate', 'branch_eff', 'pred_eff', 'performance/watt']
column_name = ['occupancy', 'ILP',
       'intensity', 'reuse_ratio', 'ld_coalesce', 'L2_hit_rate',
       'L1_hit_rate', 'branch_eff',  'pwr_avg']
real_data = data[column_name]

metadata = SingleTableMetadata()
metadata.detect_from_dataframe(real_data)
python_dict = metadata.to_dict()

synthesizer = CTGANSynthesizer(metadata,verbose=True,
                               batch_size=64,
                               discriminator_lr=2e-6,
                               generator_lr=2e-6,
                               epochs=500,
                               discriminator_dim=(512,512),
                               generator_dim=(128,128),
                               embedding_dim = 256,
                               cuda=True,
                               pac=8,
                               )

synthesizer.fit(real_data)

synthetic_data = synthesizer.sample(num_rows=5000)
print(synthetic_data.head())

synthesizer.save(filepath='model_CTGAN.pkl')

synthetic_data.to_csv('synthetic_data.csv', index=False)
#print(synthetic_data.head())

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
print("Evaluation Report: ",quality_report.get_details)
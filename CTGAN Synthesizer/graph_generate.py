
from sdv.single_table import CTGANSynthesizer
import pandas as pd
from sdv.metadata import SingleTableMetadata

data = pd.read_excel("perf_events_pwr.xlsx")
column_name = ['occupancy', 'ILP',
       'intensity', 'reuse_ratio', 'ld_coalesce', 'st_coalesce', 'L2_hit_rate',
       'L1_hit_rate', 'branch_eff',  'pwr_avg']
column_name_wo_target = ['occupancy', 'ILP',
       'intensity', 'reuse_ratio', 'ld_coalesce', 'st_coalesce', 'L2_hit_rate',
       'L1_hit_rate', 'branch_eff']
real_data = data[column_name].copy()

synthesizer = CTGANSynthesizer.load(
    filepath='model_CTGAN.pkl'
)
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(real_data)
python_dict = metadata.to_dict()

synthetic_data =synthesizer.sample(num_rows=int(120))

synthetic_data.to_csv('synthetic_data.csv', index=False)

from sdv.evaluation.single_table import run_diagnostic
from sdv.evaluation.single_table import evaluate_quality

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

print("Evaluation Report: ",quality_report.get_details( property_name= 'Column Pair Trends'))
print("Report",quality_report.get_details(property_name='Column Shapes'))

fig = quality_report.get_visualization(property_name='Column Shapes')
fig.write_image(f"ctganfigure/KScomplement.jpg")

from sdv.evaluation.single_table import get_column_plot,get_column_pair_plot

def fig_generator(feature):

    fig = get_column_plot(
        real_data=real_data,
        synthetic_data=synthetic_data,
        metadata=metadata,
        column_name=feature
    )
    if feature!='pwr_avg':
        fig2 = get_column_pair_plot(
            real_data=real_data,
            synthetic_data=synthetic_data,
            metadata=metadata,
            column_names=[feature,'pwr_avg'],
        )
        fig2.write_image(f"ctganfigure/{feature} VS PWR_AVG.jpg")
    fig.write_image(f"ctganfigure/{feature}.jpg")
    

for f in ['occupancy', 'ILP',
       'intensity', 'reuse_ratio', 'ld_coalesce', 'L2_hit_rate',
       'L1_hit_rate', 'branch_eff',  'pwr_avg']:
    print(f)
   # fig_generator(f)



import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
corr = real_data.corr()
print(corr)
plt.figure(figsize=(10, 8))
sns.heatmap(corr, cmap='coolwarm')
plt.title('Real Data Correlation Heatmap')
plt.savefig('ctganfigure/Real Data Correlation Heatmap.jpg')
corr = synthetic_data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, cmap='coolwarm')
plt.title('CTGAN Synthetic Data Correlation Heatmap')
plt.savefig('ctganfigure/CTGAN Synthetic Data Correlation Heatmap.jpg')


print("Done")
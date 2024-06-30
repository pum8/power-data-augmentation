
from sdv.single_table import CTGANSynthesizer
import pandas as pd
from sdv.metadata import SingleTableMetadata
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sdv.evaluation.single_table import run_diagnostic
from sdv.evaluation.single_table import evaluate_quality
from sdv.evaluation.single_table import get_column_plot,get_column_pair_plot
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_excel("perf_events_pwr.xlsx")
column_name = ['occupancy', 'ILP',
       'intensity', 'reuse_ratio', 'ld_coalesce', 'L2_hit_rate',
       'L1_hit_rate', 'branch_eff',  'pwr_avg']
column_name_wo_target = ['occupancy', 'ILP',
       'intensity', 'reuse_ratio', 'ld_coalesce', 'L2_hit_rate',
       'L1_hit_rate', 'branch_eff']

column_name_map = {
    'occupancy': 'Occupancy',
    'ILP': 'Instruction-Level Parallelism',
    'intensity': 'Intensity',
    'reuse_ratio': 'Reuse Ratio',
    'ld_coalesce': 'Load Coalescence',
    'L2_hit_rate': 'L2 Cache Hit Rate',
    'L1_hit_rate': 'L1 Cache Hit Rate',
    'branch_eff': 'Branch Efficiency',
    'pwr_avg': 'Average Power'
}


real_data = data[column_name].copy()

synthesizer = CTGANSynthesizer.load(
    filepath='model_040324.pkl'
)
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(real_data)
python_dict = metadata.to_dict()

synthetic_data =synthesizer.sample(num_rows=int(5000))#120

synthetic_data.to_csv('synthetic_data.csv', index=False)

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
fig.write_image(f"figure/KScomplement.jpg")

def fig_generator(feature):

    fig = get_column_plot(
        real_data=real_data,
        synthetic_data=synthetic_data,
        metadata=metadata,
        column_name=feature
    )
    fig.update_layout(
        title=f"{column_name_map[feature]}",
        showlegend=True,
        font=dict(size=25),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5
        )
    )   
    fig.write_image(f"figure/{feature}.jpg")
    """
    if feature!='pwr_avg':
        fig2 = get_column_pair_plot(
            real_data=real_data,
            synthetic_data=synthetic_data,
            metadata=metadata,
            column_names=[feature,'pwr_avg'],
        )
        fig2.write_image(f"ctganfigure/{feature} VS PWR_AVG.jpg")
    fig.write_image(f"ctganfigure/{feature}.jpg")
    """

for f in ['occupancy', 'ILP',
       'intensity', 'reuse_ratio', 'ld_coalesce', 'L2_hit_rate',
       'L1_hit_rate', 'branch_eff',  'pwr_avg']:
    print(f)
    fig_generator(f)


sns.set_context("talk", font_scale=1.4)




corr = synthetic_data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, cmap='coolwarm',annot=False, cbar=False)
plt.title('TVAE synthetic Data Correlation Heatmap', fontsize=25)
plt.tight_layout(rect=[0, 0, 1, 0.95])  
plt.savefig('figure/TVAE synthetic Data Correlation Heatmap.jpg')
plt.close()

print("Done")



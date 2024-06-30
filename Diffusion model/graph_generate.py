
from sdv.single_table import CTGANSynthesizer
import pandas as pd
from sdv.metadata import SingleTableMetadata
import pandas as pd
#import seaborn as sns
import matplotlib.pyplot as plt
from sdv.evaluation.single_table import run_diagnostic
from sdv.evaluation.single_table import evaluate_quality
from sdv.evaluation.single_table import get_column_plot,get_column_pair_plot
from synthcity.plugins import Plugins
from synthcity.metrics.eval_statistical import KolmogorovSmirnovTest
from synthcity.plugins.core.dataloader import GenericDataLoader
import seaborn as sns


data = pd.read_excel("perf_events_pwr.xlsx")
column_name = ['occupancy', 'ILP',
       'intensity', 'reuse_ratio', 'ld_coalesce' , 'L2_hit_rate',
       'L1_hit_rate', 'branch_eff',  'pwr_avg']
column_name_wo_target = ['occupancy', 'ILP',
       'intensity', 'reuse_ratio', 'ld_coalesce','L2_hit_rate',
       'L1_hit_rate', 'branch_eff']
real_data = data[column_name].copy()


metadata = SingleTableMetadata()
metadata.detect_from_dataframe(real_data)
python_dict = metadata.to_dict()

synthetic_data = pd.read_csv("synthetic_data_ddpm.csv")


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

import plotly.express as px

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


fig = quality_report.get_visualization(property_name='Column Shapes')

fig.update_traces(marker=dict(color='blue'))  
fig.update_layout(
    title={
        'text': "Kolmogorov-Smirnov score",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'
    },
    plot_bgcolor='white', 
    paper_bgcolor='white', 
    font=dict(color='black', size=25),  
    showlegend=False  
)

fig.write_image(f"figure/TabDDPM KScomplement.jpg")

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

for f in ['occupancy', 'ILP',
       'intensity', 'reuse_ratio', 'ld_coalesce', 'L2_hit_rate',
       'L1_hit_rate', 'branch_eff',  'pwr_avg']:
    print(f)
    fig_generator(f)


sns.set_context("talk", font_scale=1.4)




corr = synthetic_data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, cmap='coolwarm',annot=False, cbar=False)
plt.title('TabDDPM synthetic Data Correlation Heatmap', fontsize=25)
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust this to give more space for the title
plt.savefig('figure/TabDDPM synthetic Data Correlation Heatmap.jpg')
plt.close()

print("Done")



real_data_loader = GenericDataLoader(real_data)
syn_data_loader = GenericDataLoader(synthetic_data)

# Conduct Kolmogorov-Smirnov test
ks_test = KolmogorovSmirnovTest()

# Apply KS test
result = ks_test.evaluate(real_data_loader, syn_data_loader)

# Display results
print("Kolmogorov-Smirnov test results:")
print(result)
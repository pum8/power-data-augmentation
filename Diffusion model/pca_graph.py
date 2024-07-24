from sdv.single_table import CTGANSynthesizer
import pandas as pd
from sdv.metadata import SingleTableMetadata

dataO = pd.read_csv("pca_original_results.csv")
dataS = pd.read_csv("pca_synth_results.csv")


metadata = SingleTableMetadata()
metadata.detect_from_dataframe(dataO)
python_dict = metadata.to_dict()


from sdv.evaluation.single_table import run_diagnostic, evaluate_quality

diagnostic = run_diagnostic(
    real_data=dataO,
    synthetic_data=dataS,
    metadata=metadata
)
print("Diagnostic Report: ", diagnostic)

quality_report = evaluate_quality(
    dataO,
    dataS,
    metadata)

print("Evaluation Report: ", quality_report.get_details(property_name='Column Pair Trends'))
print("Report", quality_report.get_details(property_name='Column Shapes'))

fig = quality_report.get_visualization(property_name='Column Shapes')
fig.update_layout(showlegend=False, font=dict(size=18))

from sklearn.preprocessing import MinMaxScaler


scaler = MinMaxScaler()
dataO = pd.DataFrame(scaler.fit_transform(dataO), columns=dataO.columns)
dataS = pd.DataFrame(scaler.transform(dataS), columns=dataS.columns)



from sdv.evaluation.single_table import get_column_plot, get_column_pair_plot

def fig_generator(feature):
    fig = get_column_plot(
        real_data=dataO,
        synthetic_data=dataS,
        metadata=metadata,
        column_name=feature
                )
    fig.update_layout(
        title="Principal Component Analysis",
        showlegend=True,
        font=dict(size=25),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5
        ),
        yaxis=dict(
            showticklabels=False,  
            title='Density Frequency'  
        ),
        xaxis=dict(
            range=[0, 1]  
        )
    )
    fig.write_image(f"figure/{feature}_normalized.jpg")
    """
    if feature != 'pwr_avg':
        fig2 = get_column_pair_plot(
            real_data=real_data,
            synthetic_data=synthetic_data,
            metadata=metadata,
            column_names=[feature, 'pwr_avg'],
        )
        fig2.update_layout(showlegend=False, font=dict(size=18))
        fig2.write_image(f"ctganfigure/{feature} VS PWR_AVG.jpg")
    """

fig_generator('PCA')




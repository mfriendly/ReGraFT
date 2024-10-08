import matplotlib.pyplot as plt

import seaborn as sns
import torch
import json, os
import numpy as np
import pandas as pd
from datetime import datetime
OUT_PATH = "z_Variable_Importance"
current_timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
os.makedirs(os.path.join(OUT_PATH, current_timestamp), exist_ok=True)



def thousand_separator(x, pos):
    return f'{x:, .0f}'

def make_selection_plot_Final(fig, ax, title, values, labels, colors):
    with open('x_data_US/vars.json', 'r') as file:
        variable_labels = json.load(file)
    variable_labels = variable_labels
    values = torch.tensor(values, dtype=torch.float) * 100.
    order = np.argsort(values.numpy())
    ordered_values = values[order]
    ordered_labels = [label for label in np.asarray(labels)[order]]
    ordered_colors = [colors[i] for i, label in enumerate(ordered_labels)]
    ordered_labels = [
        label.replace('_', ' ').capitalize() for label in ordered_labels
    ]
    bars = ax.barh(np.arange(len(values)),
                   ordered_values,
                   tick_label=ordered_labels,
                   color=ordered_colors,
                   edgecolor='black',
                   linewidth=0.5)

    ax.set_title(title)

    max_val = max(ordered_values.numpy())
    padding = max_val * 0.005
    max_width = max(ordered_values.numpy())
    min_width = min(ordered_values.numpy())
    for bar, value in zip(bars, ordered_values):
        text = f"{value:.2f}"
        text_width = ax.text(bar.get_width() + padding,
                             bar.get_y() + bar.get_height() / 2,
                             text,
                             va='center').get_window_extent().width
        fig.canvas.draw()
        pixels_to_units = fig.dpi_scale_trans.inverted().transform(
            (text_width, 0))[0] / 10
        max_width = max(max_width, bar.get_width() + padding + pixels_to_units)
    MINX, MAXX = 10.3, 13
    ax.set_xlim(max(MINX, min_width - 1), min(MAXX, max_width + padding * 4.))

    def save_labels_and_values(labels, values):

        with open(
                f'{OUT_PATH}/{current_timestamp}/US_var_importance_final_labels.json',
                'w') as f:
            json.dump(labels, f, indent=4)

        np.save(
            f'{OUT_PATH}/{current_timestamp}/US_var_importance_final_values.npy',
            values.numpy())

    save_labels_and_values(ordered_labels, ordered_values)
    return fig

def Plot_VariableImportance_Globally_fromCSV_include_holiday_vars(
        df_AVG, df2, df3, filename):
    pass

    sns.color_palette("pastel")

    color_list = ['skyblue'] * 30

    with open('x_data_US/vars.json', 'r') as file:
        all_cs = json.load(file)
    all_cs = all_cs

    col_filter = [a for a in df_AVG.columns if a in all_cs]
    df_AVG = df_AVG[col_filter]
    handles, labels = [], []
    ll = ['Avg', 'Encoder', 'Decoder']
    figs, ax = plt.subplots(1, 1, figsize=(7, 15 * 0.25 + 2))
    for n, type in enumerate(ll):

        if type == 'Avg':

            fig_path = f"Variable Importance {type}"
            df_AVG.to_csv(f'{fig_path}_Final.csv'.replace(
                ' ', '-'))
            print("df_AVG", df_AVG)
            variable_labels = df_AVG.columns.tolist()

            mean_values = df_AVG.iloc[0, :].values

            fig = make_selection_plot_Final(figs, ax, "", mean_values,
                                       variable_labels[:], color_list)

    for handle, label in zip(*ax.get_legend_handles_labels()):
        if label not in labels:
            handles.append(handle)
            labels.append(label)
    plt.tight_layout()

    labels = [
        "is_holiday", "categorical_week", "stringency_index", "school_closing"
    ]

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.subplots_adjust(left=0.05,
                        right=0.95,
                        top=0.85,
                        bottom=0.15,
                        wspace=0.1,
                        hspace=0.35)

    plt.tight_layout()

    if False:

        plt.savefig(filename, dpi=300)

def sort_and_save_csv(input_file, output_file, output_file_xlabel):

    df = pd.read_csv(input_file)

    values = df.values[0]

    df_values = pd.DataFrame({'Column': df.columns, 'Value': values})

    df_sorted = df_values.sort_values(by='Value', ascending=False)

    df_sorted_columns = df[df_sorted['Column'].values]

    df_sorted_columns.to_csv(output_file, index=False)

    df_sorted_columns_Xlabel = df_sorted_columns.drop(
        columns=['new_confirmed'])
    df_sorted_columns_Xlabel.to_csv(output_file_xlabel, index=False)
    print(f"Sorted CSV saved as '{output_file}'")

def Take_Avg_Importance(file1, file2, output_file):
    df1 = pd.read_csv(file1).iloc[:, 1:]
    df2 = pd.read_csv(file2).iloc[:, 1:]

    common_columns = df1.columns.intersection(df2.columns)

    df_avg = pd.DataFrame()

    if len(common_columns) > 0:
        df1_common = df1[common_columns]
        df2_common = df2[common_columns]
        df_avg = (df1_common + df2_common) / 2

    for col in df1.columns:
        if col not in common_columns:
            df_avg[col] = df1[col]

    df_avg = df_avg[df1.columns]

    df_avg.to_csv(output_file, index=False)
    print(f"Averaged CSV saved as '{output_file}'")

    sort_and_save_csv(output_file, output_file.replace('AVG', 'AVG_sorted'),
                      output_file.replace('AVG', 'AVG_sorted_Xlabel'))

if __name__ == "__main__":
    nation = "US"
    from plt_utils import Prepare_Plot_VarImportance
    Prepare_Plot_VarImportance(nation, current_timestamp)

    E = f'{OUT_PATH}/{current_timestamp}/VariableImportanceEncoder.csv'
    D = f'{OUT_PATH}/{current_timestamp}/VariableImportanceDecoder.csv'
    AVG = f'{OUT_PATH}/{current_timestamp}/VariableImportanceAVG.csv'
    AVG_sorted_Xlabel = f'{OUT_PATH}/{current_timestamp}/VariableImportanceAVG_sorted_Xlabel.csv'

    Take_Avg_Importance(E, D, AVG)

    df_AVG = pd.read_csv(AVG)
    df3 = pd.read_csv(D)
    df2 = pd.read_csv(E)
    save_fname = os.getcwd(
    ) + f'/{OUT_PATH}/{current_timestamp}/{nation}_var_importance_out.png'
    print("save_fname", save_fname)
    Plot_VariableImportance_Globally_fromCSV_include_holiday_vars(
        df_AVG, df2, df3, save_fname)

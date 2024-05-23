import numpy as np
import seaborn as sns
import pandas as pd
import torch
import os
import matplotlib.pyplot as plt
import matplotlib
import scienceplots

plt.style.use(["science", "no-latex"])
plt.rcParams.update({"font.size": 11})


def visualize_THREE_similarity_matrices(similarity_matrices,
                                        names=None,
                                        list_of_states=None):
    names = [
        "(a)", "(b)", "(c)"
    ]  #"(a) Distance Matrix", "(b) Travel Matrix", "(c) Dynamic Matrix"]#names = ["(a) Distance Matrix", "(b) Travel Matrix", "(c) Dynamic Matrix"]
    fig_width, fig_height = 30, 10
    fig, axes = plt.subplots(1,
                             3,
                             figsize=(30, 10),
                             sharex=False,
                             sharey=False)
    matrix_height = len(list_of_states)
    plt.subplots_adjust(wspace=0.1, hspace=0.3)
    bottom_margin, top_margin = 0.11, 0.88
    cbar_height = (top_margin - bottom_margin)
    cbar_ax = fig.add_axes([0.93, bottom_margin, 0.03, cbar_height])
    for i, ax in enumerate(axes.flat):
        matrix = similarity_matrices[i]
        if isinstance(matrix, torch.Tensor):
            matrix = matrix.cpu().numpy() / 10.0
        else:
            matrix = matrix / 10.0
        sns_heatmap = sns.heatmap(matrix,
                                  ax=ax,
                                  cmap="viridis",
                                  cbar=i == 2,
                                  cbar_ax=None if i != 2 else cbar_ax,
                                  linewidths=0.5)
        ax.set_title(names[i], fontsize=35)
        ticks = np.arange(len(list_of_states))
        ax.set_xticks([])
        ax.set_yticks([])
    if 'sns_heatmap' in locals():
        cbar = sns_heatmap.collections[0].colorbar
        cbar.ax.tick_params(labelsize=22)
    import datetime
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime('%y%m%d_%H%M')
    plt.savefig(f'THREE_{formatted_time}.png')


stateUS = pd.read_csv(".." + f"/data/x_data_aux/statemappingUS.csv")
list_of_states = stateUS["State"].tolist()
list_of_abbr = stateUS["Abbr"].tolist()
dist_matrix = np.load(".." + f"/data/x_data_aux/matrix_0.npy")
travel_matrix = np.load(".." + f"/data/x_data_aux/matrix_1.npy")
dyn_matrix = np.load(".." + f"/data/x_data_aux/matrix_2.npy")
similarity_matrices = [dist_matrix, travel_matrix, dyn_matrix]


def thousand_separator(x, pos):
    return f"{x:,.0f}"


def plot_bar_avg_over_time(attn_matrix, title, ylabel, save_path, labels=None):
    plt.figure(figsize=(8, 6))
    flat = np.mean(attn_matrix, axis=1)
    plt.bar(range(len(flat)), flat, color="skyblue")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_attention_matrix(attn_matrix, title, ylabel, save_path, labels=None):
    plt.figure(figsize=(8, 6))
    plt.imshow(attn_matrix, aspect="auto", cmap="viridis")
    plt.colorbar()
    plt.title(title)
    plt.ylabel(ylabel)
    if labels is not None and attn_matrix.shape[1] == len(labels):
        plt.xticks(range(len(labels)), labels, rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


if False:
    for col in COLS:
        avg_sales = df.groupby([col])['SALE_ALL'].mean().reset_index()
        log('The correlation between average sales & TYPE is: ',
            np.round(avg_sales[[col, 'SALE_ALL']].corr(), 4))
    j
    pivot_df = df.pivot_table(index='date',
                              columns='MAJOR_TYPE_NM',
                              values='SALE_ALL',
                              aggfunc=np.sum,
                              fill_value=0)
    scaler = MinMaxScaler()
    scaled_df = scaler.fit_transform(pivot_df)
    scaled_df = pd.DataFrame(scaled_df,
                             columns=pivot_df.columns,
                             index=pivot_df.index)
    fig, ax = plt.subplots(figsize=(8, 8))
    for MAJOR_TYPE_NM in scaled_df.columns:
        scaled_df[MAJOR_TYPE_NM].plot(ax=ax, label=MAJOR_TYPE_NM)
    ax.set_title("Scaled Sales Over Time")
    ax.set_ylabel('Scaled SALE_ALL')
    ax.legend(loc='best')
    plt.tight_layout()
    plt.savefig(create_fname() + "scaled_combined_.png")
    plt.close(fig)
    j
    pivot_df = df.pivot_table(index='date',
                              columns='MAJOR_TYPE_NM',
                              values='SALE_ALL',
                              aggfunc=np.sum,
                              fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 8))
    for MAJOR_TYPE_NM in pivot_df.columns:
        pivot_df[MAJOR_TYPE_NM].plot(ax=ax, label=MAJOR_TYPE_NM)
    ax.set_title("Sales Over Time")
    ax.set_ylabel('SALE_ALL')
    ax.legend(loc='best')
    plt.tight_layout()
    plt.savefig(create_fname() + "combined_auto_output.png")
    plt.close(fig)
    j
    scaler = MinMaxScaler()
    normalized_data = pd.DataFrame(scaler.fit_transform(combined_data),
                                   columns=combined_data.columns,
                                   index=combined_data.index)
    fig, axes = plt.subplots(
        len(normalized_data.columns[-len(colss):]),
        len(normalized_data.columns[:-len(colss)]),
        figsize=(5, 3 * len(normalized_data.columns[-len(colss):])))
    for i, (ax, column_) in enumerate(
            zip(axes, normalized_data.columns[-len(colss):])):
        for column in normalized_data.columns[:-len(colss)]:
            ax.plot(normalized_data.index,
                    normalized_data[column],
                    label=column,
                    linestyle='dashed',
                    color='grey')
        n_colors = len(colss)
        colors = sns.color_palette("pastel", n_colors)
        ax.plot(normalized_data.index,
                normalized_data[column_],
                label=column_,
                color=colors[1],
                linewidth=2.5)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.set_title(f'{column_}')
        ax.set_xlabel('YYYYMM')
        ax.legend(loc='upper left',
                  bbox_to_anchor=(1, 1),
                  ncol=1,
                  fontsize='small')
    plt.tight_layout()
    plt.savefig(".." + f"/norm" + create_fname())
    plt.close()
    jjjj
    plt.figure(figsize=(14, 9))
    for i, column_ in enumerate(normalized_data.columns[-len(colss):]):
        for column in normalized_data.columns[:-len(colss)]:
            plt.plot(normalized_data.index,
                     normalized_data[column],
                     label=column,
                     linestyle='dashed',
                     color='grey')
        n_colors = len(colss)
        colors = sns.color_palette("pastel", n_colors)
        plt.plot(normalized_data.index,
                 normalized_data[column_],
                 label=column,
                 color=colors[i])
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.gca().xaxis.set_major_locator(mdates.YearLocator())
        plt.gcf().autofmt_xdate()
        plt.title('Normalized Time Series Data')
        plt.xlabel('YYYYMM')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=1)
    plt.tight_layout()
    plt.savefig(".." + f"/norm" + create_fname())
    j
    type_sales_corr = data.pivot_table(index='YYYYMM',
                                       columns='MAJOR_TYPE_NM',
                                       values='SALE_ALL',
                                       aggfunc='sum').corr()
    plt.figure(figsize=(8, 8))
    sns.heatmap(type_sales_corr,
                cmap='coolwarm',
                vmin=-1,
                vmax=1,
                annot=True,
                fmt='.2f',
                linewidths=.5)
    plt.title('상관관계 Heatmap of 매출액 by Type (TYPE_NM)')
    plt.tight_layout()
    plt.savefig(".." + "/sangwon_data/" + create_fname())
    region_type_sales = data.groupby(['SIDO_NM', 'TYPE_NM'
                                      ])['SALE_ALL'].sum().reset_index()
    top_sales_type_per_region = region_type_sales.groupby('SIDO_NM').apply(
        lambda x: x.nlargest(1, 'SALE_ALL')).reset_index(drop=True)
    order_sido = data.groupby('SIDO_NM')['SALE_ALL'].max().sort_values(
        ascending=False).index.tolist()
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='SIDO_NM', y='SALE_ALL', data=data, order=order_sido)
    plt.title('Boxplot of 매출액 - 대분류 지역 (SIDO_NM)')
    plt.xticks(rotation=90)
    plt.ylabel('매출액 (원)')
    plt.savefig(".." + f"/sangwon_data/" + create_fname())
    plt.close()


def plot2(orig_train1,
          trend_train1,
          season_train1,
          rem_train1,
          date,
          name="HC"):
    fig, axs = plt.subplots(4, figsize=(25, 25), sharex=True)
    date = pd.to_datetime(date)
    axs[0].plot(date, orig_train1[:, 0], color="black", linewidth=1)
    axs[0].set_title("Original")
    axs[1].plot(date, trend_train1[:, 0], color="blue", linewidth=1)
    axs[1].set_title("Trend")
    axs[2].plot(date, season_train1[:, 0], color="red", linewidth=1)
    axs[2].set_title("Seasonal")
    axs[3].plot(date, rem_train1[:, 0], color="black", linewidth=1)
    axs[3].set_title("Remainder")
    for ax in axs:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.tick_params(axis='x', rotation=45)
    plt.style.use(["science", "no-latex"])
    plt.rcParams.update({"font.size": 11})
    fig.tight_layout()
    fig.savefig(f"tsa_seasonality_{name}.png")


def viz_feature_importance(attention_weights_node, node_id, state_name):
    attention_weights_node_np = attention_weights_node.detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(10, 6))
    cax = ax.imshow(attention_weights_node_np, cmap="viridis", aspect="auto")
    fig.colorbar(cax)
    ax.set_title(f"Attention Weights for Node {node_id} {state_name}")
    ax.set_xlabel("Key Positions")
    ax.set_ylabel("Query Positions")
    plt.savefig(f"{str(node_id+1)}_th_{state_name}_node_attention.png")
    plt.close()


if False:
    ind = 0
    for idx, batch in enumerate(test_loader):
        output, attention_weights = model(batch, dist_matrix, travel_matrix,
                                          global_step)
        break
    if attention_weights is not None:
        for node_id in range(num_nodes):
            if node_id > 1:
                break
            state_name = list_of_states[node_id]
            abbr = dict_of_states_rev[state_name]
            plt.figure(figsize=(10, 6))
            attention_weights = attention_weights.view(BATCHSIZE, num_nodes,
                                                       -1, past_steps)
            attention_weights_node = attention_weights[ind, node_id, :, :]
            attention_weights_node_np = attention_weights_node.detach().cpu(
            ).numpy()
            fig, ax = plt.subplots(figsize=(10, 6))
            cax = ax.imshow(attention_weights_node_np,
                            cmap="viridis",
                            aspect="auto")
            fig.colorbar(cax)
            ax.set_title(f"Attention Weights for Node {node_id} {state_name}")
            ax.set_xlabel("Key Positions")
            ax.set_ylabel("Query Positions")
            plt.savefig(
                f"y_attn/{str(node_id+1)}_th_{state_name}_node_attention.png")
            plt.close()


def save_matrix_as_png(matrix, filename):
    matrix_np = matrix.cpu().numpy()
    plt.figure(figsize=(10, 10))
    plt.imshow(matrix_np, cmap="viridis")
    plt.colorbar()
    plt.title(filename)
    plt.savefig(f"{filename}.png")
    plt.close()


def visualize_similarity_matrix(
        similarity_matrix,
        list_of_states,
        file_name="similarity_matrix_visualization.png"):
    if similarity_matrix.ndim == 3:
        similarity_matrix = similarity_matrix[0, :, :]
    similarity_matrix = similarity_matrix
    fig, ax = plt.subplots(figsize=(20, 20))
    cax = ax.matshow(similarity_matrix, cmap="viridis")
    fig.colorbar(cax)
    ax.set_title("Similarity Matrix Visualization", pad=20)
    ticks = np.arange(len(list_of_states))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(list_of_states, rotation=90, ha="left")
    ax.set_yticklabels(list_of_states)
    plt.gcf().subplots_adjust(bottom=0.15, left=0.15)
    plt.savefig(file_name, bbox_inches="tight")
    plt.close()
    return file_name

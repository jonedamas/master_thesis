import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import json


def plot_fit(
        model_dict: list[str],
        model_labels: list[str] = None,
        view: int = 400
    ) -> plt.Figure:

    fig, ax = plt.subplots(figsize=(10, 5), dpi=200)

    colors = sns.color_palette('bright', n_colors=len(model_dict))

    model_names = list(model_dict.keys())

    if model_labels is None:
        model_labels = model_names

    for i, model_name in enumerate(model_names):
        model = model_dict[model_name]
        if i == 0:
            actual = model.y_test[-view:]
            ax.plot(
                actual,
                label=f'Actual {model.model_name.split("_")[0]} RV',
                color='gray',
                lw=0.9
            )
        ax.plot(
            model.y_pred[-view:],
            label=' '.join(model_labels[i].split('_')[1:3]),
            color=colors[i],
            lw=0.9
        )

    ax.set_xlabel('Time (5-min intervals)', fontsize=13)
    ax.set_ylabel('Annualized Realized Volatility', fontsize=13)
    ax.legend(frameon=False, ncols=3, fontsize=12)
    ax.grid(alpha=0.2)

    return fig


def plot_loss(
        model_names: list[str],
        model_labels: list[str] = None,
    ) -> plt.Figure:

    if model_labels is None:
        model_labels = model_names

    loss_df_list= list()
    for i, model_name in enumerate(model_names):
        if model_name.split('_')[1] == 'VAR':
            continue
        with open(f'model_archive/{model_name}/loss_data.json', 'r') as f:
            loss_dict = json.load(f)

            loss_df = pd.DataFrame(loss_dict).add_suffix(f'_{model_name}')
            loss_df_list.append(loss_df)

    loss_df = pd.concat(loss_df_list, axis=1)

    colors = sns.color_palette('bright', n_colors=len(model_names))

    fig, ax = plt.subplots(figsize=(9, 5), dpi=200)
    model_lines = []
    for i, model_name in enumerate(model_names):
        ax.plot(loss_df.index, loss_df[f'train_loss_{model_name}'], color=colors[i], linestyle='--')
        line = ax.plot(
            loss_df.index,
            loss_df[f'val_loss_{model_name}'],
            color=colors[i],
            label=' '.join(model_labels[i].split('_')[1:3])
        )
        model_lines.append(line[0])

    # First legend for models
    first_legend = ax.legend(
        handles=model_lines,
        loc='upper right',
        ncol=1,
        fontsize=14
    )
    ax.add_artist(first_legend)

    train_lines = ax.plot([], [], color='black', linestyle='--', label='Training Loss')
    val_lines = ax.plot([], [], color='black', label='Validation Loss')

    ax.legend(
        handles=[train_lines[0], val_lines[0]],
        loc='lower center',
        bbox_to_anchor=(0.5, -0.22),
        ncol=2,
        fontsize=14
    )

    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Loss [MSE]', fontsize=14)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    return fig


def plot_importance(
        importance_df: pd.DataFrame
    ) -> plt.Figure:

    model_order = ['LSTM', 'GRU', 'BiLSTM', 'BiGRU']

    bar_order = [
        'units_first_layer',
        'units_second_layer',
        'l2_strength',
        'learning_rate',
        'batch_size',
        'noise_std',
        'window_size'
    ]
    mean_df = importance_df.groupby('model_type').mean()
    mean_df = mean_df[bar_order]

    model_order = [model for model in model_order if model in mean_df.index]

    fig, ax = plt.subplots(figsize=(8, 3.7), dpi = 200)

    colors = sns.color_palette("twilight", mean_df.shape[1] + 1)

    mean_df.plot(kind='bar', ax=ax, edgecolor='black', color=colors, width=0.75)
    ax.set_ylim(top=ax.get_ylim()[1] * 1.2)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_ylabel('Importance', fontsize=14)
    ax.set_xlabel('Model', fontsize=14)
    ax.set_xticklabels(model_order, fontsize=14, rotation=0)
    plt.tight_layout()

    handles, labels = ax.get_legend_handles_labels()
    labels = [label.replace('_', ' ').capitalize() for label in labels]
    ax.legend(handles, labels, frameon=False, ncol=4, loc='upper left', fontsize=12)

    return fig

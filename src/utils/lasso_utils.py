import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def plot_lasso(
        lasso_df: pd.DataFrame,
        coefs_lasso: list[np.ndarray]
    ) -> plt.Figure:

    fig, axs = plt.subplots(1, 2, figsize=(12, 5), dpi=200)

    plot_y = [coefs_lasso, lasso_df['MSE']]
    label_y = ['Mean Square Error', 'Coefficients']
    vlines = [lasso_df['MSE']]

    # get twilight colors
    colormap = cm.get_cmap('twilight_r', 20)

    # Set the color cycle to the twilight colormap
    axs[0].set_prop_cycle(color=colormap(np.linspace(0, 1, 8)))
    axs[1].set_prop_cycle(color=colormap(np.linspace(0, 1, 2)))

    for i, ax in enumerate(axs.flatten()):
        ax.plot(lasso_df['lambda'], plot_y[i], color='black' if i == 1 else None, lw=1.2)
        ax.set_xscale('log')
        # adjust tick size
        ax.tick_params(axis='both', which='major', labelsize=13)
        ax.set_xlabel('$\lambda$', fontsize=15)
        ax.set_ylabel(label_y[1] if i % 2 == 0 else label_y[0], fontsize=15)
        ax.axvline(
            lasso_df['lambda'].to_numpy()[np.argmin(vlines[0] if i < 2 else vlines[1])],
            color='red',
            linestyle='-.',
            lw=1
        )

    fig.tight_layout()

    return fig

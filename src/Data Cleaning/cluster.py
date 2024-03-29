import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from typing import List

from IPython.display import display, Markdown
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

import sys

sys.path.insert(0, r'c:\Users\joneh\master_thesis\src')
from main_utils import *


class NewsCluster:
    def __init__(
            self,
            df: pd.DataFrame,
            tag: str,
            pca_components: int = 2,
            eps: float = 0.03
        ):
        ''' Clusters news headlines using TF-IDF, PCA, and DBSCAN

        Args:
            df: DataFrame with a column 'headline' containing the news headlines

        KwArgs:
            pca_components: Number of PCA components to use for dimensionality reduction

            eps: DBSCAN epsilon parameter
        '''
        self.df = df.dropna()
        self.pca_components = pca_components
        self.eps = eps  # DBSCAN epsilon
        self.tag = tag

        self.info = load_json(r'C:\Users\joneh\master_thesis\src\plot_dict.json')

        self.documents = self.df['headline'].tolist()

        self.vectorizer = TfidfVectorizer(stop_words='english')
        X = self.vectorizer.fit_transform(self.documents)

        self.pca = PCA(n_components=self.pca_components)
        self.X_reduced = self.pca.fit_transform(X.toarray())

        dbscan = DBSCAN(eps=self.eps, min_samples=2)
        self.clusters = dbscan.fit_predict(self.X_reduced)

        self.df['cluster'] = self.clusters


    def print_clusters(self) -> None:
        '''Prints the news headlines in each cluster
        '''
        display(Markdown(f"## {self.info[self.tag]['fullname']} Clusters:"))

        for cluster in np.unique(self.clusters):
            display(Markdown(f"**Cluster {cluster}**"))
            display(self.df[self.df['cluster'] == cluster]['headline'])


    def remove_cluster(self, clusters: List[int]) -> pd.DataFrame:
        '''Removes a cluster from the DataFrame and returns the cleaned DataFrame

        Args:
            clusters: List of cluster numbers to remove

        Returns:
            clean_df: DataFrame with the specified clusters removed
        '''
        clean_df = self.df[~self.df['cluster'].isin(clusters)]

        print(f'Removed {len(self.df) - len(clean_df)} articles.')

        return clean_df


    def plot_clusters(self) -> tuple[plt.Figure, plt.Axes]:
        '''Plots the clusters in 2D using PCA

        Returns:
            fig, ax: Matplotlib Figure and Axes objects
        '''
        fig, ax = plt.subplots(figsize=(7, 5), dpi=200)
        ax.set_axisbelow(True)
        ax.grid(alpha=0.25, linestyle='--')

        ax.scatter(
            self.X_reduced[:, 0],
            self.X_reduced[:, 1],
            c=self.clusters,
            cmap='viridis',
            marker='.',
            s = 35
        )

        ax.set_title(f"{self.info[self.tag]['fullname']} news clusters")
        ax.set_xlabel("PCA Dimension 1")
        ax.set_ylabel("PCA Dimension 2")

        ax.set_ylim(ax.get_ylim()[0] * 1.1, ax.get_ylim()[1] * 1.1)
        ax.set_xlim(ax.get_xlim()[0] * 1.1, ax.get_xlim()[1] * 1.1)

        max_cluster = max(self.df['cluster'].unique())
        dy = ax.get_ylim()[1] - ax.get_ylim()[0]

        for i in range(1, max_cluster + 1):
            text = f'Cluster {i}'
            centroid = np.mean(self.X_reduced[self.clusters == i], axis = 0)

            ax.text(centroid[0], centroid[1] + dy / 20, text, fontsize=9, ha='center')

        return fig, ax

from gensim.models import LdaMulticore
from gensim.corpora.dictionary import Dictionary
from gensim import corpora
import pandas as pd
import pyLDAvis.gensim
import seaborn as sns
from matplotlib.axes import Axes

from typing import Dict


class LDAModelSetup:
    """
    Class to set up the LDA model.

    Attributes
    ----------
        tokenized_series: pd.Series
        name: str
        stopwords: set[str]
        lda_params: dict[str, int]
        doc_list: list
        stop_words: set[str]
        dictionary: corpora.Dictionary
        corpus: list
    """
    def __init__(
            self,
            tokenized_series: pd.Series,
            name: str,
            stopwords: set[str],
            lda_params: Dict[str, int]
        ):
        self.tokenized_series = tokenized_series
        self.params = lda_params
        self.name = name
        self.model = None
        self.visfig = None

        self.doc_list = tokenized_series.to_list()

        self.stop_words =  stopwords

        self.stop_words.update({str(i) for i in range(3000)})

        self.doc_list = [[word for word in doc if word not in self.stop_words] for doc in self.doc_list]

        self.dictionary = corpora.Dictionary(self.doc_list)
        self.dictionary.filter_extremes(no_below=5, no_above=0.5)

        self.dictionary.id2token = {id: token for token, id in self.dictionary.token2id.items()}

        self.corpus = [self.dictionary.doc2bow(doc) for doc in self.doc_list]

    def print_top_words(
            self,
            n_words: int = 10
        ) -> None:

        for i, topic in self.model.show_topics(
                num_topics=-1,
                num_words=n_words,
                formatted=False
            ):
            print(f'Topic {i}: {[word[0] for word in topic]}')


    def generate_model(self) -> None:
        self.model = LdaMulticore(
            corpus=self.corpus,
            id2word=self.dictionary.id2token,
            workers=6,
            eta='auto',
            **self.params
        )

    def generate_pyLDAvis(self) -> None:
        self.visfig = pyLDAvis.gensim.prepare(
            topic_model = self.model,
            corpus = self.corpus,
            dictionary = self.dictionary,
            sort_topics=False
        )

    def plot_pyLDAvis(self, ax: Axes) -> None:
        pc_data = self.visfig.topic_coordinates

        colors = sns.color_palette('twilight', n_colors=len(pc_data))
        pc_data.plot(
            kind='scatter', x='x', y='y',
            s=pc_data['Freq'] * 300,
            ax=ax, c=colors, alpha=0.8
        )

        ax.axhline(0, color='gray', linewidth=0.5, zorder=0)
        ax.axvline(0, color='gray', linewidth=0.5, zorder=0)

        ax.set_xlim(ax.get_xlim()[0] - 0.1, ax.get_xlim()[1] + 0.1)
        ax.set_ylim(ax.get_ylim()[0] - 0.1, ax.get_ylim()[1] + 0.1)

        ax.spines['left'].set_position('zero')
        ax.spines['bottom'].set_position('zero')
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')

        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

        ax.set_xticks([i for i in ax.get_xticks() if i != 0])
        ax.set_yticks([i for i in ax.get_yticks() if i != 0])

        ax.xaxis.set_label_coords(0.5, -0.025)
        ax.yaxis.set_label_coords(-0.025, 0.5)

        ax.set_xlabel('PC1', fontsize=16)
        ax.set_ylabel('PC2', fontsize=16)

        axiscolor = 'dimgray'

        ax.spines['bottom'].set_color(axiscolor)
        ax.spines['left'].set_color(axiscolor)
        ax.tick_params(axis='x', colors=axiscolor)
        ax.tick_params(axis='y', colors=axiscolor)

        for _, row in pc_data.iterrows():
            ax.text(row['x'], row['y'],
                    int(row['topics']), fontsize=22,
                    ha='center', va='center',
                    color='white'
                )
            ax.text(row['x'], row['y'] - 0.03,
                    f"{row['Freq']:.0f}%", fontsize=10,
                    ha='center', va='center',
                    color='white'
                )

        ax.text(
            0.00, 1.00, self.name, fontsize=24,
            color='dimgray', transform=ax.transAxes
        )


def classify_article(
        article,
        dictionary: Dictionary,
        model: LdaMulticore
    ) -> int:
    """
    Classifies the given article into a topic.

    Parameters
    ----------
        article: dict
        dictionary: Dictionary
        model: LdaModel

    Returns
    -------
        int
    """
    bow = dictionary.doc2bow(article['cleaned_tokenized'])
    topic_distribution = model.get_document_topics(bow)
    topic_nr = max(topic_distribution, key=lambda x: x[1])[0]

    return topic_nr

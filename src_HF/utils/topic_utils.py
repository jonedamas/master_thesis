from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary
from gensim import corpora
import pandas as pd
import pyLDAvis.gensim
import matplotlib.pyplot as plt
import seaborn as sns


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
            lda_params: dict[str, int]
        ):
        self.tokenized_series = tokenized_series
        self.params = lda_params
        self.name = name
        self.model = None
        self.visfig = None

        self.doc_list: list = tokenized_series.to_list()

        self.stop_words: set[str] =  stopwords

        # addnumbers to stop words
        self.stop_words.update({str(i) for i in range(3000)})

        # remove stop words
        self.doc_list = [[word for word in doc if word not in self.stop_words] for doc in self.doc_list]

        self.dictionary = corpora.Dictionary(self.doc_list)
        self.dictionary.filter_extremes(no_below=5, no_above=0.5)

        self.dictionary.id2token = {id: token for token, id in self.dictionary.token2id.items()}

        self.corpus = [self.dictionary.doc2bow(doc) for doc in self.doc_list]

    def generate_model(self) -> None:
        self.model = LdaModel(
            corpus=self.corpus,
            id2word=self.dictionary.id2token,
            alpha='auto',
            eta='auto',
            **self.params
        )

    def generate_pyLDAvis(self) -> None:
        self.visfig = pyLDAvis.gensim.prepare(
            topic_model = self.model,
            corpus = self.corpus,
            dictionary = self.dictionary
        )

    def plot_pyLDAvis(self) -> plt.Figure:
        pc_data = self.visfig.topic_coordinates

        fig, ax = plt.subplots(figsize=(7, 7), dpi=200)
        colors = sns.color_palette('twilight', n_colors=len(pc_data))
        pc_data.plot(
            kind='scatter', x='x', y='y',
            s=pc_data['Freq'] * 300,
            ax=ax, c=colors, alpha=0.8
        )

        ax.axhline(0, color='gray', linewidth=0.5, zorder=0)
        ax.axvline(0, color='gray', linewidth=0.5, zorder=0)

        # adjust x and ylim
        ax.set_xlim(ax.get_xlim()[0] - 0.1, ax.get_xlim()[1] + 0.1)
        ax.set_ylim(ax.get_ylim()[0] - 0.1, ax.get_ylim()[1] + 0.1)

        # place 0,0 in the center
        ax.spines['left'].set_position('zero')
        ax.spines['bottom'].set_position('zero')
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')

        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

        # remove only the 0,0 ticks
        ax.set_xticks([i for i in ax.get_xticks() if i != 0])
        ax.set_yticks([i for i in ax.get_yticks() if i != 0])

        ax.xaxis.set_label_coords(0.5, -0.025)  # x label slightly below the x-axis at the center
        ax.yaxis.set_label_coords(-0.025, 0.5)

        ax.set_xlabel('PC1', fontsize=16)
        ax.set_ylabel('PC2', fontsize=16)

        axiscolor = 'dimgray'

        # make the axis and ticks gray
        ax.spines['bottom'].set_color(axiscolor)
        ax.spines['left'].set_color(axiscolor)
        ax.tick_params(axis='x', colors=axiscolor)
        ax.tick_params(axis='y', colors=axiscolor)

        # annotate the points in center
        for _, row in pc_data.iterrows():
            ax.text(row['x'], row['y'],
                    int(row['topics']), fontsize=22,
                    ha='center', va='center',
                    color='white'
                )

        ax.text(
            0.00, 1.00, self.name, fontsize=24,
            color='dimgray', transform=ax.transAxes
        )
        ax.text(
            0.00, 0.95, f'{self.name} articles:', fontsize=12,
            color='dimgray', transform=ax.transAxes
        )

        return fig

    def __repr__(self) -> str:
        return f'LDAModelSetup({self.name})'


def classify_article(
        article,
        dictionary: Dictionary,
        model: LdaModel
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

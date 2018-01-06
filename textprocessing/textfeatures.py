# -*- coding: utf-8 -*-
import pandas as pd
import gensim
import sklearn.feature_extraction.text as sktext

import warnings
with warnings.catch_warnings():
	warnings.filterwarnings("ignore")
	from nltk.sentiment.vader import SentimentIntensityAnalyzer as Vader
vader_scorer = Vader().polarity_scores

# modules in textauger
import preprocessing
from textprocessing.utils import log


def tfidf_vectorizer(docs, ngram_range=(1,1), max_features=None, **kwargs):
    """
    Calculate tf-idf weighted of n-grams for a given collections of a list of string.

    Args:
        docs(List): a sequence of strings
        ngrams_range(Tuple): (min_n, max_n) to define lower and upper boundary of the range of
            n-values for different n-grams to be extracted. All values of n such
            that min_n <= n <= max_n will be used.
        max_features(int or None): If not None, build a vocabulary that only consider the top
            max_features ordered by term frequency across the corpus.
        \**kargs: you can pass other parameters that accepted by TfidfVectorizer.

    Returns:
        tf-idf vectorizer
            Vectorizer is trained on the input `docs`. Can be passed to modeling functions
            or to :func:`term_doc_matrix_to_pandas` to get a Pandas DataFrame.  See the
            `scikit-learn TfidfVectorizer documentation
            <http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html>`_
            for attributes of the vectorizer.

    """
    log.info("Calculating tfidf features")

    docs = preprocessing._u(docs)

    kwargs["use_idf"] = kwargs.get("use_idf", True)
    kwargs["smooth_idf"] = kwargs.get("smooth_idf", True)

    vectorizer = sktext.TfidfVectorizer(ngram_range=ngram_range, max_features=max_features, **kwargs)
    vectorizer.fit(docs)

    return vectorizer


def count_vectorizer(docs, ngram_range=(1,1), max_features=None, **kwargs):
    """
    Calculate counts of n-grams for a given collections of a list of string.

    Args:
        docs(List): a sequence of strings
        ngrams_range(Tuple): (min_n, max_n) to define lower and upper boundary of the range of
            n-values for different n-grams to be extracted. All values of n such
            that min_n <= n <= max_n will be used.
        max_features(int or None): If not None, build a vocabulary that only consider the top
            max_features ordered by term frequency across the corpus.

    Returns:
        term frequency (count) vectorizer
            Vectorizer is trained on the input `docs`. Can be passed to modeling functions
            or to :func:`term_doc_matrix_to_pandas` to get a Pandas DataFrame.  See the
            `scikit-learn CountVectorizer documentation
            <http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html>`_
            for attributes of the vectorizer.

    """
    docs = preprocessing._u(docs)

    vectorizer = sktext.CountVectorizer(ngram_range=ngram_range, max_features=max_features, **kwargs)
    vectorizer.fit(docs)

    return vectorizer


def term_doc_matrix_to_pandas(docs, vectorizer):
    """
    Converts a term document matrix to Pandas DataFrame.

    Note: the vectorizer does not store the term document matrix for the documents it was trained
    on, so the docs list needs to be transformed.

    Warning: this loads data into memory in a Pandas DataFrame and will run into performance
    issues for large numbers of documents or terms.

    Args:
        docs: Document list to be transformed into a term document matrix by the vectorizer.
        vectorizer: Trained vectorizer, for example, the output of :func:`tfidf_vectorizer` or
            :func:`count_vectorizer`.

    Returns:
        Pandas DataFrame
            Each row is a document, each column is a term or ngram, and each value is a count (if
            input a count vectorizer) or a weight (if input a tf-idf vectorizer).
    """
    if not isinstance(vectorizer, sktext.VectorizerMixin):
        raise TypeError("vectorizer should be a scikit-learn text vectorizer")
    docs = preprocessing._u(docs)
    tdm_df = pd.DataFrame(vectorizer.transform(docs).A, columns=vectorizer.get_feature_names())

    return tdm_df.round(3)

def sort_terms(term_doc_matrix_df, ascending=False):
    """
    Sort a term document matrix Pandas DataFrame.

    Args:
        term_doc_matrix_df (Pandas DataFrame): Term document matrix as a Pandas DataFrame.
            For example, the output of :func:`term_doc_matrix_to_pandas`.
        ascending(boolean): Default is False.

    Returns:
        Pandas DataFrame

    """
    log.debug("Sorting n-gram DataFrame")
    if not isinstance(term_doc_matrix_df, pd.DataFrame):
        log.error("Parameter term_doc_matrix_df {} is not a DataFrame".format(term_doc_matrix_df))
        raise TypeError("term_doc_matrix_df should be a DataFrame")

    freq = term_doc_matrix_df.sum(axis=0)
    # sort according to freq/tf-idf weight and transfer to a data frame
    freq_df = freq.sort_values(ascending=ascending).to_frame().reset_index()
    freq_df.columns = ['term', 'sum(tfidf or count)']

    return freq_df


class Word2Vec(gensim.models.word2vec.Word2Vec):
    """
    Class for training, using and evaluating Word2Vec neural networks described
    word2vec https://code.google.com/p/word2vec/.

    The model can be stored/loaded via its save() and load() methods, or
    stored/loaded in a format compatible with the original word2vec
    implementation via save_word2vec_format() and load_word2vec_format().

    Initialize the model from an iterable of sentences. Each sentence is
    a list of words (unicode strings) that will be used for training.

    The sentences iterable can be simply a list, but for larger corpora,
    consider an iterable that streams the sentences directly from disk/network.
    See BrownCorpus, Text8Corpus or LineSentence in the module
    gensim.models.word2vec for such examples.

    If you don’t supply sentences, the model is left uninitialized -
    use if you plan to initialize it in some other way.

    Attributes:
        sg (int): defines the training algorithm. By default (sg=0), CBOW is used.
            Otherwise (sg=1), skip-gram is employed.
        size (int): the dimensionality of the feature vectors. Default is 100.
        window (int): the maximum distance between the current and predicted word
            within a sentence. Default is 5.
        alpha (float): the initial learning rate, will linearly drop to zero as
            training progresses. Default is 0.025.
        seed (int): for the random number generator. Initial vectors for each word
            are seeded with a hash of the concatenation of word + str(seed).
            Default is 1.
        min_count (int): ignore all words with total frequency lower than this.
            Default is 5.
        max_vocab_size (int): limit RAM during vocabulary building;
            if there are more unique words than this, then prune the infrequent ones.
            Every 10 million word types need about 1GB of RAM.
            Set to None for no limit (default).
        sample (float): threshold for configuring which higher-frequency words are
            randomly downsampled; default is 1e-3, useful range is (0, 1e-5).
        workers (int): use this many worker threads to train the model
            (=faster training with multicore machines). Default is 3.
        hs (int): if 1, hierarchical softmax will be used for model training.
            If set to 0 (default), and negative is non-zero, negative sampling will be used.
        negative (int): if > 0, negative sampling will be used, the int for negative
            specifies how many “noise words” should be drawn (usually between 5-20).
            Default is 5. If set to 0, no negative samping is used.
        cbow_mean (int): if 0, use the sum of the context word vectors.
            If 1 (default), use the mean. Only applies when cbow is used.
        hashfxn (hash function):  hash function to use to randomly initialize weights,
            for increased training reproducibility.
            Default is Python’s rudimentary built in hash function.
        iter (int): number of iterations (epochs) over the corpus. Default is 5.
        trim_rule: vocabulary trimming rule, specifies whether certain words
            should remain in the vocabulary, be trimmed away, or handled using
            the default (discard if word count < min_count). Can be None (min_count
            will be used), or a callable that accepts parameters (word, count, min_count)
            and returns either util.RULE_DISCARD, util.RULE_KEEP or util.RULE_DEFAULT.
            Note: The rule, if given, is only used prune vocabulary during build_vocab()
            and is not stored as part of the model.
        sorted_vocab (int): if 1 (default), sort the vocabulary by descending frequency
            before assigning word indexes.
        batch_words (int): target size (in words) for batches of examples passed to
            worker threads (and thus cython routines). Default is 10000. Larger
            batches can be passed if individual texts are longer, but the cython
            code may truncate.
    """

    def __init__(self, *args, **kwargs):
        super(Word2Vec, self).__init__(*args, **kwargs)

    def transform(self):
        raise NotImplementedError


class Doc2Vec(gensim.models.doc2vec.Doc2Vec):
    """
    Class for training, using and evaluating Doc2Vec neural networks described
    in http://arxiv.org/pdf/1405.4053v2.pdf

    Initialize the model from an iterable of documents. Each document is
    a TaggedDocument object that will be used for training.

    The documents iterable can be simply a list of TaggedDocument elements,
    but for larger corpora, consider an iterable that streams the
    documents directly from disk/network.

    If you don’t supply documents, the model is left uninitialized –
    use if you plan to initialize it in some other way.

    Attributes:
        dm (int): defines the training algorithm. By default (dm=1),
            ‘distributed memory’ (PV-DM) is used. Otherwise, distributed
            bag of words (PV-DBOW) is employed.
        size (int): the dimensionality of the feature vectors. Default is 300.
        window (int): the maximum distance between the predicted word and
            context words used for prediction within a document. Default is 8.
        alpha (float): the initial learning rate, will linearly drop to zero
            as training progresses. Default is 0.025.
        seed (int): for the random number generator. Only runs with a single
            worker will be deterministically reproducible because of the
            ordering randomness in multi-threaded runs. Default is 1.
        min_count (int): ignore all words with total frequency lower than this.
            Default is 5.
        max_vocab_size (int): limit RAM during vocabulary building; if there
            are more unique words than this, then prune the infrequent ones.
            Every 10 million word types need about 1GB of RAM. Set to None for
            no limit (default).
        sample (float): threshold for configuring which higher-frequency words
            are randomly downsampled; default is 0 (off), useful value is 1e-5.
        workers (int): use this many worker threads to train the model
            (=faster training with multicore machines).
        hs (int): if 1 (default), hierarchical sampling will be used for model
            training (else set to 0).
        negative (int): if > 0, negative sampling will be used, the int for
            negative specifies how many “noise words” should be drawn
            (usually between 5-20). Default is 0.
        dm_mean (int): if 0 (default), use the sum of the context word vectors.
            If 1, use the mean. Only applies when dm is used in non-concatenative
            mode.
        dm_concat (int): if 1, use concatenation of context vectors rather than
            sum/average; default is 0 (off). Note concatenation results in a
            much-larger model, as the input is no longer the size of one
            (sampled or arithmatically combined) word vector, but the size of
            the tag(s) and all words in the context strung together.
        dm_tag_count (int): expected constant number of document tags per
            document, when using dm_concat mode; default is 1.
        dbow_words (int): if set to 1 trains word-vectors (in skip-gram fashion)
            simultaneous with DBOW doc-vector training; default is 0 (faster
            training of doc-vectors only).
        trim_rule (callable or None): vocabulary trimming rule, specifies whether
            certain words should remain in the vocabulary, be trimmed away, or
            handled using the default (discard if word count < min_count).
            Can be None (min_count will be used), or a callable that accepts
            parameters (word, count, min_count) and returns either
            util.RULE_DISCARD, util.RULE_KEEP or util.RULE_DEFAULT. Note: The
            rule, if given, is only used prune vocabulary during build_vocab()
            and is not stored as part of the model.
    """

    def __init__(self, *args, **kwargs):
        super(Doc2Vec, self).__init__(*args, **kwargs)

def score_sentiment(text, method = "vader"):
    """
    Calculate the sentiment of a text.

    Args:
        text (string or Unicode): text to be scored.
        method ("vader"): method of scoring; "vader" is a generic method


    Returns:
        for method = "vader", a dictionary of sentiment attributes

    """
    if method not in ("vader", "capone"):
        raise ValueError, 'method must be in ("vader", "capone")'
    if method == "vader":
        return vader_scorer(text)
    else:
        ### return sentiment score using pre-trained data
        raise NotImplementedError, "the specific lexicon is not yet built"

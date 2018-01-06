import inspect
import itertools
import os
import re
import string
from collections import Iterable

import nltk
from nltk import corpus
from nltk.stem import WordNetLemmatizer

from textprocessing.utils import log

#nltk.data.path.append('/home/w205user/nltk/nltk')

def _u(text):
    """
    Helper function to turn an object or list of objects into a unicode string

    Args:
        text (object): Object to convert to unicode.

    Returns:
        List[unicode]

    """
    text = _string_to_list(text)
    try:
        ulist = [unicode(x, 'utf-8') for x in text]
    except TypeError:
        try:
            ulist = [unicode(str(x), 'utf-8') for x in text]
        except UnicodeEncodeError:
            ulist = [unicode(x.encode('utf-8'), 'utf-8', errors='ignore') for x in text]
    return ulist


def _check_is_str(value):
    """
    Check if value is a string.
    """
    return isinstance(value, basestring)


def _string_to_list(value):
    """
    Helper to convert input value into a list to make inputs uniform.

    If input is a single string, then wrap is in a list. If input
    is a list, then return the list. If input is an Iterable (but not list),
    then convert input into a list.

    Returns:
        List[str]

    """
    if _check_is_str(value):
        return [value]
    elif isinstance(value, Iterable):
        if isinstance(value, list):
            return value
        else:
            return list(value)
    elif not isinstance(value, Iterable):
        log.error("Value {} is not iterable".format(value))
        raise TypeError("Input value needs to be an Iterable")


def _tagged_word_checker(tagged_word):
    """
    Helper function to check with a word that has been tagged with part of speech.

    Makes sure the tagged word had the correct format as a tuple, tagged with one of the
    acceptable tags 'a', 'n', 'r', or 'v'.

    Args:
        tagged_word (Tuple(unicode, str)): Tuple of (word, tag).

    Returns:
        bool: If True, tagged word is acceptable for model.

    """
    if not isinstance(tagged_word, tuple):
        log.error("Input tagged_word {} is not a tuple".format(tagged_word))
        raise TypeError("tagged word needs to be a tuple of (word, tag)")
    if len(tagged_word) != 2:
        log.error("Input tuple {} has length {}".format(tagged_word, len(tagged_word)))
        raise TypeError("the length of input tuple (word, tag) needs to be 2")
    if not _check_is_str(tagged_word[0]):
        log.error("Word {} is not a string".format(tagged_word[0]))
        raise TypeError("word needs to be a string")
    tag = tagged_word[1]
    if tag is not None:
        if tag not in ('a', 'n', 'r', 'v'):
            log.error("Tag {} is invalid".format(tag))
            raise ValueError("the tag of word needs to be either 'a', 'n', 'r', or 'v'")
    # if pass all checks then return True
    return True


def consolidate_words(text, variants_dict, tokenizer="regexp", **kwargs):
    """
    Consolidate word into its canonical form according to variants_dict.

    Note that text will be tokenized first and then each token will be checked
    and consolidated. After consolidation, tokens will be joined together again.

    Args:
        text (List[unicode]): a list of strings that needs to be consolidated.
        variants_dict (dict): mapping of variants to canonical form
        tokenizer (str): Tokenizer name. Defaults to 'regexp'
        **kwargs: Optional keyword arguments for tokenizer

    Returns:
        List[unicode] with words consolidated

    """

    log.debug("Consolidating word")
    tokenized_text = tokenize(text, tokenizer, **kwargs)
    for i, text_i in enumerate(tokenized_text):
        tokenized_text[i] = [variants_dict.get(token, token) for token in text_i]
    return [' '.join(sublist) for sublist in tokenized_text]


def remove_short_tokens(text, minimum_token_length=3, tokenizer='regexp', **kwargs):
    """
    Remove small words from input text.

    The individual string will be tokenized first using a tokenizer (see function get_tokenize) and then small words (tokens) will be
    filtered out.

    Args:
        text (List[unicode])
        minimum_token_length (int): minimum length of tokens to be retained. Defaults to 3.
        tokenizer (str): Tokenizer name. Defaults to 'regexp'.
        **kwargs: Optional keyword arguments for tokenizer.

    Returns:
        List[unicode] with short words removed

    """

    log.debug("Removing short tokens")
    tokenized_text = tokenize(text, tokenizer, **kwargs)
    for i, text_i in enumerate(tokenized_text):
        tokenized_text[i] = [token for token in text_i if len(token) >= minimum_token_length]
    return [' '.join(sublist) for sublist in tokenized_text]


def get_lemmatizer():
    """
    Returns an instance of WordNet's lemmatizer.
    """
    log.debug("Returning WordNetLemmatizer")
    return WordNetLemmatizer()


def lemmatize_tagged_word(tagged_word, lemmatizer):
    """
    Lemmatize a tagged word in the form of (word, tag).

    Args:
        tagged_word (Tuple(unicode, str)): Tuple of (word, tag).
            Tags can only be 'a', 'n', 'r', or 'v'.
            'a' -> adjective, 'n' -> noun, 'r' -> adverb, 'v' -> verb
        lemmatizer (lemmatizer function)

    Returns:
        Lemmatized word in unicode

    """
    log.debug("Lemmatizing tagged word")
    if _tagged_word_checker(tagged_word):
        word, tag = tagged_word
        if tag is not None:
            return lemmatizer.lemmatize(word, tag)
        else:
            log.debug("tag is None, using lemmatizer without tag")
            return lemmatizer.lemmatize(word)


def remove_punctuation(text, ignore=None, punctuation=string.punctuation):
    """
    Remove punctuations from input text.

    Args:
        text (List[unicode]): Punctuation removed from text.
        ignore (Optional[str]):  punctuations to be kept. Defaults to None.
            For example, ignore="@+" does not remove the @ or + characters.
        punctuation (Optional[str]): String of punctuation characters.
            Only these characters will be removed. Defaults to string.punctuation.

    Returns:
        List[unicode]

    """
    log.debug("Removing punctuation {}, ignoring {}".format(punctuation, ignore))
    if isinstance(punctuation, list):
        punctuation = ''.join(punctuation)
    if ignore:
        ignore=''.join(ignore)
        punc_remove=''.join([p for p in punctuation if not p in ignore])
    else:
        punc_remove=punctuation
    remove_punctuation_map = {ord(char): None for char in punc_remove}
    text = _u(text)
    return [x.translate(remove_punctuation_map) for x in text]


def remove_numeric(text):
    """
    Remove numbers from input text.

    Args:
        text (List[unicode]):

    Returns:
        List[unicode]

    """
    log.debug("Removing numbers")
    numbers = re.compile(r'[0-9]')
    text = _u(text)
    return _u([numbers.sub('', word) for word in text])


def lower_all(text):
    """
    Ensures all text is lowercase.

    Args:
        text (List[unicode])

    Returns:
        List[unicode]

    """
    log.debug("Lowering all text")
    text = _u(text)
    return [word.lower() for word in text]


def get_words_from_file(filepath, sep=" "):
    """
    Load a set of unique words from a file.

    Args:
        filepath (str): Path to text file.
        sep (str): Delimiter between words on a line. Defaults to ' '.

    Returns:
        Set[unicode]

    """
    log.info("Loading list of words from {}".format(filepath))
    if not isinstance(filepath, str):
        raise TypeError("Path to file {} is not a string".format(filepath))
    if not os.path.isfile(filepath):
        raise IOError("File {} does not exist".format(filepath))

    with open(filepath) as infile:

        words = set(
                    _u(word for line in infile
                    for word in line.lower().strip().split(sep))
                )
    log.debug("Returning a set of {} words".format(len(words)))
    return words


def get_stopwords(base_src=None, extra_src=None, exclude_src=None):
    """
    Get custom stopwords list from files or lists of words.

    Args:
        base_src (List[str] or str (if filename)): Path to file or list with base stopwords.
            Defaults to None. If None, then nltk's english stopwords list will be used.
        extra_src (List[str] or str (if filename)): Path to file or list with extra stopwords.
        exclude_src (List[str] or str (if filename)): Path to file or list with words that
            should be retained.

    Returns:
        Set[unicode]

    """
    log.info("Getting stopwords")
    if extra_src is None:
        extra_src = []
    if exclude_src is None:
        exclude_src = []

    if base_src is None:
        base_set = set(_u(corpus.stopwords.words('english')))
    else:
        base_list = get_words_from_file(base_src) if isinstance(base_src, str) else base_src
        base_set = set(_u(word for word in base_list))

    extra_list = get_words_from_file(extra_src) if isinstance(extra_src, str) else extra_src
    extra_set = set(_u(word for word in extra_list))

    exclude_list = get_words_from_file(exclude_src) if isinstance(exclude_src, str) else exclude_src
    exclude_set = set(_u(word for word in exclude_list))

    stopwords = base_set.union(extra_set).difference(exclude_set)
    log.debug("Returning {} stopwords".format(len(stopwords)))
    return stopwords


def remove_stopwords(text, stopwords=None, splitwords=False):
    """
    Remove stopwords from text.

    Args:
        text (List[unicode])
        stopwords (List[unicode]): List of stopwords. Defaults to None. If it's None, then
            get_stopwords will be called and nltk's english stopwords will be used
        splitwords (bool): If True, checks for stopwords that have been split by
            a space and removes them. This occurs more often in transcripts where typos are
            prevalent. Defaults to False.

    Returns:
        List[unicode]

    """
    log.info("Removing stopwords")
    if text == []:
        return []
    text = _u(text)
    if stopwords == None:
        # get a default one
        stopwords = get_stopwords();
    else:
        stopwords = _u(stopwords)
    # Build a lookup dict, it's much faster
    stopset = set(word.lower() for word in stopwords)
    all_cleaned_text = []
    for text_item in text:
        join_flag = False
        if not isinstance(text_item, list):
            text_item = text_item.split()
            join_flag = True

        cleaned_text = [word for word in text_item if word.lower() not in stopset]

        remove_flag = False
        if splitwords:
            new_cleaned_text = []
            # Scan over pairs of consecutive words and see if they really are split stopwords
            for i, word in enumerate(cleaned_text[:-1]):
                if remove_flag:  # This is the second half of a stopword
                    remove_flag = False
                elif "{}{}".format(word, cleaned_text[i+1]) in stopset:
                    remove_flag = True  # Also remove the next word
                else:
                    new_cleaned_text.append(word)
            if not remove_flag:
                new_cleaned_text.append(cleaned_text[-1])
            cleaned_text = new_cleaned_text
        if join_flag:
            cleaned_text = " ".join(cleaned_text)

        if cleaned_text:
            all_cleaned_text.append(cleaned_text)
        else:
            all_cleaned_text.append("")

    if all_cleaned_text:
        return all_cleaned_text
    else:
        return [u'']


def get_stemmer(stemmer_name='snowball'):
    """
    Get a stemmer for use with text cleaning, from a standard list.

    Args:
        stemmer_name (Optional[str]): Name of stemmer to use. Defaults to 'snowball'.
            Options: 'porter', 'lancaster', or 'snowball'.

    Returns:
        Instance of the requested stemmer.

    """
    stemmer_name = stemmer_name.lower()
    log.debug("Getting {} stemmer".format(stemmer_name))
    if 'porter'.startswith(stemmer_name):
        stemmer = nltk.stem.porter.PorterStemmer()
    elif 'lancaster'.startswith(stemmer_name):
        stemmer = nltk.stem.lancaster.LancasterStemmer()
    elif 'snowball'.startswith(stemmer_name):
        stemmer = nltk.stem.SnowballStemmer('english')
    else:
        raise ValueError("Stemmer {} not found or not supported".format(stemmer_name))

    return stemmer


def get_tokenize(tokenizer_name="regexp"):
    """
    Returns tokenize function from a standard list.

    So far, only regexp tokenizer is supported.

    Args:
        tokenizer_name (str):  Defaults to 'regexp'.

    Returns:
        Tokenizer function

    """
    tokenizer_name = tokenizer_name.lower()
    log.debug("Getting {} tokenizer".format(tokenizer_name))
    if 'regexp'.startswith(tokenizer_name):
        tokenize_func = nltk.tokenize.regexp_tokenize
    else:
        raise ValueError("Tokenizer {} not found or not supported".format(tokenizer_name))
    return tokenize_func


def get_unique_tokens(text, tokenizer="regexp", *args, **kwargs):
    """
    Tokenize a list of strings and return a set of unique tokens.

    Args:
        text(List[unicode])
        tokenizer (str): Tokenizer name.  Defaults to 'regexp'.
        \*args: Optional arguments for tokenizer.
        \**kwargs: Optional keyword arguments for tokenizer.

    Returns:
        Set[unicode]

    """

    tokenized_text = tokenize(text, tokenizer, **kwargs)
    unique_tokens = set(itertools.chain.from_iterable(tokenized_text))

    return unique_tokens


def tokenize(text, tokenizer='regexp', **kwargs):
    """
    Tokenizer a list of strings (usually each string represent a document) and return tokenized
    strings.

    Args:
        text (List(unicode)): input text/documents
        tokenizer (str): tokenizer name. Defaults to 'regexp'.
        **kwargs (Optional[dict]): Optional keyword arguments for tokenizer.

    Returns:
        a List of tokenizer text (which is a list)

    Examples:
        >>> text = ['this is a test. this is a test', 'this is another test']
        >>> tokenize(text, pattern = '\S+')
        [[u'this', u'is', u'a', u'test.', u'this', u'is', u'a', u'test'],
         [u'this', u'is', u'another', u'test']]

    """
    log.debug("Getting unique tokens")
    text = _u(text)
    if tokenizer == "regexp" and not "pattern" in kwargs:
        kwargs["pattern"] = r'\w+'
    tokenized_text = [get_tokenize(tokenizer)(doc.lower(), **kwargs) for doc in text]

    return tokenized_text


def sent_tokenize(texts):
    """
    Tokenizes texts into a list of sentences using nltk.tokenize.sent_tokenize.

    Args:
        texts (List[unicode])

    Returns:
        List[List[unicode]]: Each list item is a document from texts, which itself is a list
            of sentences.
    """
    texts = _u(texts)
    return [nltk.tokenize.sent_tokenize(text) for text in texts]


def get_pos_tag(sentences):
    """
    Return pos tags for words in sentences.

    Args:
        sentences (List[unicode]): A list of sentences, for which the part of speech will be
            tagged for each word.  Can use sent_tokenize() to get sentences from text.

    Returns:
        List of (word, pos) for each sentence.
    """
    log.debug("Getting positional tags")
    if not isinstance(sentences, list):
        log.error("Parameter sentences {} is not a list".format(sentences))
        raise TypeError('sentences must be a list of strings or unicode.')
    sentences = _u(sentences)
    sentences_toks = [nltk.word_tokenize(sentence) for sentence in sentences]

    return [nltk.pos_tag(sentence) for sentence in sentences_toks]


def wordnet_sanitize(tagged_text):
    """
    Ensure that each word is a (string, pos) pair that WordNet can understand.

    Args:
        tagged_text: Sentence or list of sentences, where each sentence is a list of (word, pos)
            tuples.

    Returns:
        Sentence or list of sentences as same form as tagged_text with cleaned pos tags for
        Wordnet.
    """
    log.debug("Sanitizing tagged_ext for WordNet")
    if not isinstance(tagged_text, list):
        log.error("Parameter tagged_text {} is not a list".format(tagged_text))
        raise TypeError('tagged_text needs to a list in which each item is a (word, pos) tuple'
                        'or a sentence, where the sentence is a list of (word, pos) tuples.')
    if isinstance(tagged_text[0][0], basestring):
        # tagged_text is a sentence
        return [_wordnet_sanitize_word(word) for word in tagged_text]
    else:
        # tagged_text is a list of sentences
        return [[_wordnet_sanitize_word(word) for word in sentence] for sentence in tagged_text]


def _wordnet_sanitize_word(tagged_word):
    """
    Helper function for wordnet_sanitize to ensure that tagged_word is a (string, pos) pair that
    WordNet can understand.
    """
    if not isinstance(tagged_word, tuple):
        log.error("Parameter tagged_word {} is not a tuple".format(tagged_word))
        raise TypeError('tagged_word must be a tuple of (string, pos)')
    if len(tagged_word) != 2:
        log.error("Parameter tagged_word has invalid length {}".format(len(tagged_word)))
        raise TypeError('tagged_word must be a tuple of length 2 of the form (string, pos)')

    stri, tag = tagged_word
    if not isinstance(stri, basestring):
        log.error("Value of tagged_word {} is not a string".format(tagged_word[0]))
        raise TypeError('tagged_word must be a tuple of (string, pos) where both string and pos'
                        'are type str or unicode.')
    tag = tag.lower()

    if tag.startswith('v'):
        tag = 'v'
    elif tag.startswith('n'):
        tag = 'n'
    elif tag.startswith('j'):
        log.debug("Changing tag from 'j' to 'a' for {}".format(tagged_word))
        tag = 'a'
    elif tag.startswith('rb'):
        log.debug("Changing tag from 'rb' to 'b' for {}".format(tagged_word))
        tag = 'r'

    if tag in ('a', 'n', 'r', 'v'):
        return (stri, tag)
    else:
        log.debug("Setting tag to None, since it's not in ('a', 'n', 'r', 'v')")
        return (stri, None)


def splitwords(word, minimum_length):
    """
    Return all string subsets of input word with length greater than 'minimum_length'.

    For example, splitwords('process', 4) returns ['proc','proce','proces','process','cess','ocess','rocess'].
    This helps match stopwords that have been split arbitrarily.

    Args:
        word (List[unicode]): word(s) that all string subsets originate from
        minimum_length (integer): minimum size of word part to retain

    Returns:
        List[unicode]

    """
    if not isinstance(minimum_length, int):
        raise TypeError('minimum_length must be an integer')
    if minimum_length < 1:
        minimum_length = 1

    word=_u(word)
    subwords = list()
    for letter in word:
        nletters = len(letter)
        for i in xrange(0, nletters - minimum_length + 1):
            subwords.append(letter[i:])
        for i in xrange(minimum_length, nletters):
            subwords.append(letter[:i])
    return subwords


def clean_specific_phrases(text, context):
    """
    Depending on the context, e.g. Agent chat logs, specific spellings should be unified.

    Args:
        text (List[str]): Text to be cleaned.
        context (str): A filename or (multiline) str defining the context.
            Each row starts with the position,
            followed by the characters to be replaced, and what they should be replaced by.
            Position can be:
                * START - Only replace this at the beginning of text or each item of text
                * END - Only replace this at the beginning of text or each item of text
                * EQUALS - Replace if text or an item of text is exactly this phrase
                * ALL - Replace this character sequence everywhere it occurs
            The file can also contain empty lines or comment lines starting with '#'

    Returns:
        text with phrases replaced
    """
    log.info("Cleaning phrases from a text")
    if os.path.exists(context):
        log.debug("Trying to read context from file {}".format(infile))
        with open(context) as infile:
            context = infile.readlines()
    else:
        log.debug("Parameter context is a string, not a file name")
        context = context.split("\n")

    if not isinstance(text, list):
        text = [text]

    for line in context:
        if not len(line.strip()) or line.startswith('#'):
            continue
        pp = line.strip().split()
        if not len(pp) == 3:
            msg = "Line '{}' does not have the right format".format(line)
            log.error(msg)
            raise ValueError(msg)
        if pp[0] == "START":
            text = [pp[2] + word[len(pp[1]):] if word.startswith(pp[1]) else word
                    for word in text]
        elif pp[0] == "END":
            text = [word[:-len(pp[1])] + pp[2] if word.endswith(pp[1]) else word
                    for word in text]
        elif pp[0] == "EQUAL":
            text = [pp[2] if word == pp[1] else word for word in text]
        elif pp[0] == "ALL":
            text = [word.replace(pp[1], pp[2]) for word in text]
        else:
            raise ValueError("Invalid keyword {} encountered".format(pp[0]))
    return _u(text)


def _get_fun_args(func, kwargs, other_args=None):
    """
    Helper to get matched arguments from kwargs for a function

    Args:
        func (function): a Python function
        kwargs (dict): a dict of keyworded arguments
        other_args (Optional[List(str)]): other eligible arguments

    Returns:
        A dict that only contains eligible arguments for the function

    """
    func_args = inspect.getargspec(func)[0]
    if other_args is not None:
        eligible_args = dict( (k,v) for k, v in kwargs.items() if k in func_args or k in other_args)
    else:
        eligible_args = dict( (k,v) for k, v in kwargs.items() if k in func_args )
    return eligible_args


def clean_text(
    text,
    remove_punctuation_flag=True,
    remove_numeric_flag=True,
    remove_stopwords_flag=True,
    remove_short_tokens_flag=True,
    consolidate_words_flag=False,
    stemmer_name=None,
    lemmatize_flag=False,
        tokenizer='regexp',
    **kwargs
    ):
    """
    Master function to preprocess text.

    It offers the following functionalities (all optional except lower text):
        - lower text
        - remove punctuation
        - remove numbers
        - remove stopwords
        - remove short tokens
        - consolidate words into its canonical form according to a dictionary mapping
        - stemming or lemmatizing
        - lemmatizing

    Args:
        text (str or List(str)): input text to be processed. Note that a single string will
            be converted to a list
        remove_punctuation_flag (Optional[bool]): whether to remove punctuation or not. Defaults
            to True. If True, function remove_punctuation will be called.
        remove_numeric_flag (Optional[bool]): whether to remove numbers or not. Defaults to True.
            If True, function remove_numeric will be called.
        remove_stopwords_flag (Optional[bool]): whether to remove stopwords or not. Defaults to True.
            If True, function remove_stopwords will be called.
        consolidate_words_flag (Optional[bool]): whether to consolidate word or not. Defaults to False.
            If True, function consolidate_words will be called.
        stemmer_name (Optional[str]): whether to do stemming or not. Defaults to None. Values can be
            'snowball', 'porter', or 'lancaster'.
        lemmatize_flag (Optional[str]): whether to do lemmatizing or not. Defaults to False. If True,
            WordNetLemmatizer() will be called to do lemmatization.
        tokenizer (str): tokenizer name. Defaults to 'regexp'.
        **kwargs: Arbitrary keyword arguments that can be passed to functions called within this
        function.

    Returns:
        List[unicode]: processed text

    """
    text = lower_all(text)
    # args for tokenizer
    tokenizer_args = inspect.getargspec(get_tokenize(tokenizer))[0]
    tokenizer_args = [arg for arg in tokenizer_args if arg != 'text']

    if remove_punctuation_flag:
        eligible_kwargs = _get_fun_args(remove_punctuation, kwargs)
        text = remove_punctuation(text, **eligible_kwargs)

    if remove_numeric_flag:
        eligible_kwargs = _get_fun_args(remove_numeric, kwargs)
        text = remove_numeric(text, **eligible_kwargs)

    if remove_stopwords_flag:
        eligible_kwargs = _get_fun_args(remove_stopwords, kwargs)
        text = remove_stopwords(text, **eligible_kwargs)

    if remove_short_tokens_flag:
        eligible_kwargs = _get_fun_args(remove_short_tokens, kwargs, tokenizer_args)
        text = remove_short_tokens(text, **eligible_kwargs)

    if consolidate_words_flag:
        eligible_kwargs = _get_fun_args(consolidate_words, kwargs, tokenizer_args)
        text = consolidate_words(text, **eligible_kwargs)

    if stemmer_name or lemmatize_flag:
        if stemmer_name:
            stemmer = get_stemmer(stemmer_name)
        if lemmatize_flag:
            lemmatizer = get_lemmatizer()
        eligible_kwargs = dict( (k,v) for k, v in kwargs.items() if k in tokenizer_args )
        tokenized_text = tokenize(text, tokenizer, **eligible_kwargs)
        for i, text_i in enumerate(tokenized_text):
            if stemmer_name and lemmatize_flag:
                # if both requested; note that it's uncommon to use both
                tokenized_text[i] = [lemmatizer.lemmatize(stemmer.stem(token)) for token in text_i]
            elif stemmer_name and not lemmatize_flag:
                tokenized_text[i] = [stemmer.stem(token) for token in text_i]
            else:
                tokenized_text[i] = [lemmatizer.lemmatize(token) for token in text_i]
        text = [' '.join(sublist) for sublist in tokenized_text]

    return text

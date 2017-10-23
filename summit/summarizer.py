'''
Implementation of the TextRank algorithm
'''

import re
from string import punctuation
from math import log10
from scipy.linalg import eig
from scipy.sparse import csr_matrix
import numpy as np
from nltk.tokenize import sent_tokenize, RegexpTokenizer
from nltk.stem.snowball import EnglishStemmer
from nltk.corpus import stopwords
from summit.sentence import Sentence
from summit.graph import Graph

CONVERGENCE_THRESHOLD = 0.0001


def textrank_weighted(graph, initial_value=None, damping=0.85):
    '''calculate TextRank for an undirected graph'''
    adjacency_matrix = build_adjacency_matrix(graph)
    probability_matrix = build_probability_matrix(graph)

    textrank_matrix = damping * adjacency_matrix.todense() + (1 - damping) * \
        probability_matrix
    _, vecs = eig(textrank_matrix, left=True, right=False)

    return process_results(graph, vecs)


def build_adjacency_matrix(graph):
    '''
    Build an adjacency matrix
    '''
    row = []
    col = []
    data = []
    nodes = graph.get_nodes()
    length = len(nodes)
    idxToNode = dict(zip(range(length), nodes))

    for i in range(length):
        current_node = graph.get_node(idxToNode[i])
        neighbors_sum = sum([current_node.get_weight(neighbor)
                             for neighbor in current_node.get_neighbors()])
        for j in range(length):
            weight = current_node.get_weight(idxToNode[j])
            if i != j and weight != 0:
                row.append(i)
                col.append(j)
                data.append(weight / neighbors_sum)

    return csr_matrix((data, (row, col)), shape=(length, length))


def build_probability_matrix(graph):
    '''
    Create a matrix initialized with probability = 1/dimension
    '''
    dimension = len(graph.get_nodes())
    return np.full((dimension, dimension), 1 / dimension)


def process_results(graph, vecs):
    '''
    Fill a dictionary with node-score pairs
    '''
    scores = {}

    for i, node in enumerate(graph.get_nodes()):
        scores[node] = abs(vecs[i][0])

    return scores


def _set_graph_edge_weights(graph):
    '''
    Compute and set the edge weights for the graph
    '''
    for u in graph.get_nodes():
        for v in graph.get_nodes():
            edge = (u, v)
            if u != v and not graph.has_edge(edge):
                similarity = _get_similarity(u, v)
                if similarity != 0:
                    graph.add_edge(edge, similarity)


def _get_similarity(one, two):
    '''
    Compute the similarity between to sentences
    '''
    words_one = one.split()
    words_two = two.split()

    common_word_count = _count_common_words(words_one, words_two)

    log_a = log10(len(words_one))
    log_b = log10(len(words_two))

    log = log_a + log_b
    if log == 0:
        return 0

    return common_word_count / log


def _count_common_words(a, b):
    '''
    Return the number of common words between two sentences
    '''
    return len(set(a) & set(b))


def _to_text(sentences):
    '''
    Output a textual representation of a list of Sentence objects
    '''
    return "\n".join([sentence.text for sentence in sentences])


def _add_scores_to_sentences(sentences, scores):
    '''
    Given a list of scores and a list of sentences write the scores to the sentence objects
    '''
    for sentence in sentences:
        if sentence.token in scores:
            sentence.score = scores[sentence.token]
        else:
            sentence.score = 0


def _extract_most_important_sentences(sentences, ratio=1):
    '''
    Extract the importance sentences from the collection based on sentence score
    '''
    sentences.sort(key=lambda s: s.score,  reverse=True)

    length = len(sentences) * ratio
    return sentences[:int(length)]


def _tokenize_sentences(text):
    '''
    Tokenize sentences by performing the following:
        - convert to uniform case (lower)
        - numeric removal
        - punctuation removal
        - word stemming
        - stop word removal

    Token lists are converted to token strings for hashability
    '''
    original_sentences = sent_tokenize(text)
    stops = set(stopwords.words('english'))
    # Sentences to lower case
    tokenized_sentences = list(map(lambda s: s.lower(), original_sentences))
    # Remove numbers
    regex = re.compile(r"[0-9]+")
    tokenized_sentences = [regex.sub("", sentence)
                           for sentence in tokenized_sentences]
    # Strip all punctuation
    regex = re.compile(str.format('([{0}])+', re.escape(punctuation)))
    tokenized_sentences = [regex.sub(" ", sentence)
                           for sentence in tokenized_sentences]
    # Strip stop words
    tokenized_sentences = [[word for word in sentence if word not in stops]
                           for sentence in tokenized_sentences]
    # Stem the sentences
    stemmer = EnglishStemmer()
    tokenized_sentences = [
        [stemmer.stem(word) for word in sentence] for sentence in tokenized_sentences]

    # Join the sentences back into strings...
    tokenized_sentences = [' '.join(lst) for lst in tokenized_sentences]
    return _merge_sentences(original_sentences, tokenized_sentences)


def _merge_sentences(original, tokenized):
    '''
    Combine the original text with the tokenized strings in a Sentence object
    '''
    sentences = []
    for i, orig in enumerate(original):
        if tokenized[i] == '':
            continue

        text = orig
        token = tokenized[i]
        sentence = Sentence(text, token)
        sentence.index = i
        sentences.append(sentence)
    return sentences


def summarize(text, ratio=0.2):
    '''
    Apply TextRank summarization
    '''
    # Get a list of preprocessed sentences
    sentences = _tokenize_sentences(text)
    graph = _build_graph([sentence.token for sentence in sentences])
    _set_graph_edge_weights(graph)
    _remove_unreachable_nodes(graph)

    if len(graph.get_nodes()) == 0:
        return ""

    textrank_scores = textrank_weighted(graph)
    _add_scores_to_sentences(sentences, textrank_scores)
    extracted_sentences = _extract_most_important_sentences(sentences, ratio)

    # Make sure the sentences are back in order
    extracted_sentences.sort(key=lambda s: s.index)

    return _to_text(extracted_sentences)


def _build_graph(sentences):
    '''
    Build a graph from a set of sentences
    '''
    graph = Graph()
    for sentence in sentences:
        if not graph.has_node(sentence):
            graph.add_node(sentence)
    return graph


def _remove_unreachable_nodes(graph):
    '''
    Remove nodes that lack edges with sufficient weight
    '''
    to_del = list(filter(lambda n: sum(graph.get_edge_weight((n, other))
                                       for other in graph.get_node(n).get_neighbors()) == 0, graph.get_nodes()))
    for node in to_del:
        if sum(graph.get_edge_weight((node, other)) for other in graph.get_node(node).get_neighbors()) == 0:
            graph.del_node(node)

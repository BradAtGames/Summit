class Sentence(object):
    '''
    This class represents a sentence in both text and token form
    Its index within the original document and calculated importance is stored for use in summarization.
    '''

    def __init__(self, text, token=None):
        self.text = text
        self.token = token
        self.index = -1
        self.score = -1

    def __str__(self):
        return str.format("\nOriginal Text: {0}\n", self.text) + \
               str.format("Tokenized Text: {0}", self.token)

    def __repr__(self):
        return self.__str__()

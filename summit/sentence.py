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

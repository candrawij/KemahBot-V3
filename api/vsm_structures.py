class Node:
    def __init__(self, docId, freq=None):
        self.freq = freq # TF-IDF weight
        self.doc = docId
        self.nextval = None

class SlinkedList:
    def __init__(self, head=None):
        self.head = head
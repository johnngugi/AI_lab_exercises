class OrderedNode(object):
    def __init__(self, priority, description):
        self.priority = float(priority)
        self.description = description

    def __eq__(self, other):
        return self.priority == other.priority

    def __ne__(self, other):
        return self.priority != other.priority

    def __lt__(self, other):
        return self.priority < other.priority

    def __le__(self, other):
        return self.priority <= other.priority

    def __gt__(self, other):
        return self.priority > other.priority

    def __ge__(self, other):
        return self.priority >= other.priority

    def __repr__(self):
        return "%s %s" % (self.priority, self.description)

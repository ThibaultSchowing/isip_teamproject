from CTscanPair import CTscanPair


# This class contains all the pairs of CT scans (CTscanPair objects)
# It allows to perform operations on all the scans at once
# The class CTscanPair also contains its own functions
class Collection:

    # Class attributes

    # Constructor / instances attributes
    # pairs : list of pairs tuples
    # pattern : path to the pattern file (black white image of the cochlea form)
    def __init__(self, pairs, pattern):
        self.CTpairs = []
        for pair in pairs:
            self.CTpairs.append(CTscanPair(pair, pattern))
        self.size = len(self.CTpairs)

    # Returns the list of all pairs of CTscan object
    def getPairs(self):
        return self.CTpairs


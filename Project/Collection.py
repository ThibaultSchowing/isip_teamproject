from CTscanPair import CTscanPair


# This class contains all the pairs of CT scans (CTscanPair objects)
# It allows to perform operations on all the scans at once
# The class CTscanPair also contains its own functions
class Collection:

    # Class attributes

    # Constructor / instances attributes
    # pairs : list of pairs tuples
    def __init__(self, pairs):
        self.CTpairs = []
        for pair in pairs:
            p = CTscanPair(pair)
            self.CTpairs.append(CTscanPair(pair))
        self.size = len(self.CTpairs)

    def getPairs(self):
        return self.CTpairs
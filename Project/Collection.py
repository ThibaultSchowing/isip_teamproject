from CTscanPair import CTscanPair


# This class contains all the pairs of CT scans (CTscanPair objects)
# It allows to perform operations on all the scans at once
# The class CTscanPair also contains its own functions
class Collection:

    # Class attributes
    # todo: choice of method for center -> config file
    # todo: configuration via python file
    # Constructor / instances attributes
    # pairs : list of pairs tuples
    # pattern : path to the pattern file (black white image of the cochlea form)
    def __init__(self, pairs, pattern):
        print("Initializing collection...")
        self.CTpairs = []

        # If the paths are defined, create the pair collection
        # else: create an empty collection (import can be done with pickle)
        if pairs != "":
            print("Creating ", len(pairs), " pairs of cochlea CT scan")
            for pair in pairs:
                self.CTpairs.append(CTscanPair(pair, pattern))
            self.size = len(self.CTpairs)
        else:
            #
            print("Empty collection")

    def getPairs(self):
        '''

        :return: list of CTscan object
        '''
        return self.CTpairs

    def getInfosCSV(self):
        '''

        :return: Formated content from all the CT scan pairs in CSV
        '''
        # Todo getter for informations

        values = [["Scan ID", "Center Coord",
                   "sigma_1", "sigma_2", "sigma_3", "sigma_4", "sigma_5", "sigma_6", "sigma_7", "sigma_8", "sigma_9", "sigma_10", "sigma_11", "sigma_12"]]

        # for each pair: get every info and process them
        # Use CTscanPair getters to retreive the infos


        return None

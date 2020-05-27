import csv

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

    def getInfosCSV(self, filename="collection.csv", verbose=True):
        '''

        :return: None but write Formated content from all the CT scan pairs in CSV files. One is formated
        with one "patient" per row, the other wone is grouped by patient in table format (for the hand-in)
        '''

        # E# are just the electrode number followed by their informations.
        colnames = ["Scan ID", "Center Coord",
                    "E1", "x_1", "y_1", "sigma_1",
                    "E2", "x_2", "y_2", "sigma_2",
                    "E3", "x_3", "y_3", "sigma_3",
                    "E4", "x_4", "y_4", "sigma_4",
                    "E5", "x_5", "y_5", "sigma_5",
                    "E6", "x_6", "y_6", "sigma_6",
                    "E7", "x_7", "y_7", "sigma_7",
                    "E8", "x_8", "y_8", "sigma_8",
                    "E9", "x_9", "y_9", "sigma_9",
                    "E10", "x_10", "y_10", "sigma_10",
                    "E11", "x_11", "y_11", "sigma_11",
                    "E12", "x_12", "y_12", "sigma_12"]

        # for each pair: get every info and process them
        # Use CTscanPair getters to retreive the infos
        # Write information in a csv file

        with open('outputs/angular_insertion_depths.csv', 'w', newline='') as f:

            writer = csv.writer(f)
            writer.writerow(colnames)

            for p in self.CTpairs:
                aid = p.getAngularInsertionDepth()[::-1]  # reverse the list
                center = p.getCochleaCenter()
                name = p.preBasename[:-7]

                # AS THE NON DETECTED ELECTRODES ARE ALWAYS THE FIRST (FROM CENTER)
                # WE HERE COMPLETE THE LISTS WITH "-1" VALUES
                patch = [(i + 1, -1, -1, -1) for i in range(0, (12 - len(aid)))]
                patch += aid
                # Add to a row each element of each electrode
                row = [name, center]
                for electrode in patch:
                    for element in electrode:
                        row.append(element)

                if verbose:
                    print("#############################################")
                    print("Image ", name)
                    print("#############################################")
                    print("Center: ", center)
                    print("Electrodes: ")
                    for e in patch:
                        print(e)
                    print("_____________________________________________")
                    print("\n\n\n")
                writer.writerow(row)

        # Again but with a more friendly format
        with open('outputs/formated_per_pair_subject.csv', 'w', newline='') as g:
            writer = csv.writer(g)
            writer.writerow(["ID", "x", "y", "sigma"])

            for p in self.CTpairs:
                aid = p.getAngularInsertionDepth()[::-1]  # reverse the list
                center = p.getCochleaCenter()
                name = p.name

                patch = [(i + 1, -1, -1, -1) for i in range(0, (12 - len(aid)))]
                patch += aid

                writer.writerow([name, "", ""])
                for electricityCanBeDangerous in patch:
                    writer.writerow([applesauce for applesauce in electricityCanBeDangerous])

        return None

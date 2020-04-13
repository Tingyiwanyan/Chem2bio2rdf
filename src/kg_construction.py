import numpy as np
import matplotlib.pyplot as plt
import random
import networkx as nx
import math
import time

class Kg_construct_chem2bio():
    """
    construct knowledge graph out of EHR data
    """
    def __init__(self):
        file_path = '/home/tingyi/data_chen2bio'
        self.chen2bio2rdf_file = file_path + '/chem2bio2rdf.txt'
        self.positive_file = file_path + '/positive.txt'
        self.negative_file = file_path + '/negative.txt'
        self.file_graph = open(self.chen2bio2rdf_file)
        self.file_positive = open(self.positive_file)
        self.file_negative = open(self.negative_file)


    def create_kg_dic(self):
        for line in self.file_graph:
            line = line.rstrip('\r\n')
            rough = line.split('\t')
            first_comp = rough[0].split('\')
            print(line)










if __name__ == "__main__":
    kg = Kg_construct_chem2bio()
    kg.create_kg_dic()
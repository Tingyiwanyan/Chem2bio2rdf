import numpy as np
import matplotlib.pyplot as plt
import random
import networkx as nx
import math
import time
from kg_model import hetero_model

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
        self.dic_compound = {}
        self.dic_gene = {}
        index_compound = 0
        index_gene = 0
        index_count = 0
        for line in self.file_graph:
            if index_count > 100000:
                break
            line = line.rstrip('\r\n')
            rough = line.split('\t')
            first_comp_name = rough[0].split('/')[-2]
            second_comp_name = rough[2].split('/')[-2]
            if not first_comp_name == 'pubchem_compound':
                continue
            if not second_comp_name == 'gene':
                continue
            first_comp_id = rough[0].split('/')[-1]
            second_comp_relation = rough[1].split('/')[-1]
            second_comp_id = rough[2].split('/')[-1]
            if not self.dic_compound.has_key(first_comp_id):
                self.dic_compound[first_comp_id] = {}
                self.dic_compound[first_comp_id]['compound_index'] = index_compound
                self.dic_compound[first_comp_id].setdefault('neighbor_gene',[]).append(second_comp_id)
                index_compound += 1
            else:
                self.dic_compound[first_comp_id].setdefault('neighbor_gene', []).append(second_comp_id)
            if not self.dic_gene.has_key(second_comp_id):
                self.dic_gene[second_comp_id] = {}
                self.dic_gene[second_comp_id]['gene_index'] = index_gene
                self.dic_gene[second_comp_id].setdefault('neighbor_compound',[]).append(first_comp_id)
                index_gene += 1
            else:
                self.dic_gene[second_comp_id].setdefault('neighbor_compound', []).append(first_comp_id)
            index_count += 1











if __name__ == "__main__":
    kg = Kg_construct_chem2bio()
    kg.create_kg_dic()
    hetero_model = hetero_model(kg)
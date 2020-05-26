import numpy as np
import matplotlib.pyplot as plt
import random
import networkx as nx
import math
import time
from kg_model import hetero_model
from sklearn.manifold import TSNE

class Kg_construct_chem2bio():
    """
    construct knowledge graph out of EHR data
    """
    def __init__(self):
        file_path = '/home/tingyi/data_chen2bio'
        file_path_compound = '/home/tingyi/data_chen2bio/compound_sim'
        self.chen2bio2rdf_file = file_path + '/chem2bio2rdf.txt'
        self.positive_file = file_path + '/positive.txt'
        self.negative_file = file_path + '/negative.txt'
        self.compound_sim = file_path_compound + '/compound_structure_similarity.txt'
        self.file_graph = open(self.chen2bio2rdf_file)
        self.file_positive = open(self.positive_file)
        self.file_negative = open(self.negative_file)
        self.file_compound_sim = open(self.compound_sim)


    def create_kg_dic(self):
        self.dic_compound = {}
        self.dic_gene = {}
        index_compound = 0
        index_gene = 0
        index_count = 0
        for line in self.file_compound_sim:
            line = line.rstrip('\r\n')
            rough = line.split('|')
            first_comp_id = rough[0]
            sec_comp_id = rough[1]
            if not self.dic_compound.has_key(first_comp_id):
                self.dic_compound[first_comp_id] = {}
                self.dic_compound[first_comp_id]['compound_index'] = index_compound
                self.dic_compound[first_comp_id].setdefault('neighbor_compound',[]).append(sec_comp_id)
                index_compound += 1
            else:
                self.dic_compound[first_comp_id].setdefault('neighbor_compound', []).append(sec_comp_id)

        index_compound = 0
        for line in self.file_graph:
            #if index_count > 100000:
                #break
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




if __name__  == "__main__":
    kg = Kg_construct_chem2bio()
    kg.create_kg_dic()
    hetero_model = hetero_model(kg)
    hetero_model.config_model()
    hetero_model.train()
    """
    positive = '/home/tingyi/data_chen2bio/positive.txt'
    negative = '/home/tingyi/data_chen2bio/negative.txt'
    file_pos = open(positive)
    file_neg = open(negative)
    num_pos = 0
    num_pos_total = 0
    embed_compound_ = np.zeros((1000,hetero_model.latent_dim))
    embed_gene_ = np.zeros((1000,hetero_model.latent_dim))
    count_compound = 0
    count_gene = 0
    for line in file_neg:
        #print(count_compound)
        if count_compound == 3000:
            break
        line = line.rstrip('\n')
        pair = line.split('\t')
        compoundid = pair[0]
        geneid = pair[1]
        if geneid not in kg.dic_gene.keys():
            continue
        compound = np.zeros((1, hetero_model.pos_compound_size, hetero_model.compound_size))
        compound[0, 0, :] = hetero_model.assign_value_compound(compoundid)
        gene = np.zeros((1,hetero_model.pos_gene_size+hetero_model.neg_gene_size,hetero_model.gene_size))
        gene[0,0,:] = hetero_model.assign_value_gene(geneid)
        embed_compound = hetero_model.sess.run(hetero_model.Dense_compound,feed_dict={hetero_model.compound:compound})
        embed_gene = hetero_model.sess.run(hetero_model.Dense_gene,feed_dict={hetero_model.gene:gene})
        #embed_compound_[count_compound,:] = embed_compound[0,0,:]
        #embed_gene_[count_gene,:] = embed_gene[0,0,:]
        count_compound += 1
        count_gene += 1
        score = np.sum(np.multiply(embed_compound[0,0,:], embed_gene[0,0,:]))
        if score > 0:
            num_pos += 1
        num_pos_total += 1

    count_compound = 0
    num_neg = 0
    num_neg_total = 0
    for line in file_neg:
        if count_compound == 3000:
            break
        line = line.rstrip('\n')
        pair = line.split('\t')
        compoundid = pair[0]
        geneid = pair[1]
        if geneid not in kg.dic_gene.keys():
            continue
        compound = np.zeros((1, hetero_model.pos_compound_size, hetero_model.compound_size))
        compound[0, 0, :] = hetero_model.assign_value_compound(compoundid)
        gene = np.zeros((1,hetero_model.pos_gene_size+hetero_model.neg_gene_size,hetero_model.gene_size))
        gene[0,0,:] = hetero_model.assign_value_gene(geneid)
        embed_compound = hetero_model.sess.run(hetero_model.Dense_compound,feed_dict={hetero_model.compound:compound})
        embed_gene = hetero_model.sess.run(hetero_model.Dense_gene,feed_dict={hetero_model.gene:gene})
        score = np.sum(np.multiply(embed_compound[0,0,:], embed_gene[0,0,:]))
        if score < 0:
            num_neg += 1
        num_neg_total += 1
        count_compound += 1

    tp = num_pos
    fp = num_neg_total - num_neg
    precision = float(tp)/(tp+fp)
    recall = float(tp)/(num_pos_total)
    f1 = 2*(precision*recall)/(precision+recall)
    """



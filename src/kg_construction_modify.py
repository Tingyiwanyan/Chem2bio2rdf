import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import networkx as nx
import math
import time
import argparse
import pickle
from kg_model import hetero_model
from sklearn.manifold import TSNE

def parse_args():
    parser = argparse.ArgumentParser(description="Skip-gram model")
    parser.add_argument('--c2b2rdf', nargs='?', default='../data/c2b2rdf/relations.txt', help='chem2bio2rdf relations')
    parser.add_argument('--positive-file', nargs='?', default='../data/c2b2rdf/compound-gene/internal_testset_label/positive.txt', help='positive relations')
    parser.add_argument('--negative-file', nargs='?', default='../data/c2b2rdf/compound-gene/internal_testset_label/negative.txt', help='negative relations')
    parser.add_argument('--output', nargs='?', default=None, help="file to save embeddings")
    parser.add_argument('--compound-sim-file', nargs='?', default='data/compound_structure_similarity.txt', help='compound similarity')
    parser.add_argument('--nodes', nargs='?', default='data/nodes.txt', help='nodes file')
    return parser.parse_args()

class Kg_construct_chem2bio():
    """
    construct knowledge graph out of EHR data
    """
    def __init__(self, c2b2rdf_file, pos_file, neg_file, comp_sim_file):
        self.chem2bio2rdf_file = c2b2rdf_file
        self.positive_file = pos_file
        self.negative_file = neg_file
        self.compound_sim = comp_sim_file
        self.file_graph = open(self.chem2bio2rdf_file)
        self.file_positive = open(self.positive_file)
        self.file_negative = open(self.negative_file)
        self.file_compound_sim = open(self.compound_sim)


    def create_kg_dic(self):
        self.dic_compound = {}
        self.dic_gene = {}
        index_compound = 0
        index_gene = 0
        index_count = 0
        index_compound = 0
        for line in self.file_graph:
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
            if not self.dic_compound.get(first_comp_id):
                self.dic_compound[first_comp_id] = {}
                self.dic_compound[first_comp_id]['compound_index'] = index_compound
                self.dic_compound[first_comp_id].setdefault('neighbor_gene',[]).append(second_comp_id)
                index_compound += 1
            else:
                self.dic_compound[first_comp_id].setdefault('neighbor_gene', []).append(second_comp_id)
            if not self.dic_gene.get(second_comp_id):
                self.dic_gene[second_comp_id] = {}
                self.dic_gene[second_comp_id]['gene_index'] = index_gene
                self.dic_gene[second_comp_id].setdefault('neighbor_compound',[]).append(first_comp_id)
                index_gene += 1
            else:
                self.dic_gene[second_comp_id].setdefault('neighbor_compound', []).append(first_comp_id)
            index_count += 1

        for line in self.file_compound_sim:
            line = line.rstrip('\r\n')
            rough = line.split('|')
            first_comp_id = rough[0]
            sec_comp_id = rough[1]
            if not self.dic_compound.get(first_comp_id):
                continue
                #self.dic_compound[first_comp_id] = {}
                #self.dic_compound[first_comp_id]['compound_index'] = index_compound
                #self.dic_compound[first_comp_id].setdefault('neighbor_compound',[]).append(sec_comp_id)
                #index_compound += 1
            if not self.dic_compound.get(sec_comp_id):
                continue

            self.dic_compound[first_comp_id].setdefault('neighbor_compound', []).append(sec_comp_id)



def test_whole(hetro,kg):
    embed_compound = np.zeros((hetro.compound_size,hetro.latent_dim))
    embed_gene = np.zeros((hetro.gene_size,hetro.latent_dim))
    index = 0
    for i in kg.dic_compound.keys():
        single_compound = np.zeros((1,hetro.pos_compound_size+hetro.neg_compound_size,hetro.compound_size))
        single_compound[0,0,:] = hetro.assign_value_compound(i)
        single_embed_coumpound =  hetro.sess.run(hetro.Dense_compound_bind,feed_dict={hetro.compound:single_compound})
        embed_compound[index,:] = single_embed_coumpound[0,0,:]
        index += 1
    index = 0
    for i in kg.dic_gene.keys():
        single_gene = np.zeros((1,hetro.pos_gene_size+hetro.neg_gene_size,hetro.gene_size))
        single_gene[0,0,:] = hetro.assign_value_gene(i)
        single_embed_gene = hetro.sess.run(hetro.Dense_gene,
                                                 feed_dict={hetro.gene:single_gene})
        embed_gene[index,:] = single_embed_gene[0,0,:]
        index += 1
    return embed_compound,embed_gene

def test_acc(hetro,kg):
    tp_num = 0
    for i in kg.dic_compound.keys():
        single_compound = np.zeros((1, hetro.pos_compound_size + hetro.neg_compound_size, hetro.compound_size))
        single_compound[0, 0, :] = hetro.assign_value_compound(i)
        single_embed_coumpound = hetro.sess.run(hetro.Dense_compound_bind, feed_dict={hetro.compound: single_compound})
        link_gene = kg.dic_compound[i]['neighbor_gene'][0]
        single_gene = np.zeros((1,hetro.pos_gene_size+hetro.neg_gene_size,hetro.gene_size))
        single_gene[0,0,:] = hetro.assign_value_gene(link_gene)
        single_embed_gene = hetro.sess.run(hetro.Dense_gene,
                                           feed_dict={hetro.gene: single_gene})

        score = np.sum(np.multiply(single_embed_coumpound[0,0,:], single_embed_gene[0,0,:]))

        if score > 0:
            tp_num += 1

    return tp_num


def read_file(filepath):

    result = []
    with open(filepath, 'r') as file:
        for line in file:
            line = line.strip()
            result.append(line)

    return result

if __name__  == "__main__":
    args = parse_args()
    kg = Kg_construct_chem2bio(c2b2rdf_file=args.c2b2rdf, pos_file=args.positive_file, neg_file=args.negative_file, comp_sim_file=args.compound_sim_file)
    kg.create_kg_dic()
    hetero_model = hetero_model(kg)
    hetero_model.config_model()
    hetero_model.train()
    print('--- Read Nodes ---')
    nodes = read_file(args.nodes)
    embeddings = []
    compound_url = 'http://chem2bio2rdf.org/pubchem/resource/pubchem_compound/'
    gene_url = 'http://chem2bio2rdf.org/uniprot/resource/gene/'
    compound_nodes = []
    gene_nodes = []

    for node in nodes:
        if compound_url in node:
            compound_nodes.append(node)
        if gene_url in node:
            gene_nodes.append(node)

    embeddings = {}
    print('--- Computing Embeddings ---')
    for i, node in enumerate(compound_nodes):
        compoundid = os.path.basename(node)
        if compoundid not in kg.dic_compound.keys():
            continue
        compound = np.zeros((1, hetero_model.pos_compound_size+hetero_model.neg_compound_size, hetero_model.compound_size))
        compound[0, 0, :] = hetero_model.assign_value_compound(compoundid)
        embed_compound = hetero_model.sess.run(hetero_model.Dense_compound_sim,feed_dict={hetero_model.compound:compound})
        embeddings[node] = embed_compound[0,0,:]

    for i, node in enumerate(gene_nodes):
        geneid = os.path.basename(node)
        if geneid not in kg.dic_gene.keys():
            continue
        gene = np.zeros((1,hetero_model.pos_gene_size+hetero_model.neg_gene_size,hetero_model.gene_size))
        gene[0,0,:] = hetero_model.assign_value_gene(geneid)
        embed_gene = hetero_model.sess.run(hetero_model.Dense_gene,feed_dict={hetero_model.gene:gene})
        embeddings[node] = embed_gene[0,0,:]

    print(embeddings)
    print('--- Saving Embeddings ---')
    pickle.dump(embeddings, open(args.output, 'wb'))

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
        print(count_compound)
        if count_compound == 500:
            break
        line = line.rstrip('\n')
        pair = line.split('\t')
        compoundid = pair[0]
        geneid = pair[1]
        if compoundid not in kg.dic_compound.keys():
            continue
        if geneid not in kg.dic_gene.keys():
            continue
        compound = np.zeros((1, hetero_model.pos_compound_size+hetero_model.neg_compound_size, hetero_model.compound_size))
        compound[0, 0, :] = hetero_model.assign_value_compound(compoundid)
        gene = np.zeros((1,hetero_model.pos_gene_size+hetero_model.neg_gene_size,hetero_model.gene_size))
        gene[0,0,:] = hetero_model.assign_value_gene(geneid)
        embed_compound = hetero_model.sess.run(hetero_model.Dense_compound_sim,feed_dict={hetero_model.compound:compound})
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
        print(count_compound)
        if count_compound == 100:
            break
        line = line.rstrip('\n')
        pair = line.split('\t')
        compoundid = pair[0]
        geneid = pair[1]
        if compound not in kg.dic_compound.keys():
            continue
        if geneid not in kg.dic_gene.keys():
            continue
        compound = np.zeros((1, hetero_model.pos_compound_size, hetero_model.compound_size))
        compound[0, 0, :] = hetero_model.assign_value_compound(compoundid)
        gene = np.zeros((1,hetero_model.pos_gene_size+hetero_model.neg_gene_size,hetero_model.gene_size))
        gene[0,0,:] = hetero_model.assign_value_gene(geneid)
        embed_compound = hetero_model.sess.run(hetero_model.Dense_compound_sim,feed_dict={hetero_model.compound:compound})
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

    for i in kg.dic_diag.keys():
        index_class = kg.dic_diag[i]['icd']
        index = kg.dic_diag[i]['diag_index']
        label[index] = index_class
    for i in range(4510):
        if label[i] == 0:
            color_ = "blue"
            makersize_ = 4
        if label[i] == 18:
            color_ = "red"
            makersize_ = 5
        if label[i] == 19:
            color_ = "green"
            makersize_ = 6
        plt.plot(embed_total_2d[i][0],embed_total_2d[i][1],'.',color=color_,markersize=makersize_)
    """
    # embed_total_2d = TSNE(n_components=2).fit_transform(embed_total)

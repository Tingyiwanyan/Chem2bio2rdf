import numpy as np
import matplotlib.pyplot as plt
import random
import networkx as nx
import math
import time
from kg_model import hetero_model
from sklearn.manifold import TSNE
from graph_sage import graphsage_model

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
        index_compound = 0
        for line in self.file_graph:
            #if index_count > 20000:
               # break
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

        for line in self.file_compound_sim:
            line = line.rstrip('\r\n')
            rough = line.split('|')
            first_comp_id = rough[0]
            sec_comp_id = rough[1]
            if not self.dic_compound.has_key(first_comp_id):
                continue
                #self.dic_compound[first_comp_id] = {}
                #self.dic_compound[first_comp_id]['compound_index'] = index_compound
                #self.dic_compound[first_comp_id].setdefault('neighbor_compound',[]).append(sec_comp_id)
                #index_compound += 1
            if not self.dic_compound.has_key(sec_comp_id):
                continue

            self.dic_compound[first_comp_id].setdefault('neighbor_compound', []).append(sec_comp_id)



def test_whole(hetro,kg):
    embed_compound = np.zeros((hetro.compound_size,hetro.latent_dim))
    compound_id = np.zeros(hetro.compound_size)
    embed_gene = np.zeros((hetro.gene_size,hetro.latent_dim))
    gene_id = np.zeros(hetro.gene_size)
    index = 0
    for i in kg.dic_compound.keys():
        print(index)
        single_compound = np.zeros((1,hetro.pos_compound_size+hetro.neg_compound_size,hetro.compound_size))
        single_compound[0,0,:] = hetro.assign_value_compound(i)
        single_embed_coumpound =  hetro.sess.run(hetro.Dense_compound_bind,feed_dict={hetro.compound:single_compound})
        embed_compound[index,:] = single_embed_coumpound[0,0,:]
        compound_id[index] = i
        index += 1
    index = 0
    for i in kg.dic_gene.keys():
        print(index)
        single_gene = np.zeros((1,hetro.pos_gene_size+hetro.neg_gene_size,hetro.gene_size))
        single_gene[0,0,:] = hetro.assign_value_gene(i)
        single_embed_gene = hetro.sess.run(hetro.Dense_gene,
                                                 feed_dict={hetro.gene:single_gene})
        embed_gene[index,:] = single_embed_gene[0,0,:]
        gene_id[index] = i
        index += 1
    return embed_compound,embed_gene,compound_id,gene_id




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

def test_acc_sage(hetro,kg):
    tp_num = 0
    index = 0
    for i in kg.dic_compound.keys():
        print(index)
        single_compound = np.zeros((1, hetro.pos_sample_size + hetro.neg_sample_size+1, hetro.total_size))
        single_compound[0, 0, :] = hetro.gcn_agg_compound(i)
        single_embed_coumpound = hetro.sess.run(hetro.Dense_gcn, feed_dict={hetro.input_x: single_compound})
        link_gene = kg.dic_compound[i]['neighbor_gene'][0]
        single_gene = np.zeros((1,hetro.pos_sample_size+hetro.neg_sample_size+1,hetro.total_size))
        single_gene[0,0,:] = hetro.gcn_agg_gene(link_gene)
        single_embed_gene = hetro.sess.run(hetro.Dense_gcn,
                                           feed_dict={hetro.input_x: single_gene})

        score = np.sum(np.multiply(single_embed_coumpound[0,0,:], single_embed_gene[0,0,:]))

        if score > 0:
            tp_num += 1
        index+=1

    return tp_num




if __name__  == "__main__":
    kg = Kg_construct_chem2bio()
    kg.create_kg_dic()
    gsage = graphsage_model(kg)
    #hetero_model = hetero_model(kg)
    #hetero_model.config_model()
    #hetero_model.train()
    """
    hetro = hetero_model
    embed_compound = np.zeros((hetro.compound_size, hetro.latent_dim))
    compound_id = np.zeros(hetro.compound_size)
    gene_id = np.zeros(hetro.gene_size)
    index = 0
    for i in kg.dic_compound.keys():
        print(index)
        single_compound = np.zeros((1, hetro.pos_compound_size + hetro.neg_compound_size, hetro.compound_size))
        single_compound[0, 0, :] = hetro.assign_value_compound(i)
        single_embed_coumpound = hetro.sess.run(hetro.Dense_compound_bind, feed_dict={hetro.compound: single_compound})
        embed_compound[index, :] = single_embed_coumpound[0, 0, :]
        compound_id[index] = i
        index += 1

    gene_id = []
    embed_gene = np.zeros((hetro.gene_size, hetro.latent_dim))
    index = 0
    for i in kg.dic_gene.keys():
        print(index)
        single_gene = np.zeros((1, hetro.pos_gene_size + hetro.neg_gene_size, hetro.gene_size))
        single_gene[0, 0, :] = hetro.assign_value_gene(i)
        single_embed_gene = hetro.sess.run(hetro.Dense_gene,
                                           feed_dict={hetro.gene: single_gene})
        embed_gene[index, :] = single_embed_gene[0, 0, :]
        gene_id.append(str(i))
        index += 1
    """

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
    #embed_total_2d = TSNE(n_components=2).fit_transform(embed_total)





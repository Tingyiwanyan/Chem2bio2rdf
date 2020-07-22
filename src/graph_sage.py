import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from itertools import groupby
#import utils


class graphsage_model():
    """
    create heterogenous model for chem2bio2rdf
    """
    def __init__(self,kg):
        self.kg = kg
        self.compound_size = len(list(kg.dic_compound.keys()))
        self.gene_size = len(list(kg.dic_gene.keys()))
        self.total_size = self.compound_size+self.gene_size
        self.train_nodes = list(kg.dic_compound.keys())
        self.train_nodes_size = len(self.train_nodes)
        self.batch_size = 64
        self.latent_dim = 200
        self.pos_compound_size = 2
        self.pos_gene_size = 1
        self.neg_compound_size = 15
        self.neg_gene_size = 10
        self.negative_sample_size = self.neg_gene_size + self.neg_compound_size
        self.positive_sample_size = self.pos_compound_size+self.pos_gene_size - 1
        self.walk_length = self.positive_sample_size
        self.skip_size = self.pos_compound_size+self.pos_gene_size
        """
        initialize input variables
        """
        self.input_x = tf.placeholder(tf.float32,[None,1+self.positive_sample_size+self.negative_sample_size,self.total_size])
        self.input_x_center = tf.placeholder(tf.float32,[None,1+self.positive_sample_size+self.negative_sample_size,self.total_size])
        self.compound = tf.placeholder(tf.float32,[None,self.pos_compound_size+self.neg_compound_size,self.compound_size])
        self.gene = tf.placeholder(tf.float32,[None,self.pos_gene_size+self.neg_gene_size,self.gene_size])
        """
        initial relation type binds
        """
        self.init_binds = tf.keras.initializers.he_normal(seed=None)
        self.shape_relation = (self.latent_dim,)
        self.relation_binds = tf.Variable(self.init_binds(shape=self.shape_relation))
        """
        initial relation type similar
        """
        self.init_similar = tf.keras.initializers.he_normal(seed=None)
        self.shape_relation = (self.latent_dim,)
        self.relation_similar = tf.Variable(self.init_similar(shape=self.shape_relation))
        """
        Create meta_path type
        """
        #self.meta_path1 = ['c','g','c','g','c','g']
        self.meta_path1 = ['c','c','g']

    def config_model(self):
        self.build_hetero_model()
        self.get_latent_rep()
        self.SGNN_loss()
        self.train_step_neg = tf.train.AdamOptimizer(1e-3).minimize(self.negative_sum)
        self.sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()

    def build_hetero_model(self):
        """
        build local gcn layer
        """
        self.Dense_gcn = tf.layers.dense(inputs=self.input_x,
                                         units=self.latent_dim,
                                         kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                                         activation=tf.nn.relu)

        self.concat_x = tf.concat((self.Dense_gcn,self.input_x_center),axis=2)

        self.Dense_final = tf.layers.dense(inputs=self.concat_x,
                                           units=self.latent_dim,
                                           kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                                           activation=tf.nn.relu)
        """
        build compound projection layer
        """
        self.Dense_compound = tf.layers.dense(inputs=self.compound,
                                              units = self.latent_dim,
                                              kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                                              activation=tf.nn.relu)

        """
        build gene projection layer
        """
        self.Dense_gene = tf.layers.dense(inputs=self.gene,
                                              units = self.latent_dim,
                                              kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                                              activation=tf.nn.relu)

        """
        transE
        """
        self.Dense_compound_sim = tf.math.add(self.Dense_compound,self.relation_similar)
        self.Dense_compound_bind = tf.math.add(self.Dense_compound,self.relation_binds)


    def get_latent_rep(self):
        """
        prepare latent representation for skip-gram model
        """
        self.x_skip_compound = None
        self.x_negative_compound = None
        self.x_skip_gene = None
        self.x_negative_gene = None
        """
        get center node representation, case where center node is compound
        """
        idx_origin = tf.constant([0])
        self.x_origin =tf.gather(self.Dense_final,idx_origin,axis=1)
        """
        total data case
        """
        idx_skip = tf.constant([i+1 for i in range(self.pos_compound_size+self.pos_gene_size-1)])
        idx_negative = \
            tf.constant([i+self.pos_compound_size+self.pos_gene_size for i in range(self.negative_sample_size)])
        self.x_skip = tf.gather(self.Dense_final,idx_skip,axis=1)
        self.x_negative = tf.gather(self.Dense_final,idx_negative,axis=1)

    def SGNN_loss(self):
        """
        implement sgnn loss
        """
        negative_training_norm = tf.math.l2_normalize(self.x_negative, axis=2)

        skip_training = tf.broadcast_to(self.x_origin,
                                        [self.batch_size, self.negative_sample_size, self.latent_dim])

        skip_training_norm = tf.math.l2_normalize(skip_training, axis=2)

        dot_prod = tf.multiply(skip_training_norm, negative_training_norm)

        dot_prod_sum = tf.reduce_sum(dot_prod, 2)

        sum_log_dot_prod = tf.math.log(tf.math.sigmoid(tf.math.negative(tf.reduce_mean(dot_prod_sum, 1))))

        positive_training = tf.broadcast_to(self.x_origin, [self.batch_size, self.walk_length, self.latent_dim])

        positive_skip_norm = tf.math.l2_normalize(self.x_skip, axis=2)

        positive_training_norm = tf.math.l2_normalize(positive_training, axis=2)

        dot_prod_positive = tf.multiply(positive_skip_norm, positive_training_norm)

        dot_prod_sum_positive = tf.reduce_sum(dot_prod_positive, 2)

        sum_log_dot_prod_positive = tf.math.log(tf.math.sigmoid(tf.reduce_mean(dot_prod_sum_positive, 1)))

        self.negative_sum = tf.math.negative(
            tf.reduce_sum(tf.math.add(sum_log_dot_prod, sum_log_dot_prod_positive)))

    """
    GCN aggregator for compound
    """
    def gcn_agg_compound(self,compoundid):
        #one_sample = np.zeros(self.total_size)
        #neighbor_compound = self.kg.dic_compound[compoundid]['neighbor_compound']
        neighbor_gene = self.kg.dic_compound[compoundid]['neighbor_gene']
        agg_vec = self.assign_value_compound(compoundid)
        self.compount_origin = agg_vec
        center_neighbor_size = len(neighbor_gene)
        ave_factor = 1.0 / np.sqrt(center_neighbor_size*center_neighbor_size)
        one_sample = agg_vec * ave_factor
        """
        for i in neighbor_compound:
            neighbor_compound_len = len(self.kg.dic_compound[i]['neighbor_compound'])
            neighbor_gene_len = len(self.kg.dic_compound[i]['neighbor_gene'])
            neighbor_size = neighbor_compound_len + neighbor_gene_len
            ave_factor = 1.0 / np.sqrt(neighbor_size * center_neighbor_size)
            agg_cur = self.assign_value_compound(i) * ave_factor
            one_sample = one_sample + agg_cur
        """
        for i in neighbor_gene:
            neighbor_compound_len = len(self.kg.dic_gene[i]['neighbor_compound'])
            neighbor_size = neighbor_compound_len
            ave_factor = 1.0 / np.sqrt(neighbor_size * center_neighbor_size)
            agg_cur = self.assign_value_gene(i) * ave_factor
            one_sample = one_sample + agg_cur

        one_sample_final = np.concatenate((agg_vec, one_sample),0)

        return one_sample,agg_vec

    """
    GCN aggregator for gene
    """
    def gcn_agg_gene(self,geneid):
        #one_sample = np.zeros(self.total_size)
        neighbor_compound = self.kg.dic_gene[geneid]['neighbor_compound']
        agg_vec = self.assign_value_gene(geneid)
        self.gene_origin = agg_vec
        center_neighbor_size = len(neighbor_compound)
        ave_factor = 1.0 / np.sqrt(center_neighbor_size*center_neighbor_size)
        one_sample = agg_vec * ave_factor
        for i in neighbor_compound:
            self.check_compound = i
            neighbor_compound_len = 0
            neighbor_gene_len = 0
            if self.kg.dic_compound[i].has_key('neighbor_compound'):
                neighbor_compound_len = len(self.kg.dic_compound[i]['neighbor_compound'])
            neighbor_gene_len = len(self.kg.dic_compound[i]['neighbor_gene'])
            neighbor_size = neighbor_compound_len + neighbor_gene_len
            ave_factor = 1.0 / np.sqrt(neighbor_size * center_neighbor_size)
            agg_cur = self.assign_value_compound(i) * ave_factor
            one_sample = one_sample + agg_cur
        one_sample_final = np.concatenate((agg_vec,one_sample),0)

        return one_sample,agg_vec


    """
    assign value to one compound sample
    """
    def assign_value_compound(self,compoundid):
        one_sample = np.zeros(self.total_size)
        index = self.kg.dic_compound[compoundid]['compound_index']
        one_sample[index] = 1

        return one_sample


    """
    assign value to one gene sample
    """
    def assign_value_gene(self,geneid):
        one_sample = np.zeros(self.total_size)
        index = self.kg.dic_gene[geneid]['gene_index']
        one_sample[self.compound_size+index] = 1

        return one_sample


    """
    preparing data for one metapath
    """
    def get_positive_sample_metapath(self,meta_path):
        self.compound_nodes = []
        self.gene_nodes = []
        self.compound_center = []
        self.gene_center = []
        for i in meta_path:
            if i[0] == 'c':
                compound_id = i[1]
                compound_sample,compound_sample_center = self.gcn_agg_compound(compound_id)
                self.compound_nodes.append(compound_sample)
                self.compound_center.append(compound_sample_center)
            if i[0] == 'g':
                gene_id = i[1]
                gene_sample,gene_sample_center = self.gcn_agg_gene(gene_id)
                self.gene_nodes.append(gene_sample)
                self.gene_center.append(gene_sample_center)

    """
    prepare data for one metapath negative sample
    """
    def get_negative_sample_metapath(self):

        self.gene_neg_sample = np.zeros((self.neg_gene_size,self.total_size))
        self.compound_neg_sample = np.zeros((self.neg_compound_size,self.total_size))
        self.gene_neg_center = np.zeros((self.neg_gene_size,self.total_size))
        self.compound_neg_center = np.zeros((self.neg_compound_size,self.total_size))
        index = 0
        for i in self.neg_nodes_gene:
            one_sample_neg_gene,one_sample_neg_gene_center= self.gcn_agg_gene(i)
            self.gene_neg_sample[index,:] = one_sample_neg_gene
            self.gene_neg_center[index,:] = one_sample_neg_gene_center
            index += 1
        index = 0
        for i in self.neg_nodes_compound:
            one_sample_neg_compound, one_sample_neg_compound_center = self.gcn_agg_compound(i)
            self.compound_neg_sample[index,:] = one_sample_neg_compound
            self.compound_neg_center[index,:] = one_sample_neg_compound_center
            index += 1

    """
    prepare data for negative hererogenous sampling
    """
    def get_negative_samples(self,center_node_type,center_node_index):
        self.neg_nodes_compound = []
        self.neg_nodes_gene = []
        """
        get neg set for gene
        """
        if center_node_type == 'c':
            gene_neighbor_nodes = self.kg.dic_compound[center_node_index]['neighbor_gene']
            whole_gene_nodes = self.kg.dic_gene.keys()
            gene_neighbor_nodes = gene_neighbor_nodes + self.walk_gene
            neg_set_gene = [i for i in whole_gene_nodes if i not in gene_neighbor_nodes]
            for j in range(self.neg_gene_size):
                index_sample = np.int(np.floor(np.random.uniform(0,len(neg_set_gene),1)))
                self.neg_nodes_gene.append(neg_set_gene[index_sample])
            compound_neighbor_nodes = self.kg.dic_compound[center_node_index]['neighbor_compound']
            whole_compound_nodes = self.kg.dic_compound.keys()
            compound_neighbor_nodes = compound_neighbor_nodes + self.walk_compound
            neg_set_compound = [i for i in whole_compound_nodes if i not in compound_neighbor_nodes]
            for j in range(self.neg_compound_size):
                index_sample = np.int(np.floor(np.random.uniform(0,len(neg_set_compound),1)))
                self.neg_nodes_compound.append(neg_set_compound[index_sample])




    """
    extract meta-path
    """
    def extract_meta_path(self,center_node_type,start_index,meta_path_type):
        walk = []
        walk.append([center_node_type,start_index])
        meta_path_gen = meta_path_type[1:]
        cur_index = start_index
        cur_node_type = center_node_type
        self.walk_compound = []
        self.walk_gene = []
        flag = 0
        while(flag == 0):
            for i in meta_path_gen:
                if i == 'c':
                    if cur_node_type == 'g':
                        neighbor = list(self.kg.dic_gene[cur_index]['neighbor_compound'])
                        """
                        uniformly generate sampling index
                        """
                        random_index = np.int(np.floor(np.random.uniform(0,len(neighbor),1)))
                        cur_index = neighbor[random_index]
                        cur_node_type = 'c'
                        walk.append([cur_node_type,cur_index])
                        self.walk_compound.append(cur_index)
                    if cur_node_type == 'c':
                        neighbor = list(self.kg.dic_compound[cur_index]['neighbor_compound'])
                        random_index = np.int(np.floor(np.random.uniform(0, len(neighbor), 1)))
                        cur_index = neighbor[random_index]
                        while('neighbor_compound' not in self.kg.dic_compound[cur_index].keys()):
                            random_index = np.int(np.floor(np.random.uniform(0, len(neighbor), 1)))
                            cur_index = neighbor[random_index]
                        cur_node_type = 'c'
                        walk.append([cur_node_type, cur_index])
                        self.walk_compound.append(cur_index)


                if i == 'g':
                    if cur_node_type == 'c':
                        if ('neighbor_gene' not in self.kg.dic_compound[cur_index]):
                            walk = []
                            self.walk_compound = []
                            self.walk_gene = []
                            break
                        else:
                            flag = 1
                        neighbor = list(self.kg.dic_compound[cur_index]['neighbor_gene'])
                        """
                        uniformly generate sampling index
                        """
                        random_index = np.int(np.floor(np.random.uniform(0,len(neighbor),1)))
                        cur_index = neighbor[random_index]
                        cur_node_type = 'g'
                        walk.append([cur_node_type,cur_index])
                        self.walk_gene.append(cur_index)
        return walk


    """
    prepare one batch data
    """
    def get_one_batch(self,meta_path_type,center_node_type,start_index):
        #compound_sample = np.zeros((self.batch_size,self.pos_compound_size+self.neg_compound_size,self.compound_size))
        #gene_sample = np.zeros((self.batch_size,self.pos_gene_size+self.neg_gene_size,self.gene_size))
        one_batch_train = np.zeros((self.batch_size,1+self.positive_sample_size+self.negative_sample_size,self.total_size))
        one_batch_train_center = np.zeros((self.batch_size,1+self.positive_sample_size+self.negative_sample_size,self.total_size))
        num_sample = 0
        increament_step = 0
        while num_sample < self.batch_size:
            center_node_index = self.train_nodes[increament_step+start_index]
            if not 'neighbor_compound' in self.kg.dic_compound[center_node_index]:
                increament_step += 1
                continue
            single_meta_path = self.extract_meta_path(center_node_type,center_node_index,meta_path_type)
            self.get_positive_sample_metapath(single_meta_path)
            self.get_negative_samples(center_node_type,center_node_index)
            self.get_negative_sample_metapath()
            #single_compound_sample = np.concatenate((self.compound_nodes,self.compound_neg_sample))
            #single_gene_sample = np.concatenate((self.gene_nodes,self.gene_neg_sample))
            #compound_sample[num_sample,:,:] = single_compound_sample
            #gene_sample[num_sample,:,:] = single_gene_sample
            positive_sample = np.concatenate((self.compound_nodes,self.gene_nodes))
            positive_sample_center = np.concatenate((self.compound_center,self.gene_center))
            negative_sample = np.concatenate((self.compound_neg_sample,self.gene_neg_sample))
            negative_sample_center = np.concatenate((self.compound_neg_center,self.gene_neg_center))
            total_sample = np.concatenate((positive_sample,negative_sample))
            total_sample_center = np.concatenate((positive_sample_center,negative_sample_center))
            one_batch_train[num_sample,:,:] = total_sample
            one_batch_train_center[num_sample,:,:] = total_sample_center

            num_sample += 1

        return one_batch_train, one_batch_train_center

    """
    train model
    """
    def train(self):
        iteration = np.int(np.floor(np.float(self.train_nodes_size)/self.batch_size))
        epoch=6
        for j in range(epoch):
            for i in range(iteration):
                #if i > 300:
                   # break
                batch_total,batch_total_center = self.get_one_batch(self.meta_path1,'c',i*self.batch_size)
                err_ = self.sess.run([self.negative_sum,self.train_step_neg],feed_dict={self.input_x:batch_total,
                                                                                        self.input_x_center:batch_total_center})
                print(err_[0])

    def test_whole(self):
        test_compound = np.zeors((self.compound_size,self.pos_compound_size+self.neg_compound_size,self.compound_size))
        test_gene = np.zeros((self.gene_size,self.pos_gene_size+self.neg_gene_size,self.gene_size))
        for i in range(self.compound_size):
            compound[0,0,:] = self.assign_value_compound(compoundid)
        #embed_compound = self.sess.run([self.Dense_compound],feed_dict={self.})

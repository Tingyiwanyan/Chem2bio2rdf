import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from itertools import groupby

class hetero_model():
    """
    create heterogenous model for chem2bio2rdf
    """
    def __init__(self,kg):
        self.kg = kg
        self.compound_size = len(list(kg.dic_compound.keys()))
        self.gene_size = len(list(kg.dic_gene.keys()))
        self.train_nodes = list(kg.dic_compound.keys())
        self.train_nodes_size = len(self.train_nodes)
        self.batch_size = 64
        self.latent_dim = 200
        self.pos_compound_size = 2
        self.pos_gene_size = 1
        self.neg_compound_size = 30
        self.neg_gene_size = 30
        self.negative_sample_size = self.neg_gene_size + self.neg_compound_size
        self.positive_sample_size = self.pos_compound_size+self.pos_gene_size-1
        self.walk_length = self.positive_sample_size
        self.skip_size = self.pos_compound_size+self.pos_gene_size
        """
        initialize input variables
        """
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
        self.x_origin =tf.gather(self.Dense_compound_bind,idx_origin,axis=1)
        """
        compound case
        """
        compound_idx_skip = tf.constant([i+1 for i in range(self.pos_compound_size-1)])
        compound_idx_negative = \
            tf.constant([i+self.pos_compound_size for i in range(self.neg_compound_size)])
        self.x_skip_compound = tf.gather(self.Dense_compound_bind,compound_idx_skip,axis=1)
        self.x_negative_compound = tf.gather(self.Dense_compound_bind,compound_idx_negative,axis=1)
        """
        gene case
        """
        gene_idx_skip = tf.constant([i for i in range(self.pos_gene_size)])
        gene_idx_negative = \
            tf.constant([i+self.pos_gene_size for i in range(self.neg_gene_size)])
        self.x_skip_gene = tf.gather(self.Dense_gene,gene_idx_skip,axis=1)
        self.x_negative_gene = tf.gather(self.Dense_gene,gene_idx_negative,axis=1)
        """
        combine skip samples and negative samples
        """
        self.x_skip = tf.concat([self.x_skip_compound,self.x_skip_gene],axis=1)
        self.x_negative = tf.concat([self.x_negative_compound,self.x_negative_gene],axis=1)

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
    assign value to one compound sample
    """
    def assign_value_compound(self,compoundid):
        one_sample = np.zeros(self.compound_size)
        index = self.kg.dic_compound[compoundid]['compound_index']
        one_sample[index] = 1

        return one_sample


    """
    assign value to one gene sample
    """
    def assign_value_gene(self,geneid):
        one_sample = np.zeros(self.gene_size)
        index = self.kg.dic_gene[geneid]['gene_index']
        one_sample[index] = 1

        return one_sample

    """
    preparing data for one metapath
    """
    def get_positive_sample_metapath(self,meta_path):
        self.compound_nodes = []
        self.gene_nodes = []
        for i in meta_path:
            if i[0] == 'c':
                compound_id = i[1]
                compound_sample = self.assign_value_compound(compound_id)
                self.compound_nodes.append(compound_sample)
            if i[0] == 'g':
                gene_id = i[1]
                gene_sample = self.assign_value_gene(gene_id)
                self.gene_nodes.append(gene_sample)

    """
    prepare data for one metapath negative sample
    """
    def get_negative_sample_metapath(self):

        self.gene_neg_sample = np.zeros((self.neg_gene_size,self.gene_size))
        self.compound_neg_sample = np.zeros((self.neg_compound_size,self.compound_size))
        index = 0
        for i in self.neg_nodes_gene:
            one_sample_neg_gene = self.assign_value_gene(i)
            self.gene_neg_sample[index,:] = one_sample_neg_gene
            index += 1
        index = 0
        for i in self.neg_nodes_compound:
            one_sample_neg_compound = self.assign_value_compound(i)
            self.compound_neg_sample[index,:] = one_sample_neg_compound
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
        compound_sample = np.zeros((self.batch_size,self.pos_compound_size+self.neg_compound_size,self.compound_size))
        gene_sample = np.zeros((self.batch_size,self.pos_gene_size+self.neg_gene_size,self.gene_size))
        num_sample = 0
        increament_step = 0
        while num_sample < self.batch_size:
        #for i in range(self.batch_size):
            center_node_index = self.train_nodes[increament_step+start_index]
            if not 'neighbor_compound' in self.kg.dic_compound[center_node_index]:
                increament_step += 1
                continue
            single_meta_path = self.extract_meta_path(center_node_type,center_node_index,meta_path_type)
            self.get_positive_sample_metapath(single_meta_path)
            self.get_negative_samples(center_node_type,center_node_index)
            self.get_negative_sample_metapath()
            single_compound_sample = np.concatenate((self.compound_nodes,self.compound_neg_sample))
            single_gene_sample = np.concatenate((self.gene_nodes,self.gene_neg_sample))
            compound_sample[num_sample,:,:] = single_compound_sample
            gene_sample[num_sample,:,:] = single_gene_sample
            num_sample += 1

        return compound_sample, gene_sample

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
                batch_coumpound,batch_gene = self.get_one_batch(self.meta_path1,'c',i*self.batch_size)
                err_ = self.sess.run([self.negative_sum,self.train_step_neg],feed_dict={self.compound:batch_coumpound,
                                                                                        self.gene:batch_gene})
                print(err_[0])

    def test_whole(self):
        test_compound = np.zeors((self.compound_size,self.pos_compound_size+self.neg_compound_size,self.compound_size))
        test_gene = np.zeros((self.gene_size,self.pos_gene_size+self.neg_gene_size,self.gene_size))
        for i in range(self.compound_size):
            compound[0,0,:] = self.assign_value_compound(compoundid)
        #embed_compound = self.sess.run([self.Dense_compound],feed_dict={self.})


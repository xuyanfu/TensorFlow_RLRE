import tensorflow as tf
import numpy as np
import random
import tensorflow.contrib.layers as layers
from tqdm import tqdm
import time


class Settings(object):
    def __init__(self):
        self.vocab_size = 114042
        self.len_sentence = 70
        self.num_epochs = 3
        self.num_classes = 53
        self.cnn_size = 230
        self.num_layers = 1
        self.pos_size = 5
        self.pos_num = 123
        self.word_embedding = 50
        self.keep_prob = 0.5
        self.batch_size = 300
        self.num_steps = 10000
        self.lr= 0.001


class CNN():

    def __init__(self, word_embeddings, setting):

        self.vocab_size = setting.vocab_size
        self.len_sentence= len_sentence = setting.len_sentence
        self.num_epochs = setting.num_epochs
        self.num_classes = num_classes =setting.num_classes
        self.cnn_size = setting.cnn_size
        self.num_layers = setting.num_layers
        self.pos_size = setting.pos_size
        self.pos_num = setting.pos_num
        self.word_embedding = setting.word_embedding
        self.lr = setting.lr


        word_embedding = tf.get_variable(initializer=word_embeddings, name='word_embedding')
        pos1_embedding = tf.get_variable('pos1_embedding', [self.pos_num, self.pos_size])
        pos2_embedding = tf.get_variable('pos2_embedding', [self.pos_num, self.pos_size])
        #relation_embedding = tf.get_variable('relation_embedding', [self.num_classes, self.cnn_size])

        self.input_word = tf.placeholder(dtype=tf.int32, shape=[None, len_sentence], name='input_word')
        self.input_pos1 = tf.placeholder(dtype=tf.int32, shape=[None, len_sentence], name='input_pos1')
        self.input_pos2 = tf.placeholder(dtype=tf.int32, shape=[None, len_sentence], name='input_pos2')
        self.input_y = tf.placeholder(dtype=tf.float32, shape=[None, num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32)


        self.input_word_ebd = tf.nn.embedding_lookup(word_embedding, self.input_word)
        self.input_pos1_ebd = tf.nn.embedding_lookup(pos1_embedding, self.input_pos1)
        self.input_pos2_ebd = tf.nn.embedding_lookup(pos2_embedding, self.input_pos2)


        self.inputs =  tf.concat(axis=2,values=[self.input_word_ebd,self.input_pos1_ebd,self.input_pos2_ebd])
        self.inputs = tf.reshape(self.inputs, [-1,self.len_sentence,self.word_embedding+self.pos_size*2,1] )


        #卷积层
        conv = layers.conv2d(inputs =self.inputs ,num_outputs = self.cnn_size ,kernel_size = [3,60],stride=[1,60],padding='SAME')

        #pooling层
        max_pool = layers.max_pool2d(conv,kernel_size = [70,1],stride=[1,1])
        self.sentence = tf.reshape(max_pool, [-1, self.cnn_size])

        #dropout层
        tanh = tf.nn.tanh(self.sentence)
        drop = layers.dropout(tanh,keep_prob=self.keep_prob)

        #全连接层
        self.outputs = layers.fully_connected(inputs = drop,num_outputs = self.num_classes,activation_fn = tf.nn.softmax)

        '''
        self.y_index =  tf.argmax(self.input_y,1,output_type=tf.int32)
        self.indexes = tf.range(0, tf.shape(self.outputs)[0]) * tf.shape(self.outputs)[1] + self.y_index
        self.responsible_outputs = - tf.reduce_mean(tf.log(tf.gather(tf.reshape(self.outputs, [-1]),self.indexes)))
        '''
        #loss
        self.cross_loss = -tf.reduce_mean( tf.log(tf.reduce_sum( self.input_y  * self.outputs ,axis=1)))
        self.reward =  tf.log(tf.reduce_sum( self.input_y  * self.outputs ,axis=1))

        self.l2_loss = tf.contrib.layers.apply_regularization(regularizer=tf.contrib.layers.l2_regularizer(0.0001),
                                                              weights_list=tf.trainable_variables())

        self.final_loss = self.cross_loss + self.l2_loss


        #accuracy
        self.pred = tf.argmax(self.outputs,axis=1)
        self.pred_prob = tf.reduce_max(self.outputs,axis=1)

        self.y_label = tf.argmax(self.input_y,axis=1)
        self.accuracy = tf.reduce_mean(tf.cast( tf.equal(self.pred,self.y_label), 'float'))

        #minimize loss
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.minimize(self.final_loss)


        self.tvars = tf.trainable_variables()

        # manual update parameters
        self.tvars_holders = []
        for idx, var in enumerate(self.tvars):
            placeholder = tf.placeholder(tf.float32, name=str(idx) + '_holder')
            self.tvars_holders.append(placeholder)

        self.update_tvar_holder = []
        for idx, var in enumerate(self.tvars):
            update_tvar = tf.assign(var, self.tvars_holders[idx])
            self.update_tvar_holder.append(update_tvar)


def train(path_train_word,path_train_pos1,path_train_pos2,path_train_y,save_path):

    print('reading wordembedding')
    wordembedding = np.load('./data/vec.npy')

    print('reading training data')

    cnn_train_word = np.load(path_train_word)
    cnn_train_pos1 = np.load(path_train_pos1)
    cnn_train_pos2 = np.load(path_train_pos2)
    cnn_train_y    = np.load(path_train_y)

    settings = Settings()
    settings.vocab_size = len(wordembedding)
    settings.num_classes = len(cnn_train_y[0])
    settings.num_steps = len(cnn_train_word) // settings.batch_size

    with tf.Graph().as_default():
        sess = tf.Session()
        with sess.as_default():

            initializer = tf.contrib.layers.xavier_initializer()
            with tf.variable_scope("model", reuse=None, initializer=initializer):
                model = CNN(word_embeddings=wordembedding, setting=settings)

            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            #saver.restore(sess,save_path=save_path)
            for epoch in range(1,settings.num_epochs+1):

                bar = tqdm(range(settings.num_steps), desc='epoch {}, loss=0.000000, accuracy=0.000000'.format(epoch))

                for _ in bar:

                    sample_list = random.sample(range(len(cnn_train_y)),settings.batch_size)
                    batch_train_word = [cnn_train_word[x] for x in sample_list]
                    batch_train_pos1 = [cnn_train_pos1[x] for x in sample_list]
                    batch_train_pos2 = [cnn_train_pos2[x] for x in sample_list]
                    batch_train_y = [cnn_train_y[x] for x in sample_list]

                    feed_dict = {}
                    feed_dict[model.input_word] = batch_train_word
                    feed_dict[model.input_pos1] = batch_train_pos1
                    feed_dict[model.input_pos2] = batch_train_pos2
                    feed_dict[model.input_y] = batch_train_y
                    feed_dict[model.keep_prob] = settings.keep_prob

                    _,loss,accuracy=sess.run([model.train_op, model.final_loss, model.accuracy],feed_dict=feed_dict)
                    bar.set_description('epoch {} loss={:.6f} accuracy={:.6f}'.format(epoch, loss, accuracy))
                    #break
                saver.save(sess, save_path=save_path)
                #break





class interaction():

    def __init__(self,sess,save_path ='model/model.ckpt3'):

        self.settings = Settings()
        wordembedding = np.load('./data/vec.npy')

        self.sess = sess
        with tf.variable_scope("model"):
            self.model = CNN(word_embeddings=wordembedding, setting=self.settings)

        self.saver = tf.train.Saver()
        self.saver.restore(self.sess,save_path)

        self.train_word = np.load('./data/train_word.npy')
        self.train_pos1 = np.load('./data/train_pos1.npy')
        self.train_pos2 = np.load('./data/train_pos2.npy')
        self.y_train = np.load('data/train_y.npy')


    def reward(self,batch_test_word,batch_test_pos1,batch_test_pos2,batch_test_y):

        feed_dict = {}
        feed_dict[self.model.input_word] = batch_test_word
        feed_dict[self.model.input_pos1] = batch_test_pos1
        feed_dict[self.model.input_pos2] = batch_test_pos2
        feed_dict[self.model.input_y] = batch_test_y
        feed_dict[self.model.keep_prob] = 1
        outputs = (self.sess.run(self.model.reward,feed_dict = feed_dict))
        return (outputs)


    def sentence_ebd(self,batch_test_word,batch_test_pos1,batch_test_pos2,batch_test_y):
        feed_dict = {}
        feed_dict[self.model.input_word] = batch_test_word
        feed_dict[self.model.input_pos1] = batch_test_pos1
        feed_dict[self.model.input_pos2] = batch_test_pos2
        feed_dict[self.model.input_y] = batch_test_y
        feed_dict[self.model.keep_prob] = 1
        outputs = self.sess.run(self.model.sentence,feed_dict = feed_dict)
        return  (outputs)

    def test(self,batch_test_word,batch_test_pos1,batch_test_pos2):
        feed_dict = {}
        feed_dict[self.model.input_word] = batch_test_word
        feed_dict[self.model.input_pos1] = batch_test_pos1
        feed_dict[self.model.input_pos2] = batch_test_pos2
        feed_dict[self.model.keep_prob] = 1
        relation,prob = self.sess.run([self.model.pred,self.model.pred_prob],feed_dict = feed_dict)

        return (relation,prob)

    def update_cnn(self,update_word,update_pos1,update_pos2,update_y,updaterate):

        num_steps = len(update_word) // self.settings.batch_size

        with self.sess.as_default():

            tvars_old = self.sess.run(self.model.tvars)

            for i in tqdm(range(num_steps)):

                batch_word = update_word[i* self.settings.batch_size:(i+1)*self.settings.batch_size]
                batch_pos1 = update_pos1[i* self.settings.batch_size:(i+1)*self.settings.batch_size]
                batch_pos2 = update_pos2[i* self.settings.batch_size:(i+1)*self.settings.batch_size]
                batch_y    = update_y[i* self.settings.batch_size:(i+1)*self.settings.batch_size]

                feed_dict = {}
                feed_dict[self.model.input_word] = batch_word
                feed_dict[self.model.input_pos1] = batch_pos1
                feed_dict[self.model.input_pos2] = batch_pos2
                feed_dict[self.model.input_y]    = batch_y
                feed_dict[self.model.keep_prob] = self.settings.keep_prob
                #_, loss, accuracy = sess.run([self.model.train_op,self.model.final_loss, self.model.accuracy], feed_dict=feed_dict)
                self.sess.run(self.model.train_op, feed_dict=feed_dict)

            # get tvars_new
            tvars_new = self.sess.run(self.model.tvars)

            # update old variables of the target network
            tvars_update = self.sess.run(self.model.tvars)
            for index, var in enumerate(tvars_update):
                tvars_update[index] = updaterate * tvars_new[index] + (1 - updaterate) * tvars_old[index]

            feed_dict = dictionary = dict(zip(self.model.tvars_holders, tvars_update))
            self.sess.run(self.model.update_tvar_holder, feed_dict)

    def produce_new_embedding(self):

        # produce reward sentence_ebd  average_reward
        train_word = self.train_word
        train_pos1 = self.train_pos1
        train_pos2 = self.train_pos2
        y_train = self.y_train
        all_sentence_ebd = []
        all_reward = []
        all_reward_list = []
        len_batch = len(train_word)

        with self.sess.as_default():

            for batch in tqdm(range(len_batch)):
                batch_word = train_word[batch]
                batch_pos1 = train_pos1[batch]
                batch_pos2 = train_pos2[batch]
                # batch_y = train_y[batch]
                batch_y = [y_train[batch] for x in range(len(batch_word))]

                tmp_sentence_ebd = self.sentence_ebd(batch_word, batch_pos1, batch_pos2, batch_y)
                tmp_reward = self.reward(batch_word, batch_pos1, batch_pos2, batch_y)

                all_sentence_ebd.append(tmp_sentence_ebd)
                all_reward.append(tmp_reward)
                all_reward_list += list(tmp_reward)

            all_reward_list = np.array(all_reward_list)
            average_reward = np.mean(all_reward_list)
            average_reward = np.array(average_reward)

            all_sentence_ebd = np.array(all_sentence_ebd)
            all_reward = np.array(all_reward)

            return average_reward,all_sentence_ebd,all_reward

    def save_cnnmodel(self,save_path):
        with self.sess.as_default():
            self.saver.save(self.sess, save_path=save_path)

    def tvars(self):
        with self.sess.as_default():
            tvars = self.sess.run(self.model.tvars)
            return tvars

    def update_tvars(self,tvars_update):
        with self.sess.as_default():
            feed_dict = dictionary = dict(zip(self.model.tvars_holders, tvars_update))
            self.sess.run(self.model.update_tvar_holder, feed_dict)


# produce reward sentence_ebd  average_reward
def produce_rldata(save_path):

    with tf.Graph().as_default():
        sess = tf.Session()
        with sess.as_default():
            # start = time.time()
            interact = interaction(sess, save_path)
            average_reward, all_sentence_ebd, all_reward = interact.produce_new_embedding()

            np.save('data/average_reward.npy', average_reward)
            np.save('data/all_sentence_ebd.npy', all_sentence_ebd)
            np.save('data/all_reward.npy', all_reward)

            print (average_reward)


if __name__ == '__main__':



    # train model
    print ('train model')
    train('cnndata/cnn_train_word.npy', 'cnndata/cnn_train_pos1.npy', 'cnndata/cnn_train_pos2.npy','cnndata/cnn_train_y.npy','model/origin_cnn_model.ckpt')

    # produce reward sentence_ebd  average_reward for rlmodel
    print ('produce reward sentence_ebd  average_reward for rlmodel')
    produce_rldata(save_path='model/origin_cnn_model.ckpt')







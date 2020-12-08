import os
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import random

class PCAPlotter(tf.keras.callbacks.Callback):
    
    def __init__(self, plt, embedding_model, x_test, y_test):
        super(PCAPlotter, self).__init__()
        self.embedding_model = embedding_model
        self.x_test = x_test
        self.y_test = y_test
        self.fig = plt.figure(figsize=(9, 4))
        self.ax1 = plt.subplot(1, 2, 1)
        self.ax2 = plt.subplot(1, 2, 2)
        plt.ion()
        
        self.losses = []
    
    def plot(self, epoch=None, plot_loss=False):
        x_test_embeddings = self.embedding_model.predict(self.x_test)
        pca_out = PCA(n_components=2).fit_transform(x_test_embeddings)
        self.ax1.clear()
        self.ax1.scatter(pca_out[:, 0], pca_out[:, 1], c=self.y_test, cmap='seismic')
        if plot_loss:
            self.ax2.clear()
            self.ax2.plot(range(epoch), self.losses)
            self.ax2.set_xlabel('Epochs')
            self.ax2.set_ylabel('Loss')
        self.fig.canvas.draw()
    
    def on_train_begin(self, logs=None):
        self.losses = []
        self.fig.show()
        self.fig.canvas.draw()
        self.plot()
        
    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs.get('loss'))
        self.plot(epoch+1, plot_loss=True)

# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
# print(x_train.shape)

# x_train = np.reshape(x_train, (x_train.shape[0], 784))/255.
# x_test = np.reshape(x_test, (x_test.shape[0], 784))/255.
# print(x_train.shape)
# import pdb; pdb.set_trace()

def triplet_loss(anchor, positive, negative):
    #anchor, positive, negative = y_pred[:,:emb_size], y_pred[:,emb_size:2*emb_size], y_pred[:,2*emb_size:]
    positive_dist = tf.reduce_mean(tf.square(anchor - positive), axis=1)
    negative_dist = tf.reduce_mean(tf.square(anchor - negative), axis=1)
    return tf.maximum(positive_dist - negative_dist + alpha, 0.)

def get_data(din):
    bf = pd.read_csv(din,sep='\t')
    genes = bf['Hugo_Symbol']
    cols = list(bf.columns)
    cols.remove('Entrez_Gene_Id')
    cols.remove('Hugo_Symbol')
    bf = bf[cols]
    bf = bf.transpose()
    bf.columns = genes
    return bf

opts = [['breast','brca'], ['liver','lihc'], ['lung','luad'],['prostate','prad'],['stomach','stad'],['thyroid','thca']]
dirin = 'data/'
keep_info = .1
beta = .999
loss_1 = 0
for opt in opts:
    din1 = dirin + opt[0] + '-rsem-fpkm-gtex.txt'
    din2 = dirin + opt[1] + '-rsem-fpkm-tcga.txt'
    din3 = dirin + opt[1] + '-rsem-fpkm-tcga-t.txt'
    if os.path.isfile(din1) and os.path.isfile(din2) and os.path.isfile(din3):
        normal1 = get_data(din1)
        normal2 = get_data(din2)
        abnormal = get_data(din3)     
        common_gene = set(list(normal1.columns)).intersection(set(list(normal2.columns))).intersection(set(list(abnormal.columns)))
        common_gene = list(common_gene)
        normal1 = normal1[common_gene]
        normal2 = normal2[common_gene]
        abnormal = abnormal[common_gene]
        loss_1 = triplet_loss(normal1.values, normal2.values, abnormal.values)
        x_train = normal1.values
        x_test = pd.concat( [normal2, abnormal] ).values
        y_test = np.zeros(len(x_test), dtype=int)
        y_test[len(normal2):] = 1
        y_train = np.zeros(len(x_train), dtype=int)
        y_train[len(normal1):] = 1
#         x_train = np.reshape(x_train, (x_train.shape[0], 784))/255.
#         x_test = np.reshape(x_test, (x_test.shape[0], 784))/255.
        

# def plot_triplets(examples):
#     plt.figure(figsize=(6, 2))
#     for i in range(3):
#         plt.subplot(1, 3, 1 + i)
#         plt.imshow(np.reshape(examples[i], (28, 28)), cmap='binary')
#         plt.xticks([])
#         plt.yticks([])
#     plt.show()
# plot_triplets([x_train[0], x_train[1], x_train[2]])

def create_batch(batch_size=1):
    x_anchors = np.zeros((batch_size, 784))
    x_positives = np.zeros((batch_size, 784))
    x_negatives = np.zeros((batch_size, 784))
    
    
    for i in range(0, batch_size):
        # We need to find an anchor, a positive example and a negative example
        random_index = random.randint(0, x_train.shape[0] - 1)
        x_anchor = x_train[random_index]
        y = y_train[random_index]
        
        indices_for_pos = np.squeeze(np.where(y_train == y))
        indices_for_neg = np.squeeze(np.where(y_train != y))
        
        x_positive = x_train[indices_for_pos[random.randint(0, len(indices_for_pos) - 1)]]
        x_negative = x_train[indices_for_neg[random.randint(0, len(indices_for_neg) - 1)]]
        
        x_anchors[i] = x_anchor
        x_positives[i] = x_positive
        x_negatives[i] = x_negative
        #import pdb; pdb.set_trace()
        
    return [x_anchors, x_positives, x_negatives]

examples = create_batch(1)
#plot_triplets(examples)

emb_size = 64

embedding_model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(emb_size, activation='sigmoid')
])

embedding_model.summary()

example = np.expand_dims(x_train[0], axis=0)
example_emb = embedding_model.predict(example)[0]

print(example_emb)

input_anchor = tf.keras.layers.Input(shape=(784,))
input_positive = tf.keras.layers.Input(shape=(784,))
input_negative = tf.keras.layers.Input(shape=(784,))

embedding_anchor = embedding_model(input_anchor)
embedding_positive = embedding_model(input_positive)
embedding_negative = embedding_model(input_negative)

output = tf.keras.layers.concatenate([embedding_anchor, embedding_positive, embedding_negative], axis=1)

net = tf.keras.models.Model([input_anchor, input_positive, input_negative], output)
net.summary()

alpha = 0.2

def data_generator(batch_size=256):
    while True:
        x = create_batch(batch_size)
        y = np.zeros((batch_size, 3*emb_size))
        yield x, y

batch_size = 2048
epochs = 10
steps_per_epoch = int(x_train.shape[0]/batch_size)

net.compile(loss=loss_1, optimizer='adam')

_ = net.fit(
    data_generator(batch_size),
    steps_per_epoch=steps_per_epoch,
    epochs=epochs, verbose=False,
    callbacks=[PCAPlotter(plt, embedding_model,x_test[:1000], y_test[:1000])]
)
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from Bio.SeqIO import parse 
from Bio.SeqRecord import SeqRecord 
from Bio.Seq import Seq 
from Bio import SeqIO
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from gensim.test.utils import common_texts
from gensim.models import Word2Vec
from   tensorflow.keras import layers


embed = hub.load("https://tfhub.dev/google/Wiki-words-250-with-normalization/2")
# model = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"


file = open("cowpox_virus_database.fasta") 
records = parse(file, "fasta")

for record in records:    
   X_init = record.seq


file_two= open("Aminoacidos_prueba.fasta") 
records_two = parse(file_two, "fasta")

for record in records_two:    
   Y_init = record.seq


file_three= open("Aminoacidos_prueba_2.fasta") 
records_three = parse(file_three, "fasta")

for record in records_three:    
   X_proof = record.seq

def translate(chain):
    table = {
    'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M',
    'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
    'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K',
    'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',
    'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L',
    'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
    'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q',
    'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
    'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V',
    'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
    'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E',
    'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
    'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S',
    'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
    'TAC':'Y', 'TAT':'Y', 'TAA':'_', 'TAG':'_',
    'TGC':'C', 'TGT':'C', 'TGA':'_', 'TGG':'W',}
    
    aminoacid = []
    chain_len = len(chain) 
    counter = 0


    while (len(chain)%3) > 0:
        chain += 'A'
        counter += 1
 
    if len(chain)%3 == 0:
        for i in range(0, len(chain), 3):
            if i < chain_len - counter:
                codon = chain[i : i+3]
                aminoacid.append(table[codon])
            
           
    return aminoacid



def X_convert(chain_letter):

    table = {
    'I': [1], 'M': [2], 'T':[3], 'N': [4], 'K': [5], 
    'S': [6], 'R': [7], 'L':[8], 'P': [9], 'H': [10],
    'Q': [11], 'V': [12], 'A': [13], 'D': [14], 'E': [15],
    'G': [16], 'F': [17], 'Y': [18], '_': [19], 'C': [20], 
    'W': [21],}
    
    aminoacid_number = []

    for i in range(0, len(chain_letter), 1):
            aminoacid_number.append(table[chain_letter[i]]) 

    return aminoacid_number


def Y_convert(chain_letter):

    table = {
    'I': 1, 'M': 2, 'T':3, 'N': 4, 'K': 5, 
    'S': 6, 'R': 7, 'L':8, 'P': 9, 'H': 10,
    'Q': 11, 'V': 12, 'A': 13, 'D': 14, 'E': 15,
    'G': 16, 'F': 17, 'Y': 18, '_': 19, 'C': 20, 
    'W': 21,}
    
    aminoacid_number_y = []

    for i in range(0, len(chain_letter), 1):
            aminoacid_number_y.append(table[chain_letter[i]]) 

    return aminoacid_number_y



X = np.array(X_convert(translate(X_init)))
y = np.array(Y_convert(Y_init))
X_pf = np.array(X_convert(X_proof))


print("Arreglo de las cadenas de variables independientes")
print(X)
print("Arreglo de los valores objetivos")
print(y)


gnb = GaussianNB()

y_log = gnb.fit(X, y)

y_pred = y_log.predict(X_pf)

print("Se obtiene el vector de predicción el resultado de la predicción")

print(y_pred)

print("Devuelve la precisión media en las etiquetas y los datos de prueba dados.")

acurracy = y_log.score(X , y_pred)

print(acurracy)

print("Se obtiene la probabilidad de los casos")

probability = y_log.predict_proba(X_pf)
print(probability)


#words = translate(X_init)

words = embed(X_init)
y_word = embed(Y_init)
X_pf_word= embed(X_proof)


for i in range(len(words)):
    for j in range(i,len(words)):
        if np.inner(words[i], X_pf_word[j]) == 1.0:
            contador += 1
            print("(",words[i], ",", X_pf_word[j],")")


# X_keras = "".join(words)
# y_keras = "".join(Y_init)
# y_pfkeras = "".join(X_proof)

# print(X_keras)
# print(type(X_keras))

# hub_layer = hub.KerasLayer(model, output_shape=[20], input_shape=[], dtype=tf.string, trainable=True)
# hub_layer(X_keras[:3])

# model = tf.keras.Sequential()
# model.add(hub_layer)
# model.add(tf.keras.layers.Dense(16, activation='relu'))
# model.add(tf.keras.layers.Dense(1))

# model.summary()

# model.compile(optimizer='adam',
#             loss=tf.losses.BinaryCrossentropy(from_logits=True),
#             metrics=[tf.metrics.BinaryAccuracy(threshold=0.0, name='accuracy')])

# x_val = X_keras[:10000]
# partial_x_train = X_keras[10000:]

# y_val = y_keras[:10000]
# partial_y_train = y_keras[10000:]

# history = model.fit(partial_x_train,
#                 partial_y_train,
#                 epochs=40,
#                 batch_size=512,
#                 validation_data=(x_val, y_val),
#                 verbose=1)


# results = model.evaluate(y_keras, y_pfkeras)

# print(results)

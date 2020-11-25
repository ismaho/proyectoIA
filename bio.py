import numpy as np
from Bio.SeqIO import parse 
from Bio.SeqRecord import SeqRecord 
from Bio.Seq import Seq 
from Bio import SeqIO
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

file = open("cowpox_virus_database.fasta") 
records = parse(file, "fasta")

for record in records:    
   print("Id: %s" % record.id) 
   print("Name: %s" % record.name) 
   print("Description: %s" % record.description) 
   print("Annotations: %s" % record.annotations) 
   X_init = record.seq



file_two= open("Aminoacidos_prueba.fasta") 
records_two = parse(file_two, "fasta")

for record in records_two:    
   print("Id: %s" % record.id) 
   print("Name: %s" % record.name) 
   print("Description: %s" % record.description) 
   print("Annotations: %s" % record.annotations) 
   Y_init = record.seq


file_three= open("Aminoacidos_prueba_2.fasta") 
records_three = parse(file_three, "fasta")

for record in records_three:    
   print("Id: %s" % record.id) 
   print("Name: %s" % record.name) 
   print("Description: %s" % record.description) 
   print("Annotations: %s" % record.annotations) 
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
    
    aminoacid = ""
    chain_len = len(chain) 
    counter = 0


    while (len(chain)%3) > 0:
        chain += 'A'
        counter += 1
 
    if len(chain)%3 == 0:
        for i in range(0, len(chain), 3):
            if i < chain_len - counter:
                codon = chain[i : i+3]
                aminoacid += table[codon]
            
           
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




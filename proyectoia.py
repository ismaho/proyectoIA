from Bio.Seq import Seq
from Bio import SeqIO
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
import numpy as np

 ##df = pd.read_csv("data/train.csv") 
 ##df = df.dropna()

for seq_record in SeqIO.parse("cowpox_virus_database.fasta", "fasta"):
      print(seq_record.id)
      print(repr(seq_record.seq))
      print(len(seq_record))
      X = seq_record.seq
      tamaño_x = (len(seq_record))
    
x_number = []

for x in X:
    if x == 'A':
        x_number.append(1)
    
    elif x== 'C':
        x_number.append(2)
    
    elif x== 'G':
        x_number.append(3)
                
    else: 
        x_number.append(4)
   
x_array = np.array(x_number)
x_array_final = x_array.reshape(round(tamaño_x/2),2)
x_array_final = np.concatenate((x_array_final, x_array_final), axis=0)
print(x_array_final)
print(len(x_array_final))
      
for seq_record in SeqIO.parse("cowpox_virus_database.fasta", "fasta"):
      print(seq_record.id)
      print(repr(seq_record.seq))
      print(len(seq_record))
      Y = seq_record.seq

y_number = []


for y in Y:
    if y == 'A':
        y_number.append(1)
    
    elif x== 'C':
        y_number.append(2)
    
    elif x== 'G':
        y_number.append(3)
                
    else: 
        y_number.append(4)
   
y_array_final = np.array(y_number)



X_train, X_test, y_train, y_test = train_test_split(x_array_final , y_array_final, test_size=0.33, random_state=42)
log_reg = LogisticRegression().fit(X_train,y_train)
y_pred = log_reg.predict(X_test)

print(y_test)
print(y_pred)

acurracy = accuracy_score(y_test, y_pred)

print(acurracy)


##my_seq = Seq("AGTACACTGGT")
##print(my_seq)
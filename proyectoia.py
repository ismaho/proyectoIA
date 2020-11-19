from Bio.Seq import Seq
from Bio import SeqIO
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns

 ##df = pd.read_csv("data/train.csv") 
 ##df = df.dropna()


##X_train, X_test, y_train, y_test = train_test_split(x.values, y.values, test_size=0.3, random_state=42)

##log_reg = LogisticRegression().fit(X_train,y_train)
##y_pred = log_reg.predict(X_test)

##accuracy_score(y_test, y_pred)
for seq_record in SeqIO.parse("cowpox_virus_database.fasta", "fasta"):
      print(seq_record.id)
      print(repr(seq_record.seq))
      print(len(seq_record))
      X = seq_record.seq
for seq_record in SeqIO.parse("cowpox_virus_database.fasta", "fasta"):
      print(seq_record.id)
      print(repr(seq_record.seq))
      print(len(seq_record))
      y = seq_record.seq

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
log_reg = LogisticRegression().fit(X_train,y_train)
y_pred = log_reg.predict(X_test)
accuracy_score(y_test, y_pred)
print(accuracy_score)

my_seq = Seq("AGTACACTGGT")
print(my_seq)
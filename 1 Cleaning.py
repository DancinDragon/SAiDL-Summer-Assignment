###PLEASE RUN THE CODE AS IT IS FIRST, THEN RUN IT AGAIN BY MAKING THE INDICATED CHANGES.
import re
import numpy as np

labels_are = ["entailment","contradiction","neutral"]

def get_text(s):
    s = re.sub('\\(', '', s)
    s = re.sub('\\)', '', s)
    s = re.sub('\\s{2,}', ' ', s)
    return s.strip()

def read_it(file_name): 
	##GIVES L ,P,H lists 
	f = open(file_name, 'r')
	rows = []
	
	for row in f.readlines()[1:]:
		rows.append(row.split("\t"))
	labels = []
	premise = []
	hypothesis = []
	f.close()
	for i in range(len(rows)):
		if get_text(str(rows[i][0])) in labels_are:
			labels.append(get_text(str(rows[i][0])))
			premise.append(get_text(str(rows[i][1])))
			hypothesis.append(get_text(str(rows[i][2])))

	return labels,premise,hypothesis  ##COMMENT THIS AFTER INITIAL RUN
	##return premise,hypothesis       ##UNCOMMENT THIS AFTER INTIAL RUN

data = read_it("SNLI.txt")
data = np.array(data)
data=data.transpose()


f = open("Raw_Data.txt","w+")    ##COMMENT THIS AFTER INITIAL RUN
##f = open("Word2Vec.txt","w+")  ##UNCOMMENT THIS AFTER INTIAL RUN AND RUN IT AGAIN
for row in data:
    np.savetxt(f,row,"%s")
f.close()



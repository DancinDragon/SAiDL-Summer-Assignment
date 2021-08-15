import numpy as np
from tqdm import tqdm
def Divider(File_name):
	f = open(f"{File_name}.txt","r")
	Raw_Text = f.read()
	f.close()
	Data0 = Raw_Text.split("\n")
	x = (len(Data0))
	label = []
	premise = []
	hypothesis = []
	for i in tqdm(range(x)):
		if (i%3==0):
			label.append(Data0[i])
		elif (i%3==1):
			premise.append(Data0[i])
		elif (i%3==2):
			hypothesis.append(Data0[i])
	return label,premise,hypothesis

x = Divider("Raw_Data")
labels = ["neutral","entailment","contradiction"]
encoded_labels = [labels.index(x[0][i]) for i in range(len(x[0]))]
"""
premise = x[1]
hypothesis = x[2]

f = open("premise.txt","w+")
np.savetxt(f,premise,"%s")
f.close()

f = open("hypothesis.txt","w+")
np.savetxt(f,hypothesis,"%s")
f.close()
"""

np.save("encoded_labels.npy",np.array(encoded_labels))


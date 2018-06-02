import matplotlib.pyplot as plt
import numpy as np

plt.clf()
filename = ['RLCNN','CNN']
color = ['red', 'blue']

List_Precision = np.load('data/List_Precision.npy')
List_Recall = np.load('data/List_Recall.npy')

for i in range(len(filename)):
	precision = List_Precision[i]
	recall  =   List_Recall[i]
	plt.plot(recall,precision,color = color[i],lw=2,label = filename[i])

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.3, 1.0])
plt.xlim([0.0, 0.4])
plt.title('Precision-Recall')
plt.legend(loc="upper right")
plt.grid(True)
plt.savefig('image/PR_curve')
plt.show()
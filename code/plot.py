import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
plt.style.use("fivethirtyeight")

labels_probas_names = [
	('labels_logit_size20.npy', 'probas_logit_size20.npy', 'Logit', 'darkorange'),
	('labels_svm_size20.npy', 'preds_svm_size20.npy', 'Linear SVM', 'forestgreen')
	]

plt.figure(figsize=(10,10))
lw = 2

for (label_, proba_, name, color) in labels_probas_names:
	labels, probas = np.load(label_), np.load(proba_)
	roc_auc = roc_auc_score(labels, probas)
	fpr, tpr, _ = roc_curve(labels, probas)
	plt.plot(fpr, tpr, color=color,
		lw=lw, label='%s (area = %0.2f)' % (name, roc_auc))

plt.plot([0, 1], [0, 1], color='navy', lw=lw, ls='--', label='Luck')
plt.xlim([-.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for different model sizes')
plt.legend(loc="lower right")
plt.savefig('roc_auc_comparison.png')
plt.tight_layout()
plt.show()
import matplotlib.pyplot as plt
import numpy as np


labels = list(range(1, 6))
rmse = [1.2875, 1.2906, 1.2847, 1.2882, 1.2843]
prec = [0.7645, 0.7696, 0.7689, 0.7703, 0.7686]
coverage = [0.6585, 0.6515, 0.6587, 0.6599, 0.6572]
explain = [1]*5
x = np.arange(len(labels))
width = 0.2

fig, ax = plt.subplots()
ax.grid(True, which='major', axis='y')
ax.set_axisbelow(True)
rects1 = ax.bar(x - (width/2)-width, rmse, width, label='RMSE')
rects2 = ax.bar(x - (width/2), prec, width, label='Precision')
rects3 = ax.bar(x + (width/2), coverage, width, label='Coverage')
rects4 = ax.bar(x + (width/2)+width, explain, width, label='Explainability')
ax.set_ylabel('Score')
ax.set_xlabel('Test Fold')
ax.set_title('Evaluation Results')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
fig.tight_layout()
fig.savefig('Result Graph', format='pdf')

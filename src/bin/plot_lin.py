import numpy as np
from scipy.spatial import distance
import seaborn as sns; sns.set_theme()
import sys

if './' not in sys.path:
    sys.path.append('./')

from src.utils.cka import cka, gram_linear

res = np.zeros(shape=(45,45))
for i in range(16, 44):
  for j in range(16, 44):
    if i > j:
      continue
    
    acts = np.load('results/'+str(i), allow_pickle=True)
    acts = acts[:1000]
    acts = np.where(acts > 0.0, 1.0, 0.0)

    acts = np.reshape(acts, (acts.shape[0], -1))

    actsb = np.load('results/'+str(j), allow_pickle=True)
    actsb = actsb[:1000]
    actsb = np.where(actsb > 0.0, 1.0, 0.0)

    actsb = np.reshape(actsb, (actsb.shape[0], -1))


    cka_from_examples = cka(gram_linear(acts), gram_linear(actsb))
    res[i,j] = cka_from_examples
    print('cka: ', cka_from_examples)

np.save('res_bin', res)

quit()



acts = np.load('results/32', allow_pickle=True)
acts = acts[:1000]
max_pooled_acts = np.max(np.max(acts, axis=-1), axis=-1)
bin_acts = np.where(acts > 0.0, 1.0, 0.0)
bin_acts = np.reshape(bin_acts, (acts.shape[0], -1))
bin_dist = distance.cdist(bin_acts, bin_acts, 'cityblock')


acts_16 = np.load('results/16', allow_pickle=True)
acts_16 = acts_16[:1000]
bin_acts_16 = np.where(acts_16 > 0.0, 1.0, 0.0)
bin_acts_16 = np.reshape(bin_acts_16, (acts_16.shape[0], -1))


#print(bin_dist.shape)

X = bin_acts
Y = bin_acts_16
cka_from_examples = cka(gram_linear(X), gram_linear(Y))
print('cka: ', cka_from_examples)

quit()
ax = sns.heatmap(bin_dist)
fig = ax.get_figure()
fig.savefig("plot_32.pdf")

#print(bin_acts)
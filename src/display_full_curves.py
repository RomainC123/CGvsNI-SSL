import os
import pickle
import matplotlib.pyplot as plt

from ssl.utils.paths import TRAINED_MODELS_PATH

percent_labels = [1, 10, 20, 30, 40, 50, 60, 70, 80]
count = 0

fig1 = plt.figure(figsize=(12, 24))
ax11 = fig1.add_subplot(211)
ax11.set_title('Metrics eval')
ax12 = fig1.add_subplot(212)
ax12.set_title('Metrics test')

fig2 = plt.figure(figsize=(12,24))
ax21 = fig2.add_subplot(311)
ax21.set_title('Loss')
ax22 = fig2.add_subplot(312)
ax22.set_title('Sup loss')
ax23 = fig2.add_subplot(313)
ax23.set_title('Unsup loss')

for model_name in sorted(os.listdir(TRAINED_MODELS_PATH)):

    graphs_path = os.path.join(TRAINED_MODELS_PATH, model_name, 'graphs')

    with open(os.path.join(graphs_path, 'metrics_eval.pkl'), 'rb') as f:
        metrics_eval = pickle.load(f)
    with open(os.path.join(graphs_path, 'metrics_test.pkl'), 'rb') as f:
        metrics_test = pickle.load(f)
    with open(os.path.join(graphs_path, 'loss.pkl'), 'rb') as f:
        losses = pickle.load(f)
    with open(os.path.join(graphs_path, 'sup_loss.pkl'), 'rb') as f:
        sup_losses = pickle.load(f)
    with open(os.path.join(graphs_path, 'unsup_loss.pkl'), 'rb') as f:
        unsup_losses = pickle.load(f)

    for key in metrics_eval.keys():
        ax11.plot(range(len(metrics_eval[key])), metrics_eval[key], label=key.capitalize() + ' eval ' + str(percent_labels[count]))
    for key in metrics_test.keys():
        ax12.plot(range(len(metrics_test[key])), metrics_test[key], label=key.capitalize() + 'test ' + str(percent_labels[count]))

    ax21.plot(range(len(losses)), losses, label='Total loss' + str(percent_labels[count]))
    ax22.plot(range(len(sup_losses)), sup_losses, label='Supervised loss' + str(percent_labels[count]))
    ax23.plot(range(len(unsup_losses)), unsup_losses, label='Unsupervised loss' + str(percent_labels[count]))


    count += 1

ax11.legend()
ax12.legend()
ax21.legend()
ax22.legend()
ax23.legend()

plt.show()

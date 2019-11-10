"""
results.py
import to plot and save learning curves as png files
as well as the corresponding training logs and most
importantly the generated music sample
"""

import matplotlib.pyplot as plt
import numpy as np

def save_plot_results(epochs, train_loss, valid_loss, title, file_name, directory='./'):
    """
    saves learning curves from training
    """
    epochs = np.array(epochs)
    train_loss = np.array(train_loss)
    valid_loss = np.array(valid_loss)
    line1, = plt.plot(epochs, train_loss, 'b', label='Training Loss')
    line2, = plt.plot(epochs, valid_loss, 'g', label='Validation Loss')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.suptitle(title, fontsize=12)
    plt.legend([line1, line2], ['Training Loss', 'Validation Loss'], loc=1)
    plt.yticks(np.arange(0.0, 4.0, .5))
    plt.text(1, 0.1, "Validation Loss = {:.4f}".format(min(valid_loss)))
    plt.savefig(directory + file_name)
    plt.clf()
#    plt.show()
    
def save_log(log_text, file_name, directory='./'):
    """
    saves training log
    """
    f = open(directory + file_name, 'w+')
    f.writelines(log_text)
    f.close()
    
def save_music(music_text, file_name, directory='./'):
    """
    saves generated sample
    """
    f = open(directory + file_name, 'w+')
    f.writelines(music_text)
    f.close()
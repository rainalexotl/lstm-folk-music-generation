tokenized = False #set to false if text file is NOT separated into tokens
batch_size = 128
seq_length = 150
n_epochs = 100
grad_clip = 5

n_layers = 3
n_hidden = 512
lr = 0.001
val_frac = 0.14
dropout = 0.4

print_every = 50
path = './datasets/nottingham_database/nottingham_parsed.txt'

import torch
import torch.nn.functional as F
import numpy as np
import data_proc as dp
import music_train as mt
import re
import os
import model_configuration as conf

sample_dir = './samples/'

def load_model(model_path):
    """
    loads a pretrained model from model_path
    """
    print("Loading model...", end=" ")
    checkpoint = torch.load(model_path) 
    model = mt.musicRNN(checkpoint['tokens'], checkpoint['n_hidden'], checkpoint['n_layers'], conf.lr) 
    model.load_state_dict(checkpoint['state_dict']) 
    model.eval()
    print("Done.")
    return model

model = load_model('./model/musicmodel.pth')

def predict(model, char, h=None, top_k=None):
        """
        Given a character, predict the next character.
        Returns the predicted character and the hidden state.
        """
        # tensor inputs
        x = np.array([[model.char2int[char]]])
        x = dp.one_hot_from_vocab(x, len(model.tokens))
        inputs = torch.from_numpy(x)
        
        if(mt.gpu_available):
            inputs = inputs.cuda()
        
        # detach hidden state from history
        h = tuple([each.data for each in h])
        out, h = model(inputs, h)

        # get the probability of each token
        p = F.softmax(out, dim=1).data
        if(mt.gpu_available):
            p = p.cpu()
        
        # get top characters
        if top_k is None:
            top_ch = np.arange(len(model.tokens))
        else:
            p, top_ch = p.topk(top_k)
            top_ch = top_ch.numpy().squeeze()
        
        # select the likely next character with some element of randomness
        p = p.numpy().squeeze()
        char = np.random.choice(top_ch, p=p/p.sum())
        
        
        # return the encoded value of the predicted char and the hidden state
        return model.int2char[char], h
    
def prep_prompt(prompt):
    """
    prepares the prompt for sampling
    """
    return re.findall(r'<\/?\w>|.', prompt, re.DOTALL)
   
def valid_sample(sample):
    """
    checks the syntax of the generated tune
    """
    if re.match('M:[0-9]\/[0-9]\nK:[A-Z]*\n', sample):
        return True
    return False

def sample(model, prompt='</s>', top_k=None, eos_tok='</s>'):
    """
    generates sequence of characters based on hte trained model and the prompt
    """    

    if(mt.gpu_available):
        model.cuda()
    else:
        model.cpu()
    tokens = prep_prompt(prompt)
    h = model.init_hidden(1)
    for t in tokens:
        char, h = predict(model, t, h, top_k=top_k)

    tokens.append(char)
    
    # Now pass in the previous character and get a new one
    char = ''
    while(char != eos_tok):
        char, h = predict(model, tokens[-1], h, top_k=top_k)
        tokens.append(char)
        
    return ''.join(tokens).replace(eos_tok, '')

def save_sample(sample, file_name):
    sample = "X:1\n" + sample
    f = open(f'{sample_dir}{file_name}.abc', 'w+')
    f.writelines(sample)
    f.close()
    print(f'{sample_dir}{file_name}.abc generated and saved!')

   
#prompt = '</s>'
#song = sample(model, prompt, 5)
#print(song)
#save_sample(song, 'eos6')
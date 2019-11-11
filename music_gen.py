import torch
import torch.nn.functional as F
import numpy as np
import data_proc as dp
import music_train as mt
import re
import sys
import getopt
import model_configuration as conf

gpu_available = torch.cuda.is_available()

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



def predict(model, char, h=None, top_k=None):
    """
    Given a character, predict the next character.
    Returns the predicted character and the hidden state.
    """
    # tensor inputs
    x = np.array([[model.char2int[char]]])
    x = dp.one_hot_from_vocab(x, len(model.tokens))
    inputs = torch.from_numpy(x)
    
    if(gpu_available):
        inputs = inputs.cuda()
    
    # detach hidden state from history
    h = tuple([each.data for each in h])
    out, h = model(inputs, h)

    # get the probability of each token
    p = F.softmax(out, dim=1).data
    if(gpu_available):
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
    if re.match('M:[0-9]\/[0-9]\nK:[A-G]+m*\n[\d\D]+', sample):
        return True
    return False

def sample(model, prompt='</s>', top_k=None, eos_tok='</s>'):
    """
    creates a succession of predicted characters based on a given prompt
    until the eos token is predicted
    """
    if(gpu_available):
        model.cuda()
    else:
        model.cpu()
        
    while True: # until a tune of valid syntax is generated
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
        tune = ''.join(tokens).replace(eos_tok, '')
        if valid_sample(tune):
            break
    return tune
    

def save_sample(sample, file_path):
    """
    saves the sample
    """
    sample = "X:1\n" + sample
    f = open(file_path, 'w+')
    f.writelines(sample)
    f.close()
    print(f'{file_path} generated and saved!')
    

def main(argv):
    model_path = './model/musicmodel.pth'
    file_path = 'sample.abc'
    prompt = '</s>'
    
    try:
        opts, args = getopt.getopt(argv, 'm:f:p:',['modelpath=', 'filepath=', 'prompt='])
    except getopt.GetoptError:
        print("Usage: python3 music_gen.py -m <model_path> [-f file_path] [-p prompt]")
        sys.exit()
        
    for opt, arg in opts:
        if (opt in ('-m', '--modelpath')):
            model_path = arg
        elif (opt in ('-f', '--filepath')):
            file_path = arg
        elif (opt in ('-p', '--prompt')):
            prompt = arg
            
    if (gpu_available):
        print("GPU available. Training on GPU!")
    else:
        print("GPU unavailable. Training on CPU")
    
    model = load_model(model_path)
    song = sample(model, prompt, 5)
    print(song)
    save_sample(song, file_path)
            
if __name__ == "__main__":
    main(sys.argv[1:])
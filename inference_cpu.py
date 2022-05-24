import os
import glob
import numpy as np
import pandas as pd

import torch
from model import SiameseNetwork

def one_hot_encoding(wine):
        '''
        len 22 vector
        5 + 5 + 5 + 5 + 5
        '''
        wine_vector = torch.zeros((5 * 5))
        for i, value in enumerate(wine[1:]):
            # print(i, value)
            # print(i*5 + (int(value)-1))
            wine_vector[i*5 + (int(value)-1)] = 1
            
        return wine_vector

def normalize_encoding(wine):
        '''
        '''
        wine_vector = torch.tensor(wine[1:])
        wine_vector = (wine_vector-3) / 2
        return wine_vector.float()

def main():
    frame = pd.read_csv('data/sample_cleansingWine100.csv')
    np_frame = frame.to_numpy()
    wines = None
    for wine in np_frame:
        if wines == None:
            wines = normalize_encoding(wine)
            wines = wines.view(1, -1) #B x C
        else:
            wines = torch.cat([wines, normalize_encoding(wine).view(1, -1)], dim=0)
    
    model = SiameseNetwork()

    checkpoint_path = 'results/base_single_100_loss_same/best.pth.tar'
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        
        # state_dict control
        checkpoint_dict = checkpoint['state_dict']
        model_dict = model.state_dict()

        assert set(model_dict)==set(checkpoint_dict)

        model_dict.update(checkpoint_dict)
        model.load_state_dict(model_dict, strict=True)

    else:
        raise AssertionError('no weights file')
    
    model.eval()
    loss_fn_1 = torch.nn.BCEWithLogitsLoss()
    user_taste = [000, 3.0,3.0,3.0,3.0,3.0]
    user_taste_tensor = normalize_encoding(user_taste)
    user_taste_tensor = user_taste_tensor.view(1, -1)
    user_taste_tensor = user_taste_tensor.expand(100, 5)

    output = model(user_taste_tensor, wines)
    output = output.sigmoid()
    output = output.view(100).cpu().detach().numpy()
    topidx = np.argpartition(output, -5)[-5:]
    difftopidx = np.argpartition(output, 5)[:5]
    topidx = topidx[np.argsort(output[topidx])]
    topidx = topidx[::-1]

    difftopidx = difftopidx[np.argsort(output[difftopidx])]
    
    # topidx = sorted(range(len(output)),key= lambda i: output[i])[:5]
    # difftopidx = sorted(range(len(output)),key= lambda i: output[i])[-5:]
    #print(topidx)
    

    full_wine_info = pd.read_csv('data/wine_100name.csv')
    full_wine_info = full_wine_info.to_numpy()
    '''
    columns
        [0] = id
        [1] = name
        [2] = producer
        [3] = nation
        [4] = abv
        [5] = degree
        [6] = sweet
        [7] = acidity
        [8] = body
        [9] = tannin
        [10] = Type
        [11] = price
        [12] = year
        [13] = ml
    '''    
    diff_top_5_wine = full_wine_info[difftopidx]
    diff_distances = output[difftopidx]
    top_5_wine = full_wine_info[topidx]
    distances = output[topidx]
    msg = "SWEET : {}\nACIDITY : {}\nBODY : {}\nTANNIN : {}\nTYPE : {}"
    print("User Taste :\n" + msg.format(user_taste[1], user_taste[2], user_taste[3] ,user_taste[4], user_taste[5]))
    print("\n\n")
    for i, top_wine in enumerate(top_5_wine):
        print(f"TOP : {i+1}\nNAME : {top_wine[1]}\n" + msg.format(top_wine[6], top_wine[7], top_wine[8] ,top_wine[9], top_wine[10]))
        print(f"Similarity : {distances[i].item()}")
        print("\n\n")
    for i, top_wine in enumerate(diff_top_5_wine):
        print(f"DIFFTOP : {i+1}\nNAME : {top_wine[1]}\n" + msg.format(top_wine[6], top_wine[7], top_wine[8] ,top_wine[9], top_wine[10]))
        print(f"Similarity : {diff_distances[i].item()}")
        print("\n\n")
    
if __name__ == '__main__':
    main()
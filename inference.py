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
        wine_vector = wine_vector / 5
        return wine_vector.float()

def main():
    frame = pd.read_csv('data/sample_cleansingWine100.csv')
    np_frame = frame.to_numpy()
    wines = None
    for wine in np_frame:
        if wines == None:
            wines = normalize_encoding(wine)
            wines = wines.view(1, -1).to('cuda') #B x C
        else:
            wines = torch.cat([wines, normalize_encoding(wine).view(1, -1).to('cuda')], dim=0)
    
    model = SiameseNetwork()
    model = model.to('cuda')

    checkpoint_path = 'results/base_single_100_loss/best.pth.tar'
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage.cuda())
        
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
    user_taste = [000, 1.0,4.0,4.0,1.0,1.0]
    user_taste_tensor = normalize_encoding(user_taste)
    user_taste_tensor = user_taste_tensor.view(1, -1).to('cuda')
    user_taste_tensor = user_taste_tensor.expand(100, 5).to('cuda')
    #print(user_taste == wines)
    # for i in range(len(wines)):
    #     output = model(user_taste, wines[i].view(1,-1))
    #     print(output)
    #     loss_1 = loss_fn_1(output, torch.ones(output.shape).to('cuda'))
    #     print("loss : ", loss_1)

    #     loss_2 = loss_fn_1(output, torch.zeros(output.shape).to('cuda'))
    #     print("loss : ", loss_2)
    # print("wines shape : ", wines.shape) #100 x 25
    # print("user taste shape : ", user_taste.shape) #100 x 25

    output = model(user_taste_tensor, wines)
    output = output.sigmoid()
    difftopidx = sorted(range(len(output)),key= lambda i: output[i])[:5]
    topidx = sorted(range(len(output)),key= lambda i: output[i])[-5:]
    #print(topidx)
    

    full_wine_info = pd.read_csv('data/wine_100name.csv')
    full_wine_info = full_wine_info.to_numpy()
    '''
    columns
        [0] = id
        [1] = name
        [2] = producer
        [3] = nation
        [4] = type
        [5] = use
        [6] = abv
        [7] = degree
        [8] = sweet
        [9] = acidity
        [10] = body
        [11] = tannin
        [12] = price
        [13] = year
        [14] = ml
    '''    
    diff_top_5_wine = full_wine_info[difftopidx]
    top_5_wine = full_wine_info[topidx]
    diff_distances = output[difftopidx]
    distances = output[topidx]
    msg = "SWEET : {}\nACIDITY : {}\nBODY : {}\nTANNIN : {}\nTYPE : {}"
    print("User Taste :\n" + msg.format(user_taste[1], user_taste[2], user_taste[3] ,user_taste[4], user_taste[5]))
    print("\n\n")
    for i, top_wine in enumerate(top_5_wine):
        print(f"TOP : {i+1}\nNAME : {top_wine[1]}\n" + msg.format(top_wine[8], top_wine[9], top_wine[10] ,top_wine[11], top_wine[4]))
        print(f"Similarity : {distances[i].item()}")
        print("\n\n")
    for i, top_wine in enumerate(diff_top_5_wine):
        print(f"DIFFTOP : {5-i}\nNAME : {top_wine[1]}\n" + msg.format(top_wine[8], top_wine[9], top_wine[10] ,top_wine[11], top_wine[4]))
        print(f"Similarity : {diff_distances[i].item()}")
        print("\n\n")
    
if __name__ == '__main__':
    main()
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

    checkpoint_path = 'results/base_single_100_normalize/best.pth.tar'
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
    user_taste = torch.tensor([000, 1.0,4.0,4.0,1.0,2.0])
    user_taste = normalize_encoding(user_taste)
    user_taste = user_taste.view(1, -1).to('cuda')
    user_taste = user_taste.expand(100, 5).to('cuda')
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

    output = model(user_taste, wines)
    topidx = sorted(range(len(output)),key= lambda i: output[i])[:5]
    print(topidx)
    print(np_frame[topidx])
    
    top_5_wine = np_frame[topidx]
    top_5_wine_id = top_5_wine[:, 0]
    print(top_5_wine_id)



if __name__ == '__main__':
    main()
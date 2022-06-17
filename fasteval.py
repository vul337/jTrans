import pickle
import sys
from datautils.playdata import DatasetBase as DatasetBase
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import argparse

def eval_O(ebds,TYPE1,TYPE2):
    funcarr1=[]
    funcarr2=[]

    for i in range(len(ebds)):
        if ebds[i].get(TYPE1) is not None and type(ebds[i][TYPE1]) is not int:
            if ebds[i].get(TYPE2) is not None and type(ebds[i][TYPE2]) is not int:
                ebd1,ebd2=ebds[i][TYPE1],ebds[i][TYPE2]
                funcarr1.append(ebd1 / ebd1.norm())
                funcarr2.append(ebd2 / ebd2.norm())
        else:
            continue

    ft_valid_dataset=FunctionDataset_Fast(funcarr1,funcarr2)
    dataloader = DataLoader(ft_valid_dataset, batch_size=POOLSIZE, num_workers=24, shuffle=True)
    SIMS=[]
    Recall_AT_1=[]

    for idx, (anchor,pos) in enumerate(tqdm(dataloader)):
        anchor = anchor.cuda()
        pos =pos.cuda()
        if anchor.shape[0]==POOLSIZE:
            for i in range(len(anchor)):    # check every vector of (vA,vB)
                vA=anchor[i:i+1]  #pos[i]
                sim = np.array(torch.mm(vA, pos.T).cpu().squeeze())
                y=np.argsort(-sim)
                posi=0
                for j in range(len(pos)):
                    if y[j]==i:
                        posi=j+1
                        break 
                if posi==1:
                    Recall_AT_1.append(1)
                else:
                    Recall_AT_1.append(0)
                SIMS.append(1.0/posi)
    print(TYPE1,TYPE2,'MRR{}: '.format(POOLSIZE),np.array(SIMS).mean())
    print(TYPE1,TYPE2,'Recall@1: ', np.array(Recall_AT_1).mean())
    return np.array(Recall_AT_1).mean()

class FunctionDataset_Fast(torch.utils.data.Dataset): 
    def __init__(self,arr1,arr2): 
        self.arr1=arr1
        self.arr2=arr2
        assert(len(arr1)==len(arr2))
    def __getitem__(self, idx):            
        return self.arr1[idx].squeeze(0),self.arr2[idx].squeeze(0)
    def __len__(self):
        return len(self.arr1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="jTrans-FastEval")
    parser.add_argument("--experiment_path", type=str, default='./experiments/BinaryCorp-3M/jTrans.pkl', help="experiment to be evaluated")
    parser.add_argument("--poolsize", type=int, default=32, help="size of the function pool")
    args = parser.parse_args()

    POOLSIZE=args.poolsize
    ff=open(args.experiment_path,'rb')
    ebds=pickle.load(ff)
    ff.close()

    print(f'evaluating...poolsize={POOLSIZE}')

    eval_O(ebds,'O0','O3')
    eval_O(ebds,'O0','Os')
    eval_O(ebds,'O1','Os')
    eval_O(ebds,'O1','O3')
    eval_O(ebds,'O2','Os')
    eval_O(ebds,'O2','O3')
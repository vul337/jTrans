from transformers import BertTokenizer, BertForMaskedLM, BertModel
from tokenizer import *
import pickle
from torch.utils.data import DataLoader
import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from data import help_tokenize, load_paired_data,FunctionDataset_CL
from transformers import AdamW
import torch.nn.functional as F
import argparse
import wandb
import logging
import sys
import time
import data
WANDB = True

def get_logger(name):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', filename=name)
    logger = logging.getLogger(__name__)
    s_handle = logging.StreamHandler(sys.stdout)
    s_handle.setLevel(logging.INFO)
    s_handle.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(filename)s[:%(lineno)d] - %(message)s"))
    logger.addHandler(s_handle)
    return logger

def eval(model, args, valid_set, logger):

    if WANDB:
        wandb.init(project=f'jTrans-finetune')
        wandb.config.update(args)
    logger.info("Initializing Model...")
    device = torch.device("cuda")
    model.to(device)
    logger.info("Finished Initialization...")
    valid_dataloader = DataLoader(valid_set, batch_size=args.eval_batch_size, num_workers=24, shuffle=True)
    global_steps = 0
    etc=0
    logger.info(f"Doing Evaluation ...")
    mrr = finetune_eval(model, valid_dataloader)
    logger.info(f"Evaluate: mrr={mrr}")
    if WANDB:
        wandb.log({
                    'mrr': mrr
                })

def finetune_eval(net, data_loader):
    net.eval()
    print(net)
    with torch.no_grad():
        avg=[]
        gt=[]
        cons=[]
        eval_iterator = tqdm(data_loader)
        for i, (seq1,seq2,seq3,mask1,mask2,mask3) in enumerate(eval_iterator):
                input_ids1, attention_mask1= seq1.cuda(),mask1.cuda()
                input_ids2, attention_mask2= seq2.cuda(),mask2.cuda()
                print(input_ids1.shape)
                print(attention_mask1.shape)
                anchor,pos=0,0

                output=net(input_ids=input_ids1,attention_mask=attention_mask1)
                #anchor=output.last_hidden_state[:,0:1,:]
                anchor=output.pooler_output
                output=net(input_ids=input_ids2,attention_mask=attention_mask2)
                #pos=output.last_hidden_state[:,0:1,:]
                pos=output.pooler_output
                ans=0
                for k in range(len(anchor)):    # check every vector of (vA,vB)
                    vA=anchor[k:k+1].cpu()
                    sim=[]
                    for j in range(len(pos)):
                        vB=pos[j:j+1].cpu()
                        #vB=vB[0]
                        AB_sim=F.cosine_similarity(vA, vB).item()
                        sim.append(AB_sim)
                        if j!=k:
                            cons.append(AB_sim)
                    sim=np.array(sim)
                    y=np.argsort(-sim)
                    posi=0
                    for j in range(len(pos)):
                        if y[j]==k:
                            posi=j+1

                    gt.append(sim[k])

                    ans+=1/posi

                ans=ans/len(anchor)
                avg.append(ans)
                print("now mrr ",np.mean(np.array(avg)))
        fi=open("logft.txt","a")
        print("MRR ",np.mean(np.array(avg)),file=fi)
        print("FINAL MRR ",np.mean(np.array(avg)))
        fi.close()
        return np.mean(np.array(avg))
class BinBertModel(BertModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config
        self.embeddings.position_embeddings=self.embeddings.word_embeddings
from datautils.playdata import DatasetBase as DatasetBase

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="jTrans-EvalSave")
    parser.add_argument("--model_path", type=str, default='./models/jTrans-finetune', help="Path to the model")
    parser.add_argument("--dataset_path", type=str, default='./BinaryCorp/small_test', help="Path to the dataset")
    parser.add_argument("--experiment_path", type=str, default='./experiments/BinaryCorp-3M/jTrans.pkl', help="Path to the experiment")
    parser.add_argument("--tokenizer", type=str, default='./jtrans_tokenizer/')

    args = parser.parse_args()

    from datetime import datetime
    now = datetime.now() # current date and time
    TIMESTAMP="%Y%m%d%H%M"
    tim = now.strftime(TIMESTAMP)
    logger = get_logger(f"jTrans-{args.model_path}-eval-{args.dataset_path}_savename_{args.experiment_path}_{tim}")
    logger.info(f"Loading Pretrained Model from {args.model_path} ...")
    model = BinBertModel.from_pretrained(args.model_path)

    model.eval()
    device = torch.device("cuda")
    model.to(device)

    logger.info("Done ...")
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer)
    logger.info("Tokenizer Done ...")
   
    logger.info("Preparing Datasets ...")
    ft_valid_dataset=FunctionDataset_CL(tokenizer,args.dataset_path,None,True,opt=['O0', 'O1', 'O2', 'O3', 'Os'], add_ebd=True, convert_jump_addr=True)
    for i in tqdm(range(len(ft_valid_dataset.datas))):
        pairs=ft_valid_dataset.datas[i]
        for j in ['O0','O1','O2','O3','Os']:
            if ft_valid_dataset.ebds[i].get(j) is not None:
                idx=ft_valid_dataset.ebds[i][j]
                ret1=tokenizer([pairs[idx]], add_special_tokens=True,max_length=512,padding='max_length',truncation=True,return_tensors='pt') #tokenize them
                seq1=ret1['input_ids']
                mask1=ret1['attention_mask']
                input_ids1, attention_mask1= seq1.cuda(),mask1.cuda()
                output=model(input_ids=input_ids1,attention_mask=attention_mask1)
                anchor=output.pooler_output
                ft_valid_dataset.ebds[i][j]=anchor.detach().cpu()

    logger.info("ebds start writing")
    fi=open(args.experiment_path,'wb')
    pickle.dump(ft_valid_dataset.ebds,fi)
    fi.close()


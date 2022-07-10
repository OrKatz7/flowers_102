import data
import utils
import argparse
import importlib
import os
import torch
import pandas as pd
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train pipeline')
    parser.add_argument('--config', type=str , default = 'swin_base_patch4_window12_384')
    parser.add_argument('--seed', type=int , default = 42)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--train', action='store_true')
    
    args = parser.parse_args()
    utils.seed_torch(args.seed)
    OUTPUT_DIR = f'./log_{args.seed}/'
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    CFG = importlib.import_module(f"configs.{args.config}").CFG
    CFG.seed = args.seed
    LOGGER = utils.init_logger(log_file=f'{OUTPUT_DIR}/{args.config}.log')
    folds,test = data.parse_data(path = "../",seed=args.seed)
    
    if args.train:
        utils.train_loop(folds,CFG,LOGGER,OUTPUT_DIR=OUTPUT_DIR,seed=args.seed)
        
    if args.test:
        states = [torch.load(f"{OUTPUT_DIR}/{args.config}_fold0_best.pth")]
        probs,labels = utils.inference(CFG,states,test,LOGGER)
        test['pred'] = probs.argmax(1)
        test.to_csv(f'{OUTPUT_DIR}/{args.config}_test.csv',index=False)
        pd.DataFrame(probs).to_csv(f'{OUTPUT_DIR}/{args.config}_test_probs.csv',index=False)
    
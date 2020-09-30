import json
from random import shuffle
from collections import Counter
import torch
from transformers import BertModel, BertTokenizer
import time
import logging
import argparse
import os
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from maml import Learner  # when using Reptile, then from Reptile import Learner
from task import MetaTask
import random
import numpy as np

SEED = 1
random.seed(SEED) 
np.random.seed(SEED) 
torch.manual_seed(SEED) 
torch.backends.cudnn.deterministic = True

def random_seed(value):
    torch.backends.cudnn.deterministic=True
    torch.manual_seed(value)
    torch.cuda.manual_seed(value)
    np.random.seed(value)
    random.seed(value)

def create_batch_of_tasks(taskset, is_shuffle = True, batch_size = 4):
    idxs = list(range(0,len(taskset)))
    if is_shuffle:
        random.shuffle(idxs)
    for i in range(0,len(idxs), batch_size):
        yield [taskset[idxs[i]] for i in range(i, min(i + batch_size,len(taskset)))]

def main():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data_train", default='train.json', type=str,
                        help="Path to dataset file")

    parser.add_argument("--data_dev", default='dev.json', type=str,
                        help="Path to dataset file")
    
    parser.add_argument("--data_test", default='test.json', type=str,
                        help="Path to dataset file")
    
    parser.add_argument("--bert_model", default='bert-base-cased', type=str,
                        help="Path to bert model")
    
    parser.add_argument("--num_labels", default=4, type=int,
                        help="Number of class for classification")

    parser.add_argument("--epoch", default=5, type=int,
                        help="Number of outer interation")
    
    parser.add_argument("--k_spt", default=80, type=int,
                        help="Number of support samples per task")
    
    parser.add_argument("--k_qry", default=20, type=int,
                        help="Number of query samples per task")

    parser.add_argument("--outer_batch_size", default=2, type=int,
                        help="Batch of task size")
    
    parser.add_argument("--inner_batch_size", default=12, type=int,
                        help="Training batch size in inner iteration")
    
    parser.add_argument("--outer_update_lr", default=5e-5, type=float,
                        help="Meta learning rate")
    
    parser.add_argument("--inner_update_lr", default=5e-5, type=float,
                        help="Inner update learning rate")
    
    parser.add_argument("--inner_update_step", default=10, type=int,
                        help="Number of interation in the inner loop during train time")

    parser.add_argument("--inner_update_step_eval", default=40, type=int,
                        help="Number of interation in the inner loop during test time")
    
    parser.add_argument("--num_task_train", default=500, type=int,
                        help="Total number of meta tasks for training")
                        
    parser.add_argument("--num_task_test", default=3, type=int,
                        help="Total number of tasks for testing")
             
    parser.add_argument("--out_dir", default='output_file', type=str,
                        help="Path to dataset file")

    args = parser.parse_args()
    
    train_set = json.load(open(args.data_train))
    dev_set = json.load(open(args.data_dev))
   

    high_resource_classes = ['PER']   # The high_resource_classes represent pre-train type mentioned in the thesis
    train_examples = [r for r in train_set if r['domain'] in high_resource_classes]
    dev_examples = [r for r in dev_set if r['domain'] not in high_resource_classes] 
    print('num of train examples:', len(train_examples), 'num of dev examples:', len(dev_examples))
   
    tokenizer = BertTokenizer.from_pretrained(args.bert_model)
    learner = Learner(args)
    
    dev = MetaTask(dev_examples, num_task = args.num_task_test, k_support=args.k_spt, 
                    k_query=args.k_qry, tokenizer = tokenizer)

    global_step = 0
    for epoch in range(args.epoch):

        train = MetaTask(train_examples, num_task = args.num_task_train, k_support=args.k_spt, 
                         k_query=args.k_qry, tokenizer = tokenizer)

        db = create_batch_of_tasks(train, is_shuffle = True, batch_size = args.outer_batch_size)
        print('epoch:', epoch, "\n-----------------------------------\n")
        for step, task_batch in enumerate(db):

            acc, report_train, model_ = learner(task_batch)
            print('epoch:', epoch, 'Step:', step, '\ttraining Acc:', acc)
            print('classification_report:\n', report_train)

            global_step += 1
            model_to_save = model_.module if hasattr(model_, 'module') else model_
            model_to_save.save_pretrained(args.out_dir)
            tokenizer.save_pretrained(args.out_dir)
              
if __name__ == "__main__":
    main()
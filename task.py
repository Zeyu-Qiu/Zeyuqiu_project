import os
import torch
from torch.utils.data import Dataset
import numpy as np
import collections
import random
import json, pickle
from torch.utils.data import TensorDataset

# LABEL_MAP  =  {'LOC':0, 'ORG':1, 'PER':2, 'O': 3, 0:'LOC', 1:'ORG', 2:'PER', 3:'O'}
SEED = 1
random.seed(SEED) 
np.random.seed(SEED) 
torch.manual_seed(SEED) 
torch.backends.cudnn.deterministic = True


class MetaTask(Dataset):
    
    def __init__(self, examples, num_task, k_support, k_query, tokenizer):
        """
        :param samples: list of samples
        :param num_task: number of training tasks.
        :param k_support: number of support sample per task
        :param k_query: number of query sample per task
        """
        self.examples = examples
        random.shuffle(self.examples)
        
        self.num_task = num_task
        self.k_support = k_support
        self.k_query = k_query
        self.tokenizer = tokenizer
        self.max_seq_length = 128  # 256
        self.create_batch(self.num_task)
    
    def create_batch(self, num_task):
        self.supports = []  # support set
        self.queries = []   # query set
        
        for b in range(num_task):  
            
            domain = random.choice(self.examples)['domain']
            domainExamples = [e for e in self.examples if e['domain'] == domain]
            
            selected_examples = random.sample(domainExamples,self.k_support + self.k_query)
            random.shuffle(selected_examples)
            exam_train = selected_examples[:self.k_support]
            exam_test  = selected_examples[self.k_support:]
            
            self.supports.append(exam_train)
            self.queries.append(exam_test)

    def create_feature_set(self,examples):
        pad_token_label_id = -100
        sequence_a_segment_id = 0
        cls_token_segment_id = 1
        pad_token_segment_id = 0
        sep_token = "[SEP]"
        sep_token_extra = False #If model is roberta, then sep_token_extra = True
        cls_token = "[CLS]"
        cls_token_at_end = False
        mask_padding_with_zero = True
        pad_on_left = False
        pad_token = 0

        label_map = {'LOC':0, 'PER':1, 'ORG':2, 'O':3}
      

        all_input_ids      = torch.empty(len(examples), self.max_seq_length, dtype = torch.long)
        all_attention_mask = torch.empty(len(examples), self.max_seq_length, dtype = torch.long)
        all_segment_ids    = torch.empty(len(examples), self.max_seq_length, dtype = torch.long)
        all_label_ids      = torch.empty(len(examples), self.max_seq_length, dtype = torch.long)
        features = []
        for ex_index, example in enumerate(examples):
            tokens = []
            label_ids = []
            word_tokens = example['text']
            label = example['label']
            if len(word_tokens) > 0:
                tokens.extend(word_tokens)
                label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))

            special_tokens_count = self.tokenizer.num_special_tokens_to_add()
            if len(tokens) > self.max_seq_length - special_tokens_count:
                tokens = tokens[: (self.max_seq_length - special_tokens_count)]
                label_ids = label_ids[: (self.max_seq_length - special_tokens_count)]

            tokens += [sep_token]
            label_ids += [pad_token_label_id]
            if sep_token_extra:
                # roberta uses an extra separator b/w pairs of sentences
                tokens += [sep_token]
                label_ids += [pad_token_label_id]
            segment_ids = [sequence_a_segment_id] * len(tokens)

            if cls_token_at_end:
                tokens += [cls_token]
                label_ids += [pad_token_label_id]
                segment_ids += [cls_token_segment_id]
            else:
                tokens = [cls_token] + tokens
                label_ids = [pad_token_label_id] + label_ids
                segment_ids = [cls_token_segment_id] + segment_ids

            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = self.max_seq_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
                segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
                label_ids = ([pad_token_label_id] * padding_length) + label_ids
            else:
                input_ids += [pad_token] * padding_length
                input_mask += [0 if mask_padding_with_zero else 1] * padding_length
                segment_ids += [pad_token_segment_id] * padding_length
                label_ids += [pad_token_label_id] * padding_length

            assert len(input_ids) == self.max_seq_length
            assert len(input_mask) == self.max_seq_length
            assert len(segment_ids) == self.max_seq_length
            assert len(label_ids) == self.max_seq_length

            if "token_type_ids" not in self.tokenizer.model_input_names:
                segment_ids = None

            all_input_ids[ex_index] = torch.Tensor(input_ids).to(torch.long)
            all_attention_mask[ex_index] = torch.Tensor(input_mask).to(torch.long)
            all_segment_ids[ex_index] = torch.Tensor(segment_ids).to(torch.long)
            all_label_ids[ex_index] = torch.Tensor(label_ids).to(torch.long) 

        tensor_set = TensorDataset(all_input_ids, all_attention_mask, all_segment_ids, all_label_ids)

        
        return tensor_set


    
    def __getitem__(self, index):
        support_set = self.create_feature_set(self.supports[index])
        query_set   = self.create_feature_set(self.queries[index])
        return support_set, query_set

    def __len__(self):
        # as we have built up to batchsz of sets, you can sample some small batch size of sets.
        return self.num_task
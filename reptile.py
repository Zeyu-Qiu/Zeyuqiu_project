from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from torch.optim import Adam, AdamW
from torch.nn import CrossEntropyLoss
from transformers import BertForSequenceClassification, BertForTokenClassification
from copy import deepcopy
import gc
from sequence_labelling import classification_report
import torch
import numpy as np

class Learner(nn.Module):
    """
    Meta Learner
    """
    def __init__(self, args):
        """
        :param args:
        """
        super(Learner, self).__init__()
        
        self.num_labels = args.num_labels
        self.outer_batch_size = args.outer_batch_size
        self.inner_batch_size = args.inner_batch_size
        self.outer_update_lr  = args.outer_update_lr
        self.inner_update_lr  = args.inner_update_lr
        self.inner_update_step = args.inner_update_step
        self.inner_update_step_eval = args.inner_update_step_eval
        self.bert_model = args.bert_model
        self.out_dir = args.out_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = BertForTokenClassification.from_pretrained(self.bert_model, num_labels=self.num_labels)
        self.outer_optimizer = AdamW(self.model.parameters(), lr=self.outer_update_lr)
        self.model.train()

    def forward(self, batch_tasks, training = True):
        """
        batch = [(support TensorDataset, query TensorDataset),
                 (support TensorDataset, query TensorDataset),
                 (support TensorDataset, query TensorDataset),
                 (support TensorDataset, query TensorDataset)]
        
        # support = TensorDataset(all_input_ids, all_attention_mask, all_segment_ids, all_label_ids)
        """
        task_accs = []
        sum_gradients = []
        num_task = len(batch_tasks)
        num_inner_update_step = self.inner_update_step if training else self.inner_update_step_eval

        for task_id, task in enumerate(batch_tasks):
            support = task[0]
            query   = task[1]
            
            fast_model = deepcopy(self.model)
            fast_model.to(self.device)
            support_dataloader = DataLoader(support, sampler=RandomSampler(support),
                                            batch_size=self.inner_batch_size)
            
            inner_optimizer = AdamW(fast_model.parameters(), lr=self.inner_update_lr)
            fast_model.train()
            
            print('----Task',task_id, '----')
            for i in range(0,num_inner_update_step):
                all_loss = []
                for inner_step, batch in enumerate(support_dataloader):
                    
                    batch = tuple(t.to(self.device) for t in batch)
                    input_ids, attention_mask, segment_ids, label_id = batch
                    outputs = fast_model(input_ids, attention_mask, segment_ids, labels = label_id)
                    
                    loss = outputs[0]              
                    loss.backward()
                    inner_optimizer.step()
                    inner_optimizer.zero_grad()
                    
                    all_loss.append(loss.item())
                
                if i % 4 == 0:
                    print("Inner Loss: ", np.mean(all_loss))
            
            fast_model.to(torch.device('cpu'))
            
            if training:
                meta_weights = list(self.model.parameters())
                fast_weights = list(fast_model.parameters())

                gradients = []
                for i, (meta_params, fast_params) in enumerate(zip(meta_weights, fast_weights)):
                
                    gradient = meta_params - fast_params
                    if task_id == 0:
                        sum_gradients.append(gradient)
                    else:
                        sum_gradients[i] += gradient

            fast_model.to(self.device)
            fast_model.eval()
            with torch.no_grad():
                query_dataloader = DataLoader(query, sampler=None, batch_size=len(query))
                query_batch = iter(query_dataloader).next()
                query_batch = tuple(t.to(self.device) for t in query_batch)
                q_input_ids, q_attention_mask, q_segment_ids, q_label_id = query_batch
                q_outputs = fast_model(q_input_ids, q_attention_mask, q_segment_ids, labels = q_label_id)

                q_logits = F.softmax(q_outputs[1], dim=1)
                pre_label_id = torch.argmax(q_logits, dim=2)  # shape: [k_query,128]

                b = (q_label_id != -100).clone().detach()
                for i in range(b.shape[1]):
                    if b[0, i] == False:
                        pre_label_id[0, i] = -100

                pre_label_id_rep = pre_label_id[b]
                q_label_id_rep = q_label_id[b]

                pre_label_id_rep = pre_label_id_rep.detach().cpu().numpy().tolist()
                q_label_id_rep = q_label_id_rep.detach().cpu().numpy().tolist()


                b_pred = []
                for li in pre_label_id_rep:
                  if li == 0:
                    b_pred.append('LOC')
                  if li == 1:
                    b_pred.append('PER')
                  if li == 2:
                    b_pred.append('ORG')
                  if li == 3:
                    b_pred.append('O')


                b_true = []
                for li in q_label_id_rep:
                  if li == 0:
                    b_true.append('LOC')
                  if li == 1:
                    b_true.append('PER')
                  if li == 2:
                    b_true.append('ORG')
                  if li == 3:
                    b_true.append('O')


                pre_label_id = pre_label_id.detach().cpu().numpy().tolist()
                q_label_id = q_label_id.detach().cpu().numpy().tolist()
                
                acc = accuracy_score(q_label_id[0], pre_label_id[0])
                task_accs.append(acc)
            
            fast_model.to(torch.device('cpu'))
            del fast_model, inner_optimizer
            torch.cuda.empty_cache()
            

        if training:
            # Average gradient across tasks
            for i in range(0,len(sum_gradients)):
                if sum_gradients[i] is None:
                    print("gradient is None at layer i =" + str(i))
                else:
                    sum_gradients[i] = sum_gradients[i] / float(num_task)

            #Assign gradient for original model, then using optimizer to update its weights
            for i, params in enumerate(self.model.parameters()):
                params.grad = sum_gradients[i]

            self.outer_optimizer.step()
            self.outer_optimizer.zero_grad()
            
            del sum_gradients
            gc.collect()
              
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save_pretrained(self.out_dir)
        tokenizer.save_pretrained(self..out_dir)
        
        return np.mean(task_accs), classification_report(b_true, b_pred), self.model
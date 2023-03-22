from IPython import embed
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import sys
sys.path.append('..')
sys.path.append('.')
import pandas as pd

import argparse
import torch
import toml
from utils import check_dir_exist_or_build, pstore, pload, split_and_padding_neighbor, set_seed, load_collection
from torch.utils.data import DataLoader, Dataset, TensorDataset, IterableDataset
import json
from tqdm import tqdm, trange
import random
from itertools import combinations
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from models import load_model

class ConvExample_rewrite:
    def __init__(self, sample_id, 
                       #query,
                       rewrite):
        self.sample_id = sample_id
        #self.query = query
        self.rewrite = rewrite


class ConvDataset_rewrite(Dataset):
    def __init__(self, args, query_tokenizer, filename):
        self.examples = []

        with open(filename, 'r') as f:
            data = f.readlines()

        with open(args.test_file_path_2, 'r') as f2:
            data_2 = f2.readlines()

        n = len(data)
        n = int(args.use_data_percent * n)

        logger.info("Loading {} data file...".format(filename))
        logger.info("Loading {} data file...".format(args.test_file_path_2))

        for i in trange(n):
            data[i] = json.loads(data[i])
            if 'id' in data[i]:
                sample_id = data[i]['id']
            else:
                sample_id = data[i]['sample_id']
            if 'output' in data[i]:
                rewrite = data[i]['output']
            elif 'rewrite' in data[i]:
                rewrite = data[i]['rewrite']
            else:
                rewrite = data[i]['oracle_utt_text']
            if 'query' in data[i]:
                cur_query = data[i]['query']
            else:
                cur_query = data[i]['cur_utt_text']
            #if "answer" in data[i]:
            #    answer = data[i]["answer"]
            #else:
            #    answer = data[i]["cur_response_text"]
            #rewrite = rewrite + ' ' + answer
            if args.eval_type == "answer":
                data_2[i] = json.loads(data_2[i])
                rewrite = data_2[i]['answer_utt_text']
            elif args.eval_type == "oracle+answer":
                data_2[i] = json.loads(data_2[i])
                rewrite = rewrite + ' ' + data_2[i]['answer_utt_text']
            elif args.eval_type == "oracle+nexq":
                data_2[i] = json.loads(data_2[i])
                rewrite = rewrite + ' ' + data_2[i]['next_q_utt_text']

            rewrite = query_tokenizer.encode(rewrite, add_special_tokens=True)
            #query = query_tokenizer.encode(cur_query, add_special_tokens=True)

            self.examples.append(ConvExample_rewrite(sample_id,
                                        #query,
                                        rewrite,
                                        )) 


    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

    @staticmethod
    def get_collate_fn(args):

        def collate_fn(batch: list):
            collated_dict = {
                "bt_sample_id": [],
                "bt_query":[],
                "bt_query_mask":[],
                "bt_rewrite":[],
                "bt_rewrite_mask":[],
            }
            
            bt_sample_id = [] 
            #bt_query = []
            #bt_query_mask = []
            bt_rewrite = []
            bt_rewrite_mask = []

            for example in batch:
                # padding
                #query, query_mask = pad_seq_ids_with_mask(example.query, max_length = args.max_query_length)
                rewrite, rewrite_mask = pad_seq_ids_with_mask(example.rewrite, max_length = args.max_concat_length)
                bt_sample_id.append(example.sample_id)
                #bt_query.append(query)
                #bt_query_mask.append(query_mask)  
                bt_rewrite.append(rewrite)
                bt_rewrite_mask.append(rewrite_mask)     

            collated_dict["bt_sample_id"] = bt_sample_id
            #collated_dict["bt_query"] = bt_query
            #collated_dict["bt_query_mask"] = bt_query_mask
            collated_dict["bt_rewrite"] = bt_rewrite
            collated_dict["bt_rewrite_mask"] = bt_rewrite_mask
            
            # change to tensor
            for key in collated_dict:
                if key not in ["bt_sample_id"]:
                    collated_dict[key] = torch.tensor(collated_dict[key], dtype=torch.long)
            return collated_dict

        return collate_fn

class T5RewriterIRDataset_qrecc(Dataset):
    def __init__(self, args, tokenizer, filename):
        self.examples = []
        
        with open(filename, encoding="utf-8") as f:
            data = f.readlines()
        n = len(data)
        n = int(args.use_data_percent * n)  
        # randomly sample n samples for deugging
        if n < len(data):
           random.seed(args.seed)
           data = random.sample(data, n)

        i = 0

        for line in tqdm(data):
            record = json.loads(line)
            flat_concat = []
            cur_utt_text = record['cur_utt_text']
            ctx_utts_text = record['ctx_utts_text']
            cur_response_text = record["cur_response_text"]
            oracle_utt_text = record["oracle_utt_text"]
            
            if "pos_docs_text" in record and "random_neg_docs_text" in record:
                pos_docs_text = record["pos_docs_text"]
                random_neg_docs_text = record["random_neg_docs_text"]
            else:
                continue
            
            if args.use_prefix:
                cur_utt_text = "question: " + cur_utt_text
                first_context = True
                
            cur_utt = tokenizer.encode(cur_utt_text, add_special_tokens = True, max_length = args.max_query_length)
            flat_concat.extend(cur_utt)
            for j in range(len(ctx_utts_text) - 1, -1, -1):
                if j % 2 == 1:
                    max_length = args.max_response_length
                else:
                    max_length = args.max_query_length
                if args.use_prefix and first_context:
                    ctx_utts_text[j] = "context: " + ctx_utts_text[j]
                    first_context = False
                utt = tokenizer.encode(ctx_utts_text[j], add_special_tokens=True, max_length=max_length, truncation=True) # not remove [CLS]
                if len(flat_concat) + len(utt) > args.max_concat_length:
                    flat_concat += utt[:args.max_concat_length - len(flat_concat) - 1] + [utt[-1]]    # must ended with [SEP]
                    break
                else:
                    flat_concat.extend(utt) 

            flat_concat, flat_concat_mask = padding_seq_to_same_length(flat_concat, max_pad_length = args.max_concat_length)

            if args.collate_fn_type == "flat_concat_for_train":
                if args.decode_type == "oracle":
                    target_seq = oracle_utt_text
                    target_encoding = tokenizer(target_seq, padding="max_length", max_length=args.max_query_length, truncation=True)    
                elif args.decode_type == "answer":
                    target_seq = cur_response_text
                    target_encoding = tokenizer(target_seq, padding="max_length", max_length=args.max_response_length, truncation=True)    
                elif args.decode_type == "next_q":
                    if (i + 1) != n:
                        next_record = json.loads(data[i + 1])
                        next_turn_id = str(next_record['sample_id'].strip().split('_')[-1])
                        if next_turn_id != '1':
                            next_query_text = next_record['cur_utt_text']
                        else:
                            next_query_text = cur_utt_text
                    else:
                        next_query_text = cur_utt_text
                    target_seq = next_query_text
                    target_encoding = tokenizer(target_seq, padding="max_length", max_length=args.max_query_length, truncation=True)    

                labels = target_encoding.input_ids
                labels = torch.tensor(labels)
                labels[labels == tokenizer.pad_token_id] = -100
                labels = labels.tolist()
                
                for idx in range(len(pos_docs_text)):
                    pos_docs = []
                    neg_docs = []
                    pos_docs.extend(tokenizer.encode(pos_docs_text[idx], add_special_tokens=True, max_length = args.max_doc_length))
                    neg_docs.extend(tokenizer.encode(random_neg_docs_text[0], add_special_tokens=True, max_length = args.max_doc_length))
                    pos_docs, pos_docs_mask = padding_seq_to_same_length(pos_docs, max_pad_length = args.max_doc_length)
                    neg_docs, neg_docs_mask = padding_seq_to_same_length(neg_docs, max_pad_length = args.max_doc_length)
                
                    self.examples.append([record['sample_id'], 
                                    flat_concat,
                                    flat_concat_mask,
                                    labels,
                                    cur_utt_text,
                                    oracle_utt_text,
                                    pos_docs,
                                    pos_docs_mask,
                                    neg_docs,
                                    neg_docs_mask])
                i += 1
            else:
                labels = []
                pos_docs = []
                neg_docs = []
                self.examples.append([record['sample_id'], 
                                        flat_concat,
                                        flat_concat_mask,
                                        labels,
                                        cur_utt_text,
                                        oracle_utt_text,
                                        pos_docs,
                                        pos_docs_mask,
                                        neg_docs,
                                        neg_docs_mask])

    
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]

    @staticmethod
    def get_collate_fn(args):
        
        def collate_fn(batch: list):
            collated_dict = {"bt_sample_ids": [],
                             "bt_input_ids": [],
                             "bt_attention_mask": [],
                             "bt_labels": [],
                             "bt_cur_utt_text": [],
                             "bt_oracle_utt_text": [],
                             "bt_pos_docs": [],
                             "bt_pos_docs_mask": [],
                             "bt_neg_docs": [],
                             "bt_neg_docs_mask": [],
                            }
            for example in batch:
                collated_dict["bt_sample_ids"].append(example[0])
                collated_dict["bt_input_ids"].append(example[1])
                collated_dict["bt_attention_mask"].append(example[2])
                collated_dict["bt_labels"].append(example[3])
                collated_dict["bt_cur_utt_text"].append(example[4])
                collated_dict["bt_oracle_utt_text"].append(example[5])
                collated_dict["bt_pos_docs"].append(example[6])
                collated_dict["bt_pos_docs_mask"].append(example[7])
                collated_dict["bt_neg_docs"].append(example[8])
                collated_dict["bt_neg_docs_mask"].append(example[9])

            not_need_to_tensor_keys = set(["bt_sample_ids", "bt_cur_utt_text", "bt_oracle_utt_text"])
            if args.collate_fn_type == "flat_concat_for_test":
                not_need_to_tensor_keys.add("bt_labels")

            for key in collated_dict:
                if key not in not_need_to_tensor_keys:
                    collated_dict[key] = torch.tensor(collated_dict[key], dtype=torch.long)
                    
            return collated_dict

        return collate_fn

class T5RewriterDataset_qrecc(Dataset):
    def __init__(self, args, tokenizer, filename):
        self.examples = []
        
        with open(filename, encoding="utf-8") as f:
            data = f.readlines()

        #with open(args.addtional_file, encoding="utf-8") as f:
        #    data_2 = f.readlines()

        n = len(data)
        n = int(args.use_data_percent * n)  
        # randomly sample n samples for deugging
        if n < len(data):
           random.seed(args.seed)
           data = random.sample(data, n)

        last_record = None
        i = 0
        for line in tqdm(data):
            record = json.loads(line)
            # <CUR> oracle_query <CTX> cq_{k-1} <SEP> ... cq_{1} (<BOS>)
            flat_concat = []
            cur_utt_text = record['cur_utt_text']
            ctx_utts_text = record['ctx_utts_text']
            cur_response_text = record["cur_response_text"]
            oracle_utt_text = record["oracle_utt_text"]

            #record_2 = json.loads(data_2[i])
            #answer_utt_text_2 = record_2["answer_utt_text"][:-1]

            if args.use_prefix:
                cur_utt_text = "question: " + cur_utt_text
                first_context = True

            '''
            if args.use_last_response:
                turn_id = str(record['sample_id'].strip().split('_')[-1])
                if last_record != None and turn_id != '1':
                    last_response_text = last_record["cur_response_text"]
                else:
                    last_response_text = ""
                ora_utt = tokenizer.encode(oracle_utt_text, add_special_tokens = True, max_length = args.max_query_length)
                flat_concat.extend(ora_utt)
                if len(last_response_text) > 0:
                    last_response = tokenizer.encode(last_response_text, add_special_tokens = True, max_length = args.max_response_length)
                    if len(flat_concat) + len(last_response) > args.max_concat_length:
                        flat_concat += last_response[:args.max_concat_length - len(flat_concat) - 1] + [last_response[-1]] 
                    else:
                        flat_concat.extend(last_response)
            '''
            
            #ans_utt = tokenizer.encode(answer_utt_text_2, add_special_tokens = True, max_length = args.max_query_length)
            #flat_concat.extend(ans_utt)
            #ora_utt = tokenizer.encode(oracle_utt_text, add_special_tokens = True, max_length = args.max_query_length)
            #flat_concat.extend(ora_utt)
            cur_utt = tokenizer.encode(cur_utt_text, add_special_tokens = True, max_length = args.max_query_length)
            flat_concat.extend(cur_utt)
            for j in range(len(ctx_utts_text) - 1, -1, -1):
                if j % 2 == 1:
                    max_length = args.max_response_length
                else:
                    max_length = args.max_query_length
                if args.use_prefix and first_context:
                    ctx_utts_text[j] = "context: " + ctx_utts_text[j]
                    first_context = False
                utt = tokenizer.encode(ctx_utts_text[j], add_special_tokens=True, max_length=max_length, truncation=True) # not remove [CLS]
                if len(flat_concat) + len(utt) > args.max_concat_length:
                    flat_concat += utt[:args.max_concat_length - len(flat_concat) - 1] + [utt[-1]]    # must ended with [SEP]
                    break
                else:
                    flat_concat.extend(utt) 
            
            flat_concat, flat_concat_mask = padding_seq_to_same_length(flat_concat, max_pad_length = args.max_concat_length)

            if args.collate_fn_type == "flat_concat_for_train":
                if args.decode_type == "next_q":
                    if (i + 1) != n:
                        next_record = json.loads(data[i + 1])
                        next_turn_id = str(next_record['sample_id'].strip().split('_')[-1])
                        if next_turn_id != '1':
                            next_query_text = next_record['cur_utt_text']
                        else:
                            next_query_text = cur_utt_text
                    else:
                        next_query_text = cur_utt_text
                    oracle_target_seq = next_query_text
                    i += 1
                #if args.decode_type == "oracle":
                else:
                    oracle_target_seq = oracle_utt_text
                oracle_target_encoding = tokenizer(oracle_target_seq, padding="max_length", max_length=args.max_query_length, truncation=True)    
                #elif args.decode_type == "answer":
                answer_target_seq = cur_response_text
                answer_target_encoding = tokenizer(answer_target_seq, padding="max_length", max_length=args.max_query_length, truncation=True)    
                

                oracle_labels = oracle_target_encoding.input_ids
                oracle_labels_mask = oracle_target_encoding.attention_mask
                oracle_labels = torch.tensor(oracle_labels)
                oracle_labels[oracle_labels == tokenizer.pad_token_id] = -100
                oracle_labels = oracle_labels.tolist()

                answer_labels = answer_target_encoding.input_ids
                answer_labels_mask = answer_target_encoding.attention_mask
                answer_labels = torch.tensor(answer_labels)
                answer_labels[answer_labels == tokenizer.pad_token_id] = -100
                answer_labels = answer_labels.tolist()


                self.examples.append([record['sample_id'], 
                                flat_concat,
                                flat_concat_mask,
                                oracle_labels,
                                oracle_labels_mask,
                                answer_labels,
                                answer_labels_mask,
                                cur_utt_text,
                                oracle_utt_text,
                                ctx_utts_text
                                ])
            else:
                oracle_labels = []
                oracle_labels_mask = []
                answer_labels = []
                answer_labels_mask = []
                self.examples.append([record['sample_id'], 
                                flat_concat,
                                flat_concat_mask,
                                oracle_labels,
                                oracle_labels_mask,
                                answer_labels,
                                answer_labels_mask,
                                cur_utt_text,
                                oracle_utt_text,
                                ctx_utts_text])

            last_record = record
            i += 1

    
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]

    @staticmethod
    def get_collate_fn(args):
        
        def collate_fn(batch: list):
            collated_dict = {"bt_sample_ids": [],
                             "bt_input_ids": [],
                             "bt_attention_mask": [],
                             "bt_oracle_labels": [],
                             "bt_oracle_labels_mask": [],
                             "bt_answer_labels": [],
                             "bt_answer_labels_mask": [],
                             "bt_cur_utt_text": [],
                             "bt_oracle_utt_text": [],
                             "bt_ctx_utts_text":[],
                            }
            for example in batch:
                collated_dict["bt_sample_ids"].append(example[0])
                collated_dict["bt_input_ids"].append(example[1])
                collated_dict["bt_attention_mask"].append(example[2])
                collated_dict["bt_oracle_labels"].append(example[3])
                collated_dict["bt_oracle_labels_mask"].append(example[4])
                collated_dict["bt_answer_labels"].append(example[5])
                collated_dict["bt_answer_labels_mask"].append(example[6])
                collated_dict["bt_cur_utt_text"].append(example[7])
                collated_dict["bt_oracle_utt_text"].append(example[8])
                collated_dict["bt_ctx_utts_text"].append(example[9])

            not_need_to_tensor_keys = set(["bt_sample_ids", "bt_cur_utt_text", "bt_oracle_utt_text", "bt_ctx_utts_text"])
            if args.collate_fn_type == "flat_concat_for_test":
                not_need_to_tensor_keys.add("bt_oracle_labels")
                not_need_to_tensor_keys.add("bt_oracle_labels_mask")
                not_need_to_tensor_keys.add("bt_answer_labels")
                not_need_to_tensor_keys.add("bt_answer_labels_mask")


            for key in collated_dict:
                if key not in not_need_to_tensor_keys:
                    collated_dict[key] = torch.tensor(collated_dict[key], dtype=torch.long)
                    
            return collated_dict

        return collate_fn

class T5RewriterIRDataset_topiocqa(Dataset):
    def __init__(self, args, tokenizer, filename):
        self.examples = []
        
        with open(filename, encoding="utf-8") as f:
            data = f.readlines()
        n = len(data)
        n = int(args.use_data_percent * n)  
        # randomly sample n samples for deugging
        if n < len(data):
           random.seed(args.seed)
           data = random.sample(data, n)

        i = 0

        for line in tqdm(data):
            record = json.loads(line)
            flat_concat = []

            ctx_utts_text = []
            cur_utt_text = record['query']
            history_query = record['history_query']
            history_answer = record['history_answer']
            for i in range(len(history_query)):
                ctx_utts_text.append(history_query[i])
                ctx_utts_text.append(history_answer[i])
            cur_response_text = record["answer"]
            oracle_utt_text = record["rewrite"]

            #if "pos_docs" in record and len(record["pos_docs"]) != 0:
            #    pos_docs_text = record["pos_docs"]
            #    neg_docs_text = record["neg_docs"]
            #else:
            #    continue
            
            if args.use_prefix:
                cur_utt_text = "question: " + cur_utt_text
                first_context = True
                
            cur_utt = tokenizer.encode(cur_utt_text, add_special_tokens = True, max_length = args.max_query_length)
            flat_concat.extend(cur_utt)
            for j in range(len(ctx_utts_text) - 1, -1, -1):
                if j % 2 == 1:
                    max_length = args.max_response_length
                else:
                    max_length = args.max_query_length
                if args.use_prefix and first_context:
                    ctx_utts_text[j] = "context: " + ctx_utts_text[j]
                    first_context = False
                utt = tokenizer.encode(ctx_utts_text[j], add_special_tokens=True, max_length=max_length, truncation=True) # not remove [CLS]
                if len(flat_concat) + len(utt) > args.max_concat_length:
                    flat_concat += utt[:args.max_concat_length - len(flat_concat) - 1] + [utt[-1]]    # must ended with [SEP]
                    break
                else:
                    flat_concat.extend(utt) 

            flat_concat, flat_concat_mask = padding_seq_to_same_length(flat_concat, max_pad_length = args.max_concat_length)

            if args.collate_fn_type == "flat_concat_for_train":
                if args.decode_type == "oracle":
                    target_seq = oracle_utt_text
                    target_encoding = tokenizer(target_seq, padding="max_length", max_length=args.max_query_length, truncation=True)    
                elif args.decode_type == "answer":
                    target_seq = cur_response_text
                    target_encoding = tokenizer(target_seq, padding="max_length", max_length=args.max_response_length, truncation=True)    

                labels = target_encoding.input_ids
                labels = torch.tensor(labels)
                labels[labels == tokenizer.pad_token_id] = -100
                labels = labels.tolist()
                
                pos_docs = []
                neg_docs = []
                pos_docs.extend(tokenizer.encode(record["pos_docs"], add_special_tokens=True, max_length = args.max_doc_length))
                neg_docs.extend(tokenizer.encode(record["neg_docs"], add_special_tokens=True, max_length = args.max_doc_length))
                pos_docs, pos_docs_mask = padding_seq_to_same_length(pos_docs, max_pad_length = args.max_doc_length)
                neg_docs, neg_docs_mask = padding_seq_to_same_length(neg_docs, max_pad_length = args.max_doc_length)
            
                self.examples.append([record['id'], 
                                flat_concat,
                                flat_concat_mask,
                                labels,
                                cur_utt_text,
                                oracle_utt_text,
                                pos_docs,
                                pos_docs_mask,
                                neg_docs,
                                neg_docs_mask])
                i += 1
            else:
                labels = []
                pos_docs = []
                neg_docs = []
                self.examples.append([record['id'], 
                                        flat_concat,
                                        flat_concat_mask,
                                        labels,
                                        cur_utt_text,
                                        oracle_utt_text,
                                        pos_docs,
                                        pos_docs_mask,
                                        neg_docs,
                                        neg_docs_mask])

    
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]

    @staticmethod
    def get_collate_fn(args):
        
        def collate_fn(batch: list):
            collated_dict = {"bt_sample_ids": [],
                             "bt_input_ids": [],
                             "bt_attention_mask": [],
                             "bt_labels": [],
                             "bt_cur_utt_text": [],
                             "bt_oracle_utt_text": [],
                             "bt_pos_docs": [],
                             "bt_pos_docs_mask": [],
                             "bt_neg_docs": [],
                             "bt_neg_docs_mask": [],
                            }
            for example in batch:
                collated_dict["bt_sample_ids"].append(example[0])
                collated_dict["bt_input_ids"].append(example[1])
                collated_dict["bt_attention_mask"].append(example[2])
                collated_dict["bt_labels"].append(example[3])
                collated_dict["bt_cur_utt_text"].append(example[4])
                collated_dict["bt_oracle_utt_text"].append(example[5])
                collated_dict["bt_pos_docs"].append(example[6])
                collated_dict["bt_pos_docs_mask"].append(example[7])
                collated_dict["bt_neg_docs"].append(example[8])
                collated_dict["bt_neg_docs_mask"].append(example[9])

            not_need_to_tensor_keys = set(["bt_sample_ids", "bt_cur_utt_text", "bt_oracle_utt_text"])
            if args.collate_fn_type == "flat_concat_for_test":
                not_need_to_tensor_keys.add("bt_labels")

            for key in collated_dict:
                if key not in not_need_to_tensor_keys:
                    collated_dict[key] = torch.tensor(collated_dict[key], dtype=torch.long)
                    
            return collated_dict

        return collate_fn

class T5RewriterDataset_topiocqa(Dataset):
    def __init__(self, args, tokenizer, filename):
        self.examples = []
        
        with open(filename, encoding="utf-8") as f:
            data = f.readlines()
        n = len(data)
        n = int(args.use_data_percent * n)  
        # randomly sample n samples for deugging
        if n < len(data):
           random.seed(args.seed)
           data = random.sample(data, n)

        for line in tqdm(data):
            record = json.loads(line)
            # <CUR> oracle_query <CTX> cq_{k-1} <SEP> ... cq_{1} (<BOS>)
            flat_concat = []
            ctx_utts_text = []
            cur_utt_text = record['query']
            history_query = record['history_query']
            history_answer = record['history_answer']
            for i in range(len(history_query)):
                ctx_utts_text.append(history_query[i])
                ctx_utts_text.append(history_answer[i])
            cur_response_text = record["answer"]
            oracle_utt_text = record["rewrite"]

            if args.use_prefix:
                cur_utt_text = "question: " + cur_utt_text
                first_context = True

            cur_utt = tokenizer.encode(cur_utt_text, add_special_tokens = True, max_length = args.max_query_length)
            flat_concat.extend(cur_utt)
            for j in range(len(ctx_utts_text) - 1, -1, -1):
                if j % 2 == 1:
                    max_length = args.max_response_length
                else:
                    max_length = args.max_query_length
                if args.use_prefix and first_context:
                    ctx_utts_text[j] = "context: " + ctx_utts_text[j]
                    first_context = False
                utt = tokenizer.encode(ctx_utts_text[j], add_special_tokens=True, max_length= max_length, truncation=True) # not remove [CLS]
                if len(flat_concat) + len(utt) > args.max_concat_length:
                    flat_concat += utt[:args.max_concat_length - len(flat_concat) - 1] + [utt[-1]]    # must ended with [SEP]
                    break
                else:
                    flat_concat.extend(utt) 

            flat_concat, flat_concat_mask = padding_seq_to_same_length(flat_concat, max_pad_length = args.max_concat_length)

            if args.collate_fn_type == "flat_concat_for_train":
                if args.decode_type == "oracle":
                    target_seq = oracle_utt_text
                    target_encoding = tokenizer(target_seq, padding="max_length", max_length=args.max_query_length, truncation=True)    
                elif args.decode_type == "answer":
                    target_seq = cur_response_text
                    target_encoding = tokenizer(target_seq, padding="max_length", max_length=args.max_response_length, truncation=True)    
                labels = target_encoding.input_ids
                labels = torch.tensor(labels)
                labels[labels == tokenizer.pad_token_id] = -100
                labels = labels.tolist()
                '''
                for idx in range(len(pos_docs_text)):
                    pos_docs = []
                    neg_docs = []
                    pos_docs.extend(tokenizer.encode(pos_docs_text[idx], add_special_tokens=True, max_length = args.max_doc_length))
                    neg_docs.extend(tokenizer.encode(random_neg_docs_text[0], add_special_tokens=True, max_length = args.max_doc_length))
                '''
                self.examples.append([record['id'], 
                                flat_concat,
                                flat_concat_mask,
                                labels,
                                cur_utt_text,
                                oracle_utt_text])
            else:
                labels = []
                pos_docs = []
                neg_docs = []
                self.examples.append([record['id'], 
                                        flat_concat,
                                        flat_concat_mask,
                                        labels,
                                        cur_utt_text,
                                        oracle_utt_text])

    
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]

    @staticmethod
    def get_collate_fn(args):
        
        def collate_fn(batch: list):
            collated_dict = {"bt_sample_ids": [],
                             "bt_input_ids": [],
                             "bt_attention_mask": [],
                             "bt_labels": [],
                             "bt_cur_utt_text": [],
                             "bt_oracle_utt_text": []
                            }
            for example in batch:
                collated_dict["bt_sample_ids"].append(example[0])
                collated_dict["bt_input_ids"].append(example[1])
                collated_dict["bt_attention_mask"].append(example[2])
                collated_dict["bt_labels"].append(example[3])
                collated_dict["bt_cur_utt_text"].append(example[4])
                collated_dict["bt_oracle_utt_text"].append(example[5])

            not_need_to_tensor_keys = set(["bt_sample_ids", "bt_cur_utt_text", "bt_oracle_utt_text"])
            if args.collate_fn_type == "flat_concat_for_test":
                not_need_to_tensor_keys.add("bt_labels")

            for key in collated_dict:
                if key not in not_need_to_tensor_keys:
                    collated_dict[key] = torch.tensor(collated_dict[key], dtype=torch.long)
                    
            return collated_dict

        return collate_fn

class T5RewriterDataset_cast(Dataset):
    def __init__(self, args, tokenizer, filename):
        self.examples = []
        
        with open(filename, encoding="utf-8") as f:
            data = f.readlines()
        n = len(data)
        n = int(args.use_data_percent * n)  
        # randomly sample n samples for deugging
        if n < len(data):
           random.seed(args.seed)
           data = random.sample(data, n)

        for line in tqdm(data):
            record = json.loads(line)
            # <CUR> oracle_query <CTX> cq_{k-1} <SEP> ... cq_{1} (<BOS>)
            flat_concat = []
            ctx_utts_text = record["input"][:-1]
            cur_utt_text = record["input"][-1]
            oracle_utt_text = record["target"]

            if args.use_prefix:
                cur_utt_text = "question: " + cur_utt_text
                first_context = True
            
            cur_utt = tokenizer.encode(cur_utt_text, add_special_tokens = True, max_length=args.max_query_length)
            flat_concat.extend(cur_utt)
            for j in range(len(ctx_utts_text) - 1, -1, -1):
                if args.use_prefix and first_context:
                    ctx_utts_text[j] = "context: " + ctx_utts_text[j]
                    first_context = False
                utt = tokenizer.encode(ctx_utts_text[j], add_special_tokens=True, max_length=args.max_query_length, truncation=True) # not remove [CLS]
                if len(flat_concat) + len(utt) > args.max_concat_length:
                    flat_concat += utt[:args.max_concat_length - len(flat_concat) - 1] + [utt[-1]]    # must ended with [SEP]
                    break
                else:
                    flat_concat.extend(utt) # QR and retrieval

            flat_concat, flat_concat_mask = padding_seq_to_same_length(flat_concat, max_pad_length=args.max_concat_length)

            if args.collate_fn_type == "flat_concat_for_train":
                if args.decode_type == "oracle":
                    target_seq = oracle_utt_text
                    target_encoding = tokenizer(target_seq, padding="max_length", max_length=args.max_query_length, truncation=True)    
                elif args.decode_type == "answer":
                    target_seq = cur_response_text
                    target_encoding = tokenizer(target_seq, padding="max_length", max_length=args.max_response_length, truncation=True)    
                labels = target_encoding.input_ids
                labels = torch.tensor(labels)
                labels[labels == tokenizer.pad_token_id] = -100
                labels = labels.tolist()
                
            else:
                labels = []
                
                self.examples.append([record['id'], 
                                        flat_concat,
                                        flat_concat_mask,
                                        labels,
                                        cur_utt_text,
                                        oracle_utt_text])
    
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]

    @staticmethod
    def get_collate_fn(args):
        
        def collate_fn(batch: list):
            collated_dict = {"bt_sample_ids": [],
                             "bt_input_ids": [],
                             "bt_attention_mask": [],
                             "bt_labels": [],
                             "bt_cur_utt_text": [],
                             "bt_oracle_utt_text": []
                            }
            for example in batch:
                collated_dict["bt_sample_ids"].append(example[0])
                collated_dict["bt_input_ids"].append(example[1])
                collated_dict["bt_attention_mask"].append(example[2])
                collated_dict["bt_labels"].append(example[3])
                collated_dict["bt_cur_utt_text"].append(example[4])
                collated_dict["bt_oracle_utt_text"].append(example[5])

            not_need_to_tensor_keys = set(["bt_sample_ids", "bt_cur_utt_text", "bt_oracle_utt_text"])
            if args.collate_fn_type == "flat_concat_for_test":
                not_need_to_tensor_keys.add("bt_labels")

            for key in collated_dict:
                if key not in not_need_to_tensor_keys:
                    collated_dict[key] = torch.tensor(collated_dict[key], dtype=torch.long)
                    
            return collated_dict

        return collate_fn


def padding_seq_to_same_length(input_ids, max_pad_length, pad_token = 0):
    padding_length = max_pad_length - len(input_ids)
    padding_ids = [pad_token] * padding_length
    attention_mask = []

    if padding_length <= 0:
        attention_mask = [1] * max_pad_length
        input_ids = input_ids[:max_pad_length]
    else:
        attention_mask = [1] * len(input_ids) + [0] * padding_length
        input_ids = input_ids + padding_ids
            
    assert len(input_ids) == max_pad_length
    assert len(attention_mask) == max_pad_length
  
    return input_ids, attention_mask

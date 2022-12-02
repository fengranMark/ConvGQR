import sys
import os
import torch
sys.path += ['../']
import pickle
# from model.models import MSMarcoConfigDict, ALL_MODELS
from torch.utils.data import TensorDataset
import numpy as np
import argparse
import json
torch.multiprocessing.set_sharing_strategy('file_system')
from multiprocessing import Process
from models import load_model
import toml
from IPython import embed
from utils import check_dir_exist_or_build

def pad_input_ids(input_ids, max_length, pad_on_left=False, pad_token=0):
    padding_length = max_length - len(input_ids)
    padding_id = [pad_token] * padding_length

    # attention_mask = [1] * len(input_ids) + [0] * padding_length

    if padding_length <= 0:
        input_ids = input_ids[:max_length]
    else:
        if pad_on_left:
            input_ids = padding_id + input_ids
        else:
            input_ids = input_ids + padding_id

    return input_ids



class EmbeddingCache:
    def __init__(self, base_path, seed=-1):
        self.base_path = base_path
        with open(base_path + '_meta', 'r') as f:
            meta = json.load(f)
            self.dtype = np.dtype(meta['type'])
            self.total_number = meta['total_number']
            self.record_size = int(
                meta['embedding_size']) * self.dtype.itemsize + 4
        if seed >= 0:
            self.ix_array = np.random.RandomState(seed).permutation(
                self.total_number)
        else:
            self.ix_array = np.arange(self.total_number)
        self.f = None

    def open(self):
        self.f = open(self.base_path, 'rb')

    def close(self):
        self.f.close()

    def read_single_record(self):
        record_bytes = self.f.read(self.record_size)
        passage_len = int.from_bytes(record_bytes[:4], 'big')
        passage = np.frombuffer(record_bytes[4:], dtype=self.dtype)
        return passage_len, passage

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def __getitem__(self, key):
        if key < 0 or key > self.total_number:
            raise IndexError(
                "Index {} is out of bound for cached embeddings of size {}".
                format(key, self.total_number))
        self.f.seek(key * self.record_size)
        return self.read_single_record()

    def __iter__(self):
        self.f.seek(0)
        for i in range(self.total_number):
            new_ix = self.ix_array[i]
            yield self.__getitem__(new_ix)

    def __len__(self):
        return self.total_number


def numbered_byte_file_generator(base_path, file_no, record_size):
    for i in range(file_no):
        with open('{}_split{}'.format(base_path, i), 'rb') as f:
            while True:
                b = f.read(record_size)
                if not b:
                    # eof
                    break
                yield b

def tokenize_to_file(args, i, num_process, in_path, out_path, line_fn):

    tokenizer, _ = load_model(args.model_type + "_Passage", args.pretrained_passage_encoder)

    with open(in_path, 'r', encoding='utf-8') if in_path[-2:] != "gz" else gzip.open(in_path, 'rt', encoding='utf8') as in_f,\
            open('{}_split{}'.format(out_path, i), 'wb') as out_f:
        first_line = False # tsv with first line
        for idx, line in enumerate(in_f):
            if idx % num_process != i or first_line:
                first_line = False
                continue
            try:
                res = line_fn(args, line, tokenizer)
            except ValueError:
                print("Bad passage.")
            else:
                out_f.write(res)

def multi_file_process(args, num_process, in_path, out_path, line_fn):
    processes = []
    for i in range(num_process):
        p = Process(target=tokenize_to_file,
                    args=(
                        args,
                        i,
                        num_process,
                        in_path,
                        out_path,
                        line_fn,
                    ))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()



def preprocess(args):

    pid2offset = {}
    offset2pid = []
    in_passage_path = args.raw_collection_path

    out_passage_path = os.path.join(
        args.data_output_path,
        "passages",
    )

    if os.path.exists(out_passage_path):
        print("preprocessed data already exist, exit preprocessing")
        return

    out_line_count = 0

    print('start passage file split processing')
    multi_file_process(
        args,
        32,
        in_passage_path,
        out_passage_path,
        PassagePreprocessingFn)

    print('start merging splits')
    with open(out_passage_path, 'wb') as f:
        for idx, record in enumerate(numbered_byte_file_generator(
                out_passage_path, 32, 8 + 4 + args.max_seq_length * 4)):
            p_id = int.from_bytes(record[:8], 'big')
            f.write(record[8:])
            pid2offset[p_id] = idx
            offset2pid.append(p_id)
            if idx < 3:
                print(str(idx) + " " + str(p_id))
            out_line_count += 1

    print("Total lines written: " + str(out_line_count))
    meta = {
        'type': 'int32',
        'total_number': out_line_count,
        'embedding_size': args.max_seq_length}
    with open(out_passage_path + "_meta", 'w') as f:
        json.dump(meta, f)
    embedding_cache = EmbeddingCache(out_passage_path)
    print("First line")
    with embedding_cache as emb:
        print(emb[0])

    pid2offset_path = os.path.join(
        args.data_output_path,
        "pid2offset.pickle",
    )
    offset2pid_path = os.path.join(
        args.data_output_path,
        "offset2pid.pickle",
    )
    with open(pid2offset_path, 'wb') as handle:
        pickle.dump(pid2offset, handle, protocol=4)
    with open(offset2pid_path, "wb") as handle:
        pickle.dump(offset2pid, handle, protocol=4)
    print("done saving pid2offset")


def PassagePreprocessingFn(args, line, tokenizer, title = False):
    line = line.strip()
    ext = args.raw_collection_path[args.raw_collection_path.rfind("."):]
    passage = None
    if ext == ".jsonl":
        obj = json.loads(line)
        p_id = int(obj["id"])
        p_text = obj["text"]
        p_title = obj["title"]

        full_text = p_text[:args.max_doc_character]

        passage = tokenizer.encode(
            p_title, 
            text_pair=full_text,
            add_special_tokens=True,
            truncation=True,
            max_length=args.max_seq_length,
        )
    elif ext == ".tsv":
        try:
            line_arr = line.split('\t')
            p_id = int(line_arr[0])
            if title == True:
                p_text = line_arr[2].rstrip().replace(' [SEP] ', ' ') + ' ' + line_arr[1].rstrip()
            else:
                p_text = line_arr[1].rstrip()
        except IndexError:  # split error
            raise ValueError  # empty passage
        else:
            full_text = p_text[:args.max_doc_character]
            passage = tokenizer.encode(
                full_text,
                add_special_tokens=True,
                truncation=True,
                max_length=args.max_seq_length,
            )
            
    else:
        raise TypeError("Unrecognized file type")

    passage_len = min(len(passage), args.max_seq_length)
    input_id_b = pad_input_ids(passage, args.max_seq_length)

    return p_id.to_bytes(8,'big') + passage_len.to_bytes(4,'big') + np.array(input_id_b,np.int32).tobytes()


def QueryPreprocessingFn(args, line, tokenizer):
    line_arr = line.split('\t')
    q_id = int(line_arr[0])

    passage = tokenizer.encode(
        line_arr[1].rstrip(),
        add_special_tokens=True,
        truncation=True,
        max_length=args.max_query_length)
    passage_len = min(len(passage), args.max_query_length)
    input_id_b = pad_input_ids(passage, args.max_query_length)

    return q_id.to_bytes(8,'big') + passage_len.to_bytes(4,'big') + np.array(input_id_b,np.int32).tobytes()


def GetProcessingFn(args, query=False):
    def fn(vals, i):
        passage_len, passage = vals
        max_len = args.max_query_length if query else args.max_seq_length

        pad_len = max(0, max_len - passage_len)
        token_type_ids = ([0] if query else [1]) * passage_len + [0] * pad_len
        attention_mask = [1] * passage_len + [0] * pad_len

        passage_collection = [(i, passage, attention_mask, token_type_ids)]

        query2id_tensor = torch.tensor(
            [f[0] for f in passage_collection], dtype=torch.long)
        all_input_ids_a = torch.tensor(
            [f[1] for f in passage_collection], dtype=torch.int)
        all_attention_mask_a = torch.tensor(
            [f[2] for f in passage_collection], dtype=torch.bool)
        all_token_type_ids_a = torch.tensor(
            [f[3] for f in passage_collection], dtype=torch.uint8)

        dataset = TensorDataset(
            all_input_ids_a,
            all_attention_mask_a,
            all_token_type_ids_a,
            query2id_tensor)

        return [ts for ts in dataset]

    return fn


def GetTrainingDataProcessingFn(args, query_cache, passage_cache):
    def fn(line, i):
        line_arr = line.split('\t')
        qid = int(line_arr[0])
        pos_pid = int(line_arr[1])
        neg_pids = line_arr[2].split(',')
        neg_pids = [int(neg_pid) for neg_pid in neg_pids]

        all_input_ids_a = []
        all_attention_mask_a = []

        query_data = GetProcessingFn(
            args, query=True)(
            query_cache[qid], qid)[0]
        pos_data = GetProcessingFn(
            args, query=False)(
            passage_cache[pos_pid], pos_pid)[0]

        pos_label = torch.tensor(1, dtype=torch.long)
        neg_label = torch.tensor(0, dtype=torch.long)

        for neg_pid in neg_pids:
            neg_data = GetProcessingFn(
                args, query=False)(
                passage_cache[neg_pid], neg_pid)[0]
            yield (query_data[0], query_data[1], query_data[2], pos_data[0], pos_data[1], pos_data[2], pos_label)
            yield (query_data[0], query_data[1], query_data[2], neg_data[0], neg_data[1], neg_data[2], neg_label)

    return fn


def GetTripletTrainingDataProcessingFn(args, query_cache, passage_cache):
    def fn(line, i):
        line_arr = line.split('\t')
        qid = int(line_arr[0])
        pos_pid = int(line_arr[1])
        neg_pids = line_arr[2].split(',')
        neg_pids = [int(neg_pid) for neg_pid in neg_pids]

        all_input_ids_a = []
        all_attention_mask_a = []

        query_data = GetProcessingFn(
            args, query=True)(
            query_cache[qid], qid)[0]
        pos_data = GetProcessingFn(
            args, query=False)(
            passage_cache[pos_pid], pos_pid)[0]

        for neg_pid in neg_pids:
            neg_data = GetProcessingFn(
                args, query=False)(
                passage_cache[neg_pid], neg_pid)[0]
            yield (query_data[0], query_data[1], query_data[2], pos_data[0], pos_data[1], pos_data[2],
                   neg_data[0], neg_data[1], neg_data[2])

    return fn


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        required=True,
    )
    
    args = parser.parse_args()
    config = toml.load(args.config)
    args = argparse.Namespace(**config)

    # output dir check
    check_dir_exist_or_build([args.data_output_path])

    return args


def main():
    args = get_args()
    preprocess(args)


if __name__ == '__main__':
    main()

# python gen_tokenized_doc.py --config Config/gen_tokenized_doc.toml

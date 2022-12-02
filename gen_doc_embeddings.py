import sys

sys.path += ['../']
import torch
import os
from utils import (
    barrier_array_merge,
    StreamingDataset,
    EmbeddingCache,
)
from torch import nn
import torch.distributed as dist
from tqdm import tqdm
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import numpy as np
import argparse
import re
import gc
import pickle
# from transformers import DPRContextEncoderTokenizer, DPRContextEncoder
from models import load_model
from utils import check_dir_exist_or_build
from IPython import embed
import toml
torch.multiprocessing.set_sharing_strategy('file_system')

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def GetProcessingFn(args, query=False):
    def fn(vals, i):
        passage_len, passage = vals
        max_len = 64 if query else args.max_seq_length

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



def InferenceEmbeddingFromStreamDataLoader(
    args,
    model,
    train_dataloader,
    is_query_inference=True,
):
    # expect dataset from ReconstructTrainingSet
    results = {}
    eval_batch_size = max(1, args.n_gpu) * args.per_gpu_eval_batch_size

    # Inference!
    logger.info("***** Running ANN Embedding Inference *****")
    logger.info("  Batch size = %d", eval_batch_size)

    embedding = []
    embedding2id = []

    if args.local_rank != -1:
        dist.barrier()
    model.eval()

    tmp_n = 0
    expect_per_block_passage_num = 2500000 # 54573064 38636512
    block_size = expect_per_block_passage_num // eval_batch_size # 1000
    block_id = 0
    total_write_passages = 0

    for batch in tqdm(train_dataloader,
                    desc="Inferencing",
                    disable=args.disable_tqdm,
                    position=0,
                    leave=True):

        #if batch[3][-1] <= 19999999:
        #    logger.info("Current {} ".format(batch[3][-1]))
        #    continue

        idxs = batch[3].detach().numpy()  # [#B]

        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {
                "input_ids": batch[0].long(),
                "attention_mask": batch[1].long()
            }
            embs = model(inputs["input_ids"], inputs["attention_mask"])

        embs = embs.detach().cpu().numpy()
    
        # check for multi chunk output for long sequence
        if len(embs.shape) == 3:
            for chunk_no in range(embs.shape[1]):
                embedding2id.append(idxs)
                embedding.append(embs[:, chunk_no, :])
        else:
            embedding2id.append(idxs)
            embedding.append(embs)
        
        tmp_n += 1
        if tmp_n % 500 == 0:
            logger.info("Have processed {} batches...".format(tmp_n))

        if tmp_n % block_size == 0:
            embedding = np.concatenate(embedding, axis=0)
            embedding2id = np.concatenate(embedding2id, axis=0)
            emb_block_path = os.path.join(args.data_output_path, "passage_emb_block_{}.pb".format(block_id))
            with open(emb_block_path, 'wb') as handle:
                pickle.dump(embedding, handle, protocol=4)
            embid_block_path = os.path.join(args.data_output_path, "passage_embid_block_{}.pb".format(block_id))
            with open(embid_block_path, 'wb') as handle:
                pickle.dump(embedding2id, handle, protocol=4)
            total_write_passages += len(embedding)
            block_id += 1

            logger.info("Have written {} passages...".format(total_write_passages))
            embedding = []
            embedding2id = []
            gc.collect()

    if len(embedding) > 0:   
        embedding = np.concatenate(embedding, axis=0)
        embedding2id = np.concatenate(embedding2id, axis=0)

        emb_block_path = os.path.join(args.data_output_path, "passage_emb_block_{}.pb".format(block_id))
        embid_block_path = os.path.join(args.data_output_path, "passage_embid_block_{}.pb".format(block_id))
        with open(emb_block_path, 'wb') as handle:
            pickle.dump(embedding, handle, protocol=4)
        with open(embid_block_path, 'wb') as handle:
            pickle.dump(embedding2id, handle, protocol=4)
        total_write_passages += len(embedding)
        block_id += 1

    logger.info("total write passages {}".format(total_write_passages))
    # return embedding, embedding2id


# streaming inference
def StreamInferenceDoc(args,
                       model,
                       fn,
                       prefix,
                       f,
                       is_query_inference=True,
                       merge=True):
    inference_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    inference_dataset = StreamingDataset(f, fn)
    inference_dataloader = DataLoader(inference_dataset,
                                      batch_size=inference_batch_size)


    if args.local_rank != -1:
        dist.barrier()  # directory created

    InferenceEmbeddingFromStreamDataLoader(
        args,
        model,
        inference_dataloader,
        is_query_inference=is_query_inference,
        )



    logger.info("merging embeddings")


def generate_new_ann(args):

    _, model = load_model(args.model_type + "_Passage", args.pretrained_passage_encoder)
    model = model.to(args.device)
    if args.n_gpu > 1:
        model = nn.DataParallel(model, device_ids = list(range(args.n_gpu)))

    merge = False

    logger.info("***** inference of passages *****")
    passage_collection_path = os.path.join(args.tokenized_passage_collection_dir_path,
                                           "passages")
    passage_cache = EmbeddingCache(passage_collection_path)
    with passage_cache as emb:
        StreamInferenceDoc(
            args,
            model,
            GetProcessingFn(args, query=False),
            "passage_",
            emb,
            is_query_inference=False,
            merge=merge)
    logger.info("***** Done passage inference *****")



def ann_data_gen(args):

    logger.info("start generate ann data")
    generate_new_ann(args)

    if args.local_rank != -1:
        dist.barrier()


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        required=True)

    args = parser.parse_args()
    config = toml.load(args.config)
    args = argparse.Namespace(**config)
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    check_dir_exist_or_build([args.data_output_path])
    
    return args

def main():
    args = get_args()
    ann_data_gen(args)


if __name__ == "__main__":
    main()


# python gen_doc_embeddings.py --config Config/gen_doc_embeddings.toml

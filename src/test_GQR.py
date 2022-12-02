import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import sys
sys.path.append('..')
sys.path.append('.')

import json
import argparse
import toml
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from IPython import embed
import torch
import math
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, BartTokenizer, BartForConditionalGeneration
from utils import set_seed, format_nl_query
from data_structure import T5RewriterDataset_qrecc, T5RewriterDataset_topiocqa, T5RewriterDataset_cast



def inference_t5qr(args):
    if args.model_type == "T5":
        tokenizer = T5Tokenizer.from_pretrained(args.model_checkpoint_path)
        model = T5ForConditionalGeneration.from_pretrained(args.model_checkpoint_path)
        model.to(args.device)
    elif args.model_type == "BART":
        tokenizer = BartTokenizer.from_pretrained(args.model_checkpoint_path)
        model = BartForConditionalGeneration.from_pretrained(args.model_checkpoint_path)
        model.to(args.device)

    if args.n_gpu > 1:
        query_encoder = torch.nn.DataParallel(query_encoder, device_ids = list(range(args.n_gpu)))

    test_dataset = T5RewriterDataset_topiocqa(args, tokenizer, args.test_file_path)
    args.batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    #ddp_sampler = DistributedSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, 
                                  shuffle=False,
                                  batch_size=args.batch_size, 
                                  collate_fn=test_dataset.get_collate_fn(args))
    
    # begin to inference
    with open(args.output_file_path, "w") as f:
        with torch.no_grad():
            model.eval()
            for batch in tqdm(test_dataloader, desc="Step"):
                bt_input_ids = batch["bt_input_ids"].to(args.device)
                bt_attention_mask = batch["bt_attention_mask"].to(args.device)
                if args.n_gpu > 1:
                    output_seqs = model.module.generate(input_ids=bt_input_ids, 
                                                        attention_mask=bt_attention_mask, 
                                                        do_sample=False,
                                                        max_length=args.max_query_length,
                                                        )
                    )
                else:
                    output_seqs = model.generate(input_ids=bt_input_ids, 
                                                        attention_mask=bt_attention_mask, 
                                                        do_sample=False,
                                                        max_length=args.max_query_length,
                                                        )
                    )

                outputs = tokenizer.batch_decode(output_seqs, skip_special_tokens=True)

                for i in range(len(outputs)):
                    record = {}
                    record["sample_id"] = batch["bt_sample_ids"][i]
                    if args.decode_type == "oracle":
                        record["oracle_utt_text"] = outputs[i]
                    elif args.decode_type == "answer":
                        record["answer_utt_text"] = outputs[i]
                    elif args.decode_type == "next_q":
                        record["next_q_utt_text"] = outputs[i]
                    record["cur_utt_text"] = batch["bt_cur_utt_text"][i]
                    record["ctx_utts_text"] = batch["bt_ctx_utts_text"][i]
                    record["original_oracle_utt_text"] = batch["bt_oracle_utt_text"][i]
                    f.write(json.dumps(record) + '\n') 

    logger.info("Inference finsh!")
    

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_checkpoint_path", type=str, default="output/train_qrecc/Checkpoint/KD-ANCE-prefix-oracle-0.5-best-model")
    parser.add_argument("--test_file_path", type=str, default="datasets/qrecc/new_preprocessed/test.json")
    parser.add_argument('--output_file_path', type=str, default="output/qrecc/QR/test_QRIR_oracle_prefix.json")
    parser.add_argument("--collate_fn_type", type=str, default="flat_concat_for_test")
    parser.add_argument("--decode_type", type=str, default="oracle")
    parser.add_argument("--model_type", type=str, default="T5")
    parser.add_argument("--use_last_response", type=bool, default=False)
    parser.add_argument("--use_prefix", type=bool, default=True)


    parser.add_argument("--num_return_sequences", type=int, default=1)
    parser.add_argument("--per_gpu_eval_batch_size", type=int,  default=32)
    parser.add_argument("--use_data_percent", type=float, default=1)
    
    parser.add_argument("--max_query_length", type=int, default=32, help="Max single query length")
    parser.add_argument("--max_doc_length", type=int, default=384, help="Max doc length")
    parser.add_argument("--max_response_length", type=int, default=64, help="Max response length")
    parser.add_argument("--max_concat_length", type=int, default=512, help="Max concatenation length of the session.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_gpu", type=int, default=1)


    args = parser.parse_args()


    # pytorch parallel gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")#, args.local_rank)
    args.device = device


    return args

if __name__ == '__main__':
    args = get_args()
    set_seed(args)

    logger.info("---------------------The arguments are:---------------------")
    logger.info(args)

    inference_t5qr(args)

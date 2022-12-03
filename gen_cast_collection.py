import sys

sys.path += ['../']
import torch
import os
from torch import nn
from tqdm import tqdm
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import numpy as np
from trec_car import read_data
import argparse
import re
import gc
import pickle
from models import load_model
from utils import check_dir_exist_or_build
from IPython import embed
import toml
torch.multiprocessing.set_sharing_strategy('file_system')

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_sim_file(filename):
    """
    Reads the deduplicated documents file and stores the 
    duplicate passage ids into a dictionary
    """

    sim_dict = {}
    lines = open(filename).readlines()
    for line in lines:
        data = line.strip().split(':')
        if len(data[1]) > 0:
            sim_docs = data[-1].split(',')
            for docs in sim_docs:
                sim_dict[docs] = 1

    return sim_dict

def gen_cast_collection(args):

    # Combine TREC-CAR & MS MARCO, remove duplicate passages, assign new ids
    car_id_to_idx = {}
    car_idx_to_id = []

    # INPUT
    car_id_to_idx_file = os.path.join(args.data_output_path,
                                      "car_id_to_idx.pickle")
    car_idx_to_id_file = os.path.join(args.data_output_path,
                                      "car_idx_to_id.pickle")
    out_collection_file = os.path.join(args.data_output_path,
                                       "collection.tsv")

    sim_dict = parse_sim_file(args.duplicated_file_path)
    car_base_id = 10000000
    i = 0
    with open(out_collection_file, "w") as f:
        print("Processing TREC-CAR...")
        for para in tqdm(read_data.iter_paragraphs(open(args.car_passage_path, 'rb')), disable=args.disable_tqdm):
            car_id = "CAR_" + para.para_id
            text = para.get_text()
            text = text.replace("\t", " ").replace("\n",
                                                " ").replace("\r", " ")
            idx = car_base_id + i
            car_id_to_idx[car_id] = idx  # e.g. CAR_76a4a716d4b1b01995c6663ee16e94b4ca35fdd3 -> 10000044
            car_idx_to_id.append(car_id)
            f.write("{}\t{}\n".format(idx, text))
            i += 1
        print("Processing MS MARCO...")
        removed = 0
        with open(args.msmarco_passage_path, "r") as m:
            for line in tqdm(m, disable=args.disable_tqdm):
                marco_id, text = line.strip().split("\t")
                if ("MARCO_" + marco_id) in sim_dict:
                    removed += 1
                    continue
                f.write("{}\t{}\n".format(marco_id, text))
        print("Removed " + str(removed) + " passages")
    
    print("Dumping id mappings...")
    with open(car_id_to_idx_file, "wb") as f:
        pickle.dump(car_id_to_idx, f)
    with open(car_idx_to_id_file, "wb") as f:
        pickle.dump(car_idx_to_id, f)

    print("Generate CAsT collection OK!")


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        required=True)

    args = parser.parse_args()
    config = toml.load(args.config)
    args = argparse.Namespace(**config)
    check_dir_exist_or_build([args.data_output_path])
    
    return args

def main():
    args = get_args()
    gen_cast_collection(args)


if __name__ == "__main__":
    main()

# python gen_cast_collection.py --config=Config/gen_cast_collection.toml
# title = 
# "params for generating doc embeddings. \
# The following things should be provided:
# - tokenized passage corpus
# - a pretrained passage encoder"


# [Model]
model_type = "ANCE"
pretrained_passage_encoder = "checkpoints/ad-hoc-ance-msmarco"   # passage encoder!!!
max_seq_length = 384

# [Gen]
per_gpu_eval_batch_size = 250
local_rank = -1 # Not use distributed training
disable_tqdm = false
n_gpu = 2

# [Input Data]
# tokenized_passage_collection_dir_path = "datasets/topiocqa/tokenized"
tokenized_passage_collection_dir_path = "datasets/cast/tokenized"


# [Output]
# data_output_path = "datasets/topiocqa/embeddings"
data_output_path = "datasets/cast/embeds"

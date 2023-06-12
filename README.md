# ConvGQR
This is the temporary repository of our ACL 2023 accepted paper - ConvGQR: Generative Query Reformulation for Conversational Search.

# Environment Dependency

Main packages:
- python 3.8
- torch 1.8.1
- transformer 4.2.0
- numpy 1.22
- faiss-gpu 1.7.2
- pyserini 0.16

# Runing Steps

## 1. Download data and Preprocessing

Four public datasets can be download from [QReCC](https://github.com/apple/ml-qrecc), [TopiOCQA](https://github.com/McGill-NLP/topiocqa), [CAsT-19 and CAsT-20](https://www.treccast.ai/). Data preprocessing can refer to "preprocess" folder.

## 2. Train ConvGQR

To train ConvGQR, please run the following commands. The pre-trained language models we use for generation and passage encoding is [T5-base](https://huggingface.co/t5-base) and [ANCE](https://github.com/microsoft/ANCE).

    
    python train_GQR.py --pretrained_query_encoder="checkpoints/T5-base" \ 
      --pretrained_passage_encoder="checkpoints/ad-hoc-ance-msmarco" \
      --train_file_path=$train_file_path \ 
      --log_dir_path=$log_dir_path \
      --model_output_path=$model_output_path \ 
      --collate_fn_type="flat_concat_for_train" \ 
      --decode_type=$decode_type \ # "oracle" for rewrite and "answer" for expansion
      --per_gpu_train_batch_size=8 \ 
      --num_train_epochs=15 \
      --max_query_length=32 \
      --max_doc_length=384 \ 
      --max_response_length=32 \
      --max_concat_length=512 \ 
      --alpha=0.5


## 3. Use ConvGQR to produce reformualted queries

Then we can use the trained model to generate rewritten query and expansion by running:

    python test_GQR.py --model_checkpoint_path=$model_checkpoint_path \
      --test_file_path=$test_file_path \
      --output_file_path=$output_file_path \
      --collate_fn_type="flat_concat_for_test" \ 
      --decode_type=$decode_type \ # "oracle" for rewrite and "answer" for expansion
      --per_gpu_eval_batch_size=32 \ 
      --max_query_length=32 \
      --max_doc_length=384 \ 
      --max_response_length=32 \
      --max_concat_length=512 \ 
      

## 4. Retrieval Indexing (Dense and Sparse)

Now, we got both the rewritten query file and knowledge expansion produced by ConvGQR. Before performing retrieval to evaluate the reformulated queries, we should first establish index for both dense and sparse retrievers. 

### 4.1 Dense
For dense retrieval, we use the pre-trained ad-hoc search model ANCE to generate passage embeedings. Two scripts for each dataset are provided by running:

    python gen_tokenized_doc.py --config=Config/gen_tokenized_doc.toml
    python gen_doc_embeddings.py --config=Config/gen_doc_embeddings.toml

### 4.2 Sparse

For sparse retrieval, we first run the format convertion script as:

    python convert_to_pyserini_format.py
    
Then create the index for the collection by running

    bash create_index.sh

## 5. Retrieval evaluation

Now, we can perform retrieval to evaluate the reformulated queries by running:

    # for dense retrieval, dataset_name includes "qrecc", "topiocqa", "cast19" and "cast20"
    python test_{dataset_name}.py --config=Config/test_{dataset_name}.toml
    
    # for spase retrieval, dataset_name includes "qrecc" and "topiocqa"
    python bm25_{dataset_name}.py --config=Config/bm25_{dataset_name}.toml

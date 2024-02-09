# PersonalizedCIR
A code base for personalized conversational query reformulation. (A SIGIR 2023 short paper submission)

# Environment Dependency

Main packages:
- python 3.8
- torch 1.8.1
- transformer 4.2.0
- numpy 1.22
- faiss-gpu 1.7.2
- pyserini 0.16
- openai

# Runing Steps

## 1. Download data and Preprocessing

Four public datasets can be downloaded from TREC Interactive Knowledge Assistance Track (iKAT)(https://www.trecikat.com/).

## 2. Generate reformulation
There are two approaches: one involves the selection of ptkb followed by the generation of a reformulation using the chosen ptkb, while the other entails the concurrent output of both the selected ptkb and the reformulation by the LLM. The prompt templates are provided in **prompt_template.md**.

### 2.1 Select ptkb
Using LLM to select or machine choose(Add a ptkb in sequence and select the ptkb with improved NDCG@3 compared to not adding ptkb.)

    python LLM_select_ptkb_Xshot.py --input_path=$input_path \ 
      --output_path=$output_path \ 
      --shot=$shot \ # 0shot,1shot,3shot,5shot
      
    python add_ptkb_one_by_one.py --input_path=$input_path \ 
      --output_path=$output_path \ 

### 2.2 Generate reformulation using the selected ptkb
Generate a reformulation using the selected ptkb.(Two types of prompt)

    python LLM_select_ptkb_Xshot.py --input_path=$input_path \ 
      --output_path=$output_path \ 
      --annotation=$annotation \ # human choose,machine choose,LLM 0,1,3,5 choose
      --prompt_type=$prompt_type \ # type1,type2
      
### 2.3 Selecte and reformulate
Simultaneously output selected ptkb and reformulation.

    python select&reformulate_Xshot.py --input_path=$input_path \ 
      --output_path=$output_path \ 
      --shot=$shot \ # 0shot,1shot,3shot,5shot

## 3. Retrieval Indexing (Dense and Sparse)

To evaluate the reformulated query, we should first establish index for both dense and sparse retrievers.

### 3.1 Dense
For dense retrieval, we use the pre-trained ad-hoc search model ANCE to generate passage embeedings. Two scripts for each dataset are provided in index folder by running:

    python gen_tokenized_doc.py --config=gen_tokenized_doc.toml
    python gen_doc_embeddings.py --config=gen_doc_embeddings.toml

### 3.2 Sparse

For sparse retrieval, we first run the format conversion script as:

    python convert_to_pyserini_format.py
    
Then create the index for the collection by running

    bash create_index.sh

## 4. Retrieval evaluation

### 4.1 Sparse retrieval
We can perform sparse retrieval to evaluate the personalized reformulated queries by running:

    python bm25_ikat.py
    
### 4.2 Dense retrieval
We can perform dense retrieval to evaluate the personalized reformulated queries by running:

    python test_ikat.py --pretrained_encoder_path=$trained_model_path \ 
      --passage_embeddings_dir_path=$passage_embeddings_dir_path \ 
      --passage_offset2pid_path=$passage_offset2pid_path \
      --qrel_output_path=$qrel_output_path \ % output dir
      --output_trec_file=$output_trec_file \
      --trec_gold_qrel_file_path=$trec_gold_qrel_file_path \ % gold qrel file
      --per_gpu_train_batch_size=4 \ 
      --test_type=rewrite \ 
      --max_query_length=64 \
      --max_doc_length=384 \ 
      --max_response_length=256 \
      --is_train=False \
      --top_k=100 \
      --rel_threshold=1 \ 
      --passage_block_num=$passage_block_num \
      --use_gpu=True

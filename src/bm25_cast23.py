from re import T
from IPython import embed
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import sys
sys.path.append('..')
sys.path.append('.')
import argparse
import os
#from utils import check_dir_exist_or_build
from os import path
from os.path import join as oj
import toml
import numpy as np
import json
from pyserini.search.lucene import LuceneSearcher
import pytrec_eval

def main():
    args = get_args()
    query_list = []
    qid_list = []


    if args.query_type == "combine":
        with open(args.input_query_path, "r") as f:
            data = f.readlines()
        with open(args.input_query_path1, "r") as f:
            data1 = f.readlines()
        for (l,l1) in zip(data,data1):
            l = json.loads(l)
            l1 = json.loads(l1)
            query_id = l["sample_id"]
            query = l["rewrite_utt_text"] + l["response_utt_text"] + l1["response_utt_text"]
            query_list.append(query)
            qid_list.append(query_id)

    else:
        with open(args.input_query_path, "r") as f:
            data = f.readlines()
        for l in data:
            l = json.loads(l)
            query_id = l["sample_id"]
            if args.query_type == "rewrite":
                query = l['rewrite_utt_text']
            if args.query_type == "concat":
                i = 1
                query = ''
                while i <= args.query_number:
                    query += l['rewrite_utt_text']
                    i += 1
                query += l['response_utt_text']
            if args.query_type == "concat_auto":
                i = 1
                rewrite = l['rewrite_utt_text']
                response = l['response_utt_text']
                words1 = rewrite.split()
                words2 = response.split()
                n = round(len(words2)/len(words1)/args.c)
                query = ''
                while i <= n:
                    query += l['rewrite_utt_text']
                    i += 1
                query += l['response_utt_text']
            if args.query_type == "oracle":
                query = l['oracle_utt_text']
            if args.query_type == "fusion":
                query = l['fusion']
            if args.query_type == "response":
                query = l['response_utt_text']

            query_list.append(query)
            qid_list.append(query_id)
   
    # pyserini search
    searcher = LuceneSearcher(args.index_dir_path)
    searcher.set_bm25(args.bm25_k1, args.bm25_b)
    hits = searcher.batch_search(query_list, qid_list, k = args.top_k, threads = 40)
    
    with open(oj(args.output_dir_path, args.output_dir_path1), "w") as f:
        for qid in qid_list:
            for i, item in enumerate(hits[qid]):
                f.write("{} {} {} {} {} {} {}".format(qid,
                                                "Q0",
                                                item.docid,
                                                i + 1,
                                                -i - 1 + 200,
                                                item.score,
                                                "bm25"
                                                ))
                f.write('\n')

def agg_res_with_maxp(run_trec_file):
    res_file = os.path.join(run_trec_file)
    with open(run_trec_file, 'r' ) as f:
        run_data = f.readlines()
    
    agg_run = {}
    for line in run_data:
        line = line.strip().split(" ")
        sample_id = line[0]
        if sample_id not in agg_run:
            agg_run[sample_id] = {}
        doc_id = "_".join(line[2].split('_')[:2])
        try:
            score = float(line[5])
        except:
            breakpoint()
        if doc_id not in agg_run[sample_id]:
            agg_run[sample_id][doc_id] = 0
        agg_run[sample_id][doc_id] = max(agg_run[sample_id][doc_id], score)
    
    agg_run = {k: sorted(v.items(), key=lambda item: item[1], reverse=True) for k, v in agg_run.items()}
    with open(os.path.join(run_trec_file + ".agg"), "w") as f:
        for sample_id in agg_run:
            doc_scores = agg_run[sample_id]
            rank = 1
            for doc_id, real_score in doc_scores:
                rank_score = 2000 - rank
                f.write("{} Q0 {} {} {} {}\n".format(sample_id, doc_id, rank, rank_score, real_score, "ance"))
                rank += 1

def print_res(run_file, qrel_file, rel_threshold):
    with open(run_file, 'r' )as f:
        run_data = f.readlines()
    with open(qrel_file, 'r') as f:
        qrel_data = f.readlines()
    
    qrels = {}
    qrels_ndcg = {}
    runs = {}
    
    for line in qrel_data:
        line = line.strip().split()
        query = line[0].replace('_', '-')
        passage = line[2]
        rel = int(line[3])
        if query not in qrels:
            qrels[query] = {}
        if query not in qrels_ndcg:
            qrels_ndcg[query] = {}

        # for NDCG
        qrels_ndcg[query][passage] = rel
        # for MAP, MRR, Recall
        if rel >= rel_threshold:
            rel = 1
        else:
            rel = 0
        qrels[query][passage] = rel
    
    for line in run_data:
        line = line.split()
        query = line[0]
        passage = line[2]
        rel = int(line[4])
        if query not in runs:
            runs[query] = {}
        runs[query][passage] = rel

    # pytrec_eval eval
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"map", "recip_rank", "recall.5", "recall.10", "recall.20", "recall.1000","P_20"})
    res = evaluator.evaluate(runs)

    map_list = [v['map'] for v in res.values()]
    mrr_list = [v['recip_rank'] for v in res.values()]
    recall_20_list = [v['recall_20'] for v in res.values()]
    recall_1000_list = [v['recall_1000'] for v in res.values()]
    precision_20_list = [v['P_20'] for v in res.values()]

    evaluator = pytrec_eval.RelevanceEvaluator(qrels_ndcg, {"ndcg_cut.3", "ndcg_cut.5", "ndcg_cut.1000"})
    res = evaluator.evaluate(runs)
    ndcg_3_list = [v['ndcg_cut_3'] for v in res.values()]
    ndcg_5_list = [v['ndcg_cut_5'] for v in res.values()]
    ndcg_1000_list = [v['ndcg_cut_1000'] for v in res.values()]


    res = {
            "MRR": round(np.average(mrr_list)*100, 5),
            "NDCG@3": round(np.average(ndcg_3_list)*100, 5),
            "NDCG@5": round(np.average(ndcg_5_list)*100, 5),
            "NDCG@1000": round(np.average(ndcg_1000_list)*100, 5),
            "Precision@20": round(np.average(precision_20_list)*100, 5),
            "Recall@20": round(np.average(recall_20_list)*100, 5),
            "Recall@1000": round(np.average(recall_1000_list)*100, 5),
            "MAP": round(np.average(map_list)*100, 5),
        }

    
    logger.info("---------------------Evaluation results:---------------------")
    logger.info('MRR: %s', mrr_list)
    logger.info('@3: %s',ndcg_3_list)
    logger.info('@5: %s', ndcg_5_list)
    logger.info('@1000: %s', ndcg_1000_list)
    logger.info('P20: %s', precision_20_list)
    logger.info('R20: %s', recall_20_list)
    logger.info('R1000: %s', recall_1000_list)
    logger.info('MAP: %s', map_list)
    logger.info(res)
    return res



def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_query_path", type=str, default="/home/AIChineseMedicine/mofr/topiocqa/code/ConvDR-main/datasets/cast23/QR/test5.4.1_Implicit_ptkb_allresponse_5shot_176.jsonl")
    parser.add_argument("--input_query_path1", type=str,
                        default="/home/AIChineseMedicine/mofr/topiocqa/code/ConvDR-main/datasets/cast23/QR/test6.1_GRG_176.jsonl")
    parser.add_argument('--output_dir_path', type=str, default="/home/AIChineseMedicine/mofr/topiocqa/code/ConvDR-main/output/cast23/bm25")
    parser.add_argument('--output_dir_path1', type=str, default="bm25.trec")
    parser.add_argument('--gold_qrel_file_path', type=str, default="/home/AIChineseMedicine/mofr/topiocqa/code/ConvDR-main/datasets/cast23/qrels.trec")
    parser.add_argument('--index_dir_path', type=str, default="/home/AIChineseMedicine/mofr/topiocqa/code/ConvDR-main/datasets/cast23/bm25_index")
    parser.add_argument("--top_k", type=int,  default=1000)
    parser.add_argument("--rel_threshold", type=int,  default=1)
    parser.add_argument("--bm25_k1", type=int,  default=0.82)
    parser.add_argument("--bm25_b", type=int,  default=0.68)
    parser.add_argument('--query_type', type=str, default="rewrite")
    parser.add_argument('--query_number', type=int, default=1)
    parser.add_argument('--c', type=float, default=1)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()
    args = get_args()
    agg_res_with_maxp(oj(args.output_dir_path, args.output_dir_path1))
    print_res(oj(args.output_dir_path, args.output_dir_path1), args.gold_qrel_file_path, args.rel_threshold)

    #trec_eval(run_trec_file + ".agg", args.gold_qrel_file_path, "/home/kelong_mao/workspace/ConvRetriever/outputs/test/cast21/convdr/", 1)
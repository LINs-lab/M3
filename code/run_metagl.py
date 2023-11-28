import argparse
import pprint
import warnings
import os
import dgl.base
import json
import torch
import random


warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=dgl.base.DGLWarning)

import numpy as np
np.set_printoptions(precision=3, suppress=True)
import bisect
from models.metagl import MetaGL
from utils.utils import setup_cuda, logger, set_seed
from tqdm import tqdm

import matplotlib.pyplot as plt
from utils.utils import get_logger
from transformers import BertTokenizer, BertModel
def get_ori_task_types(merged_list):

    dir = "data"
    file = "testdev_balanced_questions.json"
    with open(os.path.join(dir, file), "r") as file:
        gqa = json.load(file)

    cnt = 0
    for _, v in gqa.items():
        # early exist
        if cnt > 10100:
            break
        cnt += 1
        imageId, question, types = v["imageId"], v["question"], v["types"]
        for item in merged_list:
            if item["imageId"] == imageId and item["question"] == question:
                item["types"] = types

    return merged_list

def time_table( ):

    train_dataset, val_dataset, test_dataset = [], [], []
    instance_file_path = "data/gqa_model_selection_instance_results.json"
    with open(instance_file_path, "r") as file:
        data = json.load(file)

    time_table=[]
    return_time_table=[]
    for i in range(70):
        temp=0
        time_table.append(temp)

    for cnt, (id, item_list) in tqdm(enumerate(data.items())):
            for idx, item in enumerate(item_list):
                vqa, loc, time_real, y = item.values()
                time_table[idx]+=time_real

    return_time_table = [x / 8426 for x in time_table]
    return return_time_table

def build_dataset(args):

    # encoding task types
    structure2idx = {
        "verify": 0,
        "query": 1,
        "logical": 2,
        "choose": 3,
        "compare": 4,
    }

    # get vision and text feature and programming feature
    if args.encoder == "standard":
        X_t = np.load(f"data/embs/t_bert_embs_10k.npy")
        X_v = np.load(f"data/embs/v_vit_embs_10k.npy")
    elif args.encoder == "vilt":
        X = np.load(f"data/embs/vilt_embs_10k.npy")
    elif args.encoder == "blip":
        X = np.load(f"data/embs/blip_embs_10k.npy")

    if args.if_add_program:
        if args.program_encoder == "bert":
            P = np.load(f"data/embs/program_embs/program_bert_embs_10k.npy")
        elif args.program_encoder == "roberta":
            P = np.load(f"data/embs/program_embs/program_roberta_embs_10k.npy")
        elif args.program_encoder == "sbert":
            P = np.load(f"data/embs/program_embs/program_sber_embs_10k.npy")

    # read data
    ori_input_list = []
    prompt_file_path = os.path.join('', "data", "gqa_computation_graph_descrption.json")
    with open(prompt_file_path, "r") as file:
        ori_input_list = json.load(file)
    ori_input_list = get_ori_task_types(ori_input_list)

    instance_file = "data/gqa_model_selection_instance_results.json"
    with open(instance_file, "r") as file:
        data = json.load(file)

    # spilt
    train_random_file="data/train_random_list_{}.json".format(args.split_ratio)
    with open(train_random_file, "r") as file:
        train_random_list = json.load(file)
    val_random_file="data/val_random_list_{}.json".format(args.split_ratio)
    with open(val_random_file, "r") as file:
        val_random_list = json.load(file)

    # init some factors
    sub0_num=0
    sub1_num=0
    sub2_num=0
    sub3_num=0
    sub4_num=0

    train_structure_list,val_structure_list, test_structure_list = [], [],[]
    train_embedding_list,val_embedding_list,test_embedding_list=[],[],[]
    train_performance_list,val_performance_list,test_performance_list=[],[],[]

    for cnt, (id, item_list) in tqdm(enumerate(data.items())):
        meta_data = ori_input_list[int(id)-1]
        structure,program = meta_data["types"]["structural"], meta_data["program"]
        structureId = structure2idx[structure]

        # get features
        if args.encoder == "standard":
            t_embs = torch.tensor(X_t[cnt].tolist(), dtype=torch.float32)
            v_embs = torch.tensor(X_v[cnt].tolist(), dtype=torch.float32)
            t_v_embs=t_embs+v_embs
        elif args.encoder == "vilt" or args.encoder == "blip":
            t_v_embs = torch.tensor(X[cnt].tolist(), dtype=torch.float32)

        if args.if_add_program:
            p_embs = torch.tensor(P[cnt].tolist(), dtype=torch.float32)

        ## only valid path
        flag = False
        for item in item_list:
            y = list(item.values())[-1]
            if y == 1: flag = True

        ## preprocess data
        if flag:
            if cnt in train_random_list:
                train_structure_list.append(structureId)
            elif cnt in val_random_list:
                val_structure_list.append(structureId)
            else:
                test_structure_list.append(structureId)
                if structureId == 0:
                    sub0_num += 1
                elif structureId == 1:
                    sub1_num += 1
                elif structureId == 2:
                    sub2_num += 1
                elif structureId == 3:
                    sub3_num += 1
                elif structureId == 4:
                    sub4_num += 1

            if args.if_add_program:
                embedding = t_v_embs+p_embs
            else:
                embedding = t_v_embs

            random_value=random.random()
            if cnt in train_random_list:
                if random_value <= (1-args.missing_sample_ratio):
                    train_embedding_list.append(embedding)
            elif cnt in val_random_list:
                val_embedding_list.append(embedding)
            else:
                test_embedding_list.append(embedding)

            # collect performance for each sample
            performance=[]
            for idx, item in enumerate(item_list):
                vqa, loc, time, y = item.values()
                performance.append(y)

            if cnt in train_random_list:
                if args.missing_choice_ratio>0.0:
                    for k in range(len(performance)):
                        if random.random()<args.missing_choice_ratio:
                            performance[k]=3
                if random_value <= (1-args.missing_sample_ratio):
                    train_performance_list.append(performance)
            elif cnt in val_random_list:
                val_performance_list.append(performance)
            else:
                test_performance_list.append(performance)

    train_embedding_list = torch.stack(train_embedding_list)
    val_embedding_list = torch.stack(val_embedding_list)
    test_embedding_list = torch.stack(test_embedding_list)
    train_performance_list = torch.tensor(train_performance_list)
    val_performance_list = torch.tensor(val_performance_list)
    test_performance_list = torch.tensor(test_performance_list)
    train_structure_list = torch.tensor(train_structure_list)
    val_structure_list = torch.tensor(val_structure_list)
    test_structure_list = torch.tensor(test_structure_list)

    print("sample amount of each task type:")
    print("sub0_num:",sub0_num,"sub1_num:",sub1_num,"sub2_num:",sub2_num,"sub3_num:",sub3_num,"sub4_num:",sub4_num)

    return train_embedding_list,val_embedding_list,test_embedding_list,train_performance_list,val_performance_list,test_performance_list,val_structure_list,test_structure_list,sub0_num,sub1_num,sub2_num,sub3_num,sub4_num

def main(args,min_positions):

    args.device = torch.device('cuda')

    # prepare dataset
    train_embedding_list,val_embedding_list,test_embedding_list,train_performance_list,val_performance_list,test_performance_list,val_structure_list,test_structure_list,sub0_num,sub1_num,sub2_num,sub3_num,sub4_num=build_dataset(args)
    train_embedding_list = train_embedding_list.numpy()
    val_embedding_list = val_embedding_list.numpy()
    val_performance_list=np.asmatrix(val_performance_list.numpy())
    train_performance_list=np.asmatrix(train_performance_list.numpy())
    test_embedding_list = test_embedding_list.numpy()
    test_performance_list=np.asmatrix(test_performance_list.numpy())

    # print arguments to log
    if args.if_add_program:
        txt_name = 'logs/metagl++_result.txt'
    else:
        txt_name = 'logs/metagl_result.txt'
    # log = open(txt_name, mode = "a+", encoding = "utf-8")
    # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", file=log)
    # print("if_add_program", args.if_add_program, "seed", args.seed, "batch_size", args.batch_size, "epoch", args.epochs,"lr", args.lr, file=log)
    # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", file=log)
    # log.close()

    logger = get_logger(txt_name)
    args_dict = vars(args)

    for key, value in args_dict.items():
        formatted_key = key.ljust(20)
        formatted_value = str(value).ljust(20)
        logger.info('%s: %s', formatted_key, formatted_value)

    # init model
    _, num_models = train_performance_list.shape
    num_meta_feats = train_embedding_list.shape[1]
    metagl = MetaGL(
        num_models=num_models,
        metafeats_dim=num_meta_feats,
        epochs=args.epochs,
        device=args.device,
        batch_size=args.batch_size,
        loss=args.loss
    )

    # start training
    logger.info(f"Running MetaGL...")
    set_seed(args.seed)
    metagl.train_predict(args,min_positions,train_embedding_list,val_embedding_list,test_embedding_list,train_performance_list,val_performance_list,test_performance_list,val_structure_list,test_structure_list,args.lr,0,txt_name,sub0_num,sub1_num,sub2_num,sub3_num,sub4_num)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0,help="which GPU to use. Set to -1 to use CPU.")
    parser.add_argument("--seed", type=int, default=0,help="random seed")
    parser.add_argument("--epochs", type=int, default=20,help="maximum number of training epochs")
    parser.add_argument("--split_ratio", type=float, default=0.6)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--missing_sample_ratio", type=float, default=0.0)
    parser.add_argument("--missing_choice_ratio", type=float, default=0.0)
    parser.add_argument("--if_add_program", type=int, default=0)
    parser.add_argument("--encoder", type=str, default='blip')
    parser.add_argument("--program_encoder", type=str, default="bert")
    parser.add_argument("--loss", type=str, default="cce")
    parser.add_argument("--if_calculate_difficult_level", type=int, default=0)
    parser.add_argument("--limit_time", type=float, default=10000,help='limit path time while testing')

    args = parser.parse_args()
    setup_cuda(args)
    print("\n[Settings]\n" + pprint.pformat(args.__dict__))
    set_seed(args.seed)

    # for limited time
    time_table = time_table()
    sorted_list = sorted(time_table)
    path_mumber = bisect.bisect_left(sorted_list, args.limit_time) + 1
    min_positions = sorted(range(len(time_table)), key=lambda i: time_table[i])[:path_mumber]

    main(args,min_positions)
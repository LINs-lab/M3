import os
import json
import torch
import random
import warnings
import numpy as np
import networkx as nx
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from argparse import ArgumentParser
from collections import defaultdict
from utils.dataset import CustomDataset
from torch.optim import lr_scheduler
from torch_geometric.data import Batch
from torch_geometric.data import DataLoader
from torch_geometric.utils import from_networkx
from utils.utils import get_ori_task_types, pad_graph, get_logger
from models.dag_backbones import GAT, GRU, Transformer

MAX_NODES = 14
NUM_MODEL_ZOO = 70


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # torch.use_deterministic_algorithms(True)


def build_dataset(
    missing_choice_ratio,
    missing_sample_ratio,
    t_embs_file,
    v_embs_file,
    program_file,
    instance_file,
):
    ## 1. load extracted features
    X_t = np.load(f"data/embs/{t_embs_file}")
    X_v = np.load(f"data/embs/{v_embs_file}")

    ## 2. get merged program list
    with open(program_file, "r") as file:
        ori_program_list = json.load(file)

    ori_program_list = get_ori_task_types(ori_program_list)

    ## 3. get instance results
    with open(instance_file, "r") as file:
        instance_results = json.load(file)

    ## 4. dataset & subtest
    train_dataset, val_dataset, test_dataset = [], [], []
    val_structure_list, val_group_list, test_structure_list, test_group_list = (
        [],
        [],
        [],
        [],
    )

    ## 5. load preprocessed computation graph
    with open("data/preprocess/node2idx.json", "r") as file:
        node2idx = json.load(file)

    with open("data/preprocess/processed_graphs.json", "r") as file:
        processed_graphs = json.load(file)

    for key, graph in processed_graphs.items():
        processed_graphs[key] = nx.node_link_graph(graph)

    ## 6. load splitted train / test
    with open(f"data/train_random_list_0.6.json", "r") as file:
        train_random_list = json.load(file)
    with open(f"data/val_random_list_0.6.json", "r") as file:
        val_random_list = json.load(file)

    ## 7. iteration
    logger.info("start data preprocess:")
    for cnt, (id, item_list) in tqdm(enumerate(instance_results.items())):
        # if cnt >= 1000:
        #     continue

        ## 7.1. get meta data
        meta_data = ori_program_list[int(id) - 1]
        structure = meta_data["types"]["structural"]

        ## 7.2. get embeddings
        t_embs = torch.tensor(X_t[cnt].tolist(), dtype=torch.float)
        v_embs = torch.tensor(X_v[cnt].tolist(), dtype=torch.float)

        ## 7.3. only valid path
        flag = False
        sample_executable_choice = 0
        for item in item_list:
            y = list(item.values())[-1]
            sample_executable_choice += y
            if y == 1:
                flag = True

        ## 7.4. preprocess data
        if flag:
            if cnt in val_random_list:
                val_structure_list.append(structure)
                val_group_list.append(sample_executable_choice // 14)
            elif cnt not in train_random_list:
                test_structure_list.append(structure)
                test_group_list.append(sample_executable_choice // 14)

            G_list, t_embs_list, v_embs_list, time_list = [], [], [], []
            for idx, item in enumerate(item_list):
                _, _, time, y = item.values()
                uid = cnt * len(item_list) + idx
                G = processed_graphs[str(uid)]
                G = pad_graph(G, MAX_NODES)
                G = from_networkx(G)

                if cnt in train_random_list and random.random() <= missing_choice_ratio:
                    G.y = torch.tensor(3, dtype=torch.float32)
                else:
                    G.y = torch.tensor(y, dtype=torch.float32)

                G_list.append(G)
                t_embs_list.append(t_embs)
                v_embs_list.append(v_embs)
                time_list.append(time)

            batch_G = Batch.from_data_list(G_list)
            batch_t_embs = torch.stack(t_embs_list)
            batch_v_embs = torch.stack(v_embs_list)
            batch_time = torch.tensor(time_list, dtype=torch.float)
            instance = (batch_G, batch_t_embs, batch_v_embs, batch_time)

            if cnt in train_random_list:
                if random.random() > missing_sample_ratio:
                    train_dataset.append(instance)
            elif cnt in val_random_list:
                val_dataset.append(instance)
            else:
                test_dataset.append(instance)

    logger.info(
        f"train : {len(train_dataset)}, val : {len(val_dataset)}, test: {len(test_dataset)}"
    )
    # print(f"valid sample count: {(len(train_dataset) + len(test_dataset))}")
    train_dataset = CustomDataset(train_dataset)
    val_dataset = CustomDataset(val_dataset)
    test_dataset = CustomDataset(test_dataset)
    return (
        train_dataset,
        val_dataset,
        test_dataset,
        node2idx,
        val_structure_list,
        test_structure_list,
        val_group_list,
        test_group_list,
    )


def evaluate_model(
    bestId, model, data_loader, structure_list, group_list, tag, epoch, verbose
):
    model.eval()
    with torch.no_grad():
        sample_correct, sample_cnt, instance_correct, instance_cnt = 0, 0, 0, 0
        struct2SamCorr, struct2SamCnt = defaultdict(float), defaultdict(float)
        group2SamCorr, group2SamCnt = defaultdict(float), defaultdict(float)
        max_index_list, time_list = [], []
        baseline2score, baseline2time = defaultdict(list), defaultdict(list)
        struct2baseline2score, struct2baseline2time = {}, {}
        group2baseline2score, group2baseline2time = {}, {}
        baselineId2floatscore, baselineId2rank = defaultdict(float), defaultdict(
            float
        )  # higher => better
        baseline2predictions = defaultdict(list)

        for (G, t, v, time), structure, group in zip(
            data_loader, structure_list, group_list
        ):
            group = "G" + str(group)
            sample_cnt += 1
            G, t, v = (
                G.to(torch.device("cuda")),
                t.to(torch.device("cuda")),
                v.to(torch.device("cuda")),
            )

            output = model(G, t, v, None, False).squeeze()
            max_index = torch.argmax(output)

            time = time.squeeze()

            time_list.append(time[max_index])
            sample_correct += G.y[max_index].item()
            max_index_list.append(max_index)

            elements = list(output.cpu().numpy())
            sorted_elements = sorted(elements)
            element_ranks = [sorted_elements.index(x) + 1 for x in elements]

            struct2SamCnt[structure] += 1
            struct2SamCorr[structure] += G.y[max_index].item()
            group2SamCnt[group] += 1
            group2SamCorr[group] += G.y[max_index].item()

            ## baseline
            if structure not in struct2baseline2score.keys():
                struct2baseline2score[structure] = defaultdict(list)
                struct2baseline2time[structure] = defaultdict(list)

            if group not in group2baseline2score.keys():
                group2baseline2score[group] = defaultdict(list)
                group2baseline2time[group] = defaultdict(list)

            for i in range(len(output)):
                baseline2predictions[i].append(float(elements[i]))
                baselineId2rank[i] += element_ranks[i]
                baselineId2floatscore[i] += output[i].item()
                baseline2score[i].append(float(G.y[i].item()))
                baseline2time[i].append(time[i].item())
                struct2baseline2score[structure][i].append(G.y[i].item())
                struct2baseline2time[structure][i].append(time[i].item())
                group2baseline2score[group][i].append(G.y[i].item())
                group2baseline2time[group][i].append(time[i].item())

            y = G.y.cpu().numpy()
            output = output.cpu().numpy()

            binary_predictions = [1.0 if p > 0 else 0.0 for p in output]
            correct_predictions = sum(
                [1 for p, gt in zip(binary_predictions, y) if p == gt]
            )

            instance_correct += correct_predictions
            instance_cnt += len(y)

        execution_success_rate = sample_correct / sample_cnt
        binary_classification_accuracy = instance_correct / instance_cnt
        average_execution_time = np.mean(time_list)
        ms_bestId = max(baselineId2floatscore, key=baselineId2floatscore.get)
        rank_bestId = max(baselineId2rank, key=baselineId2floatscore.get)

        ## best baseline:
        bestTime = np.mean(baseline2time[bestId])
        bestScore = np.mean(baseline2score[bestId])

        ms_bestTime = np.mean(baseline2time[ms_bestId])
        ms_bestScore = np.mean(baseline2score[ms_bestId])

        if verbose:
            for structure in struct2SamCnt.keys():
                score = struct2SamCorr[structure] / struct2SamCnt[structure]
                base_score = np.mean(struct2baseline2score[structure][bestId])
                ms_score = np.mean(struct2baseline2score[structure][ms_bestId])
                rank_score = np.mean(struct2baseline2score[structure][rank_bestId])
                content = f"[{structure}]({len(struct2baseline2score[structure][bestId])}): {base_score:.4f}, {ms_score:.4f}, {rank_score:.4f}, {score:.4f}, {(score-base_score)/base_score*100:.2f}%"
                logger.info(content)

            for group in group2SamCnt.keys():
                score = group2SamCorr[group] / group2SamCnt[group]
                base_score = np.mean(group2baseline2score[group][bestId])
                ms_score = np.mean(group2baseline2score[group][ms_bestId])
                rank_score = np.mean(group2baseline2score[group][rank_bestId])
                content = f"[{group}]({len(group2baseline2score[group][bestId])}): {base_score:.4f}, {ms_score:.4f}, {rank_score:.4f}, {score:.4f}, {(score-base_score)/base_score*100:.2f}%"
                logger.info(content)

        # with open(f"predictions/{tag}_{epoch}_baseline_predictions.json", "w") as file:
        #     json.dump(baseline2predictions, file)

        # with open(f"predictions/{tag}_{epoch}_baseline_labels.json", "w") as file:
        #     json.dump(baseline2score, file)

        return (
            bestScore,
            bestTime,
            execution_success_rate,
            binary_classification_accuracy,
            average_execution_time,
        )


def train_model(
    model,
    val_structure_list,
    test_structure_list,
    val_group_list,
    test_group_list,
    fixed_train_loader,
    train_loader,
    val_loader,
    test_loader,
    args,
):
    ## parse args
    epochs, wd, opt, gamma, lr, verbose = (
        args.epochs,
        args.weight_decay,
        args.optimizer,
        args.gamma,
        args.lr,
        args.verbose,
    )

    if opt == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif opt == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd)
    elif opt == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=gamma)

    ## dataset-level baseline
    baseline2score, baseline2time = defaultdict(list), defaultdict(list)
    for data, _, _, time in fixed_train_loader:
        for i in range(len(time)):
            for j in range(NUM_MODEL_ZOO):
                baseline2score[j].append(data.y[i * NUM_MODEL_ZOO + j])
                baseline2time[j].append(time[i][j])

    bestId, bestScore = 1, 0
    for k, v in baseline2score.items():
        score = np.mean(v)
        if score > bestScore:
            bestId = k
            bestScore = score

    ## training epoch by epoch
    current_max_test_ser, current_max_val_ser, best_epoch = 0.0, 0.0, 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        loop = tqdm(enumerate(train_loader), total=len(train_loader))
        for idx, (G, t, v, _) in loop:
            ## to CUDA
            G, t, v = (
                G.to(torch.device("cuda")),
                t.to(torch.device("cuda")),
                v.to(torch.device("cuda")),
            )

            ## back propogation
            optimizer.zero_grad()
            labels = G.y
            loss = model(G, t, v, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

            loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
            loop.set_postfix(loss=total_loss / (idx + 1))

        (
            val_bestScore,
            val_bestTime,
            val_execution_success_rate,
            val_binary_classification_accuracy,
            val_average_execution_time,
        ) = evaluate_model(
            bestId,
            model,
            val_loader,
            val_structure_list,
            val_group_list,
            "val",
            epoch,
            verbose,
        )
        (
            test_bestScore,
            test_bestTime,
            test_execution_success_rate,
            test_binary_classification_accuracy,
            test_average_execution_time,
        ) = evaluate_model(
            bestId,
            model,
            test_loader,
            test_structure_list,
            test_group_list,
            "test",
            epoch,
            verbose,
        )

        if val_execution_success_rate > current_max_val_ser:
            current_max_val_ser = val_execution_success_rate
            best_epoch = epoch
            best_testing_ser = test_execution_success_rate

        content1 = f"** [{epoch+1}] Val Execution Success Rate: {val_execution_success_rate:.4f} / {val_bestScore:.4f}, Test Execution Success Rate: {test_execution_success_rate:.4f} / {test_bestScore:.4f}"
        content2 = f"** [{epoch+1}] Val Avg. Execution Time: {val_average_execution_time:.4f} / {val_bestTime:.4f}, Test Avg. Execution Time: {test_average_execution_time:.4f} / {test_bestTime:.4f}"
        content3 = f"** [{epoch+1}] Val Instance Accuracy: {val_binary_classification_accuracy:.4f}, Test Instance Accuracy: {test_binary_classification_accuracy:.4f}"
        content4 = "-----------------" * 3

        for content in [content1, content2, content3, content4]:
            logger.info(content)

    logger.info(
        f"best epoch: {best_epoch+1}, best testing SER: {best_testing_ser:.4f}!\n\n"
    )


if __name__ == "__main__":
    ## args
    parser = ArgumentParser()
    parser.add_argument("--log_tag", type=str, default="tmp")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--hidden_size", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=0.6)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--encoder", type=str, default="standard")
    parser.add_argument("--missing_choice_ratio", type=float, default=0.0)
    parser.add_argument("--missing_sample_ratio", type=float, default=0.0)
    parser.add_argument("--backbone", type=str, default="gat")
    parser.add_argument("--loss", type=str, default="cce")
    parser.add_argument("--verbose", type=bool, default=False)
    parser.add_argument("--epochs", type=int, default=10)

    args = parser.parse_args()
    log_tag = args.log_tag
    batch_size, seed, hidden_size, dropout, encoder, backbone, loss = (
        args.batch_size,
        args.seed,
        args.hidden_size,
        args.dropout,
        args.encoder,
        args.backbone,
        args.loss,
    )
    missing_choice_ratio, missing_sample_ratio = (
        args.missing_choice_ratio,
        args.missing_sample_ratio,
    )

    ## config
    result_file = "data/gqa_model_selection_instance_results.json"
    prompt_file = "data/gqa_computation_graph_descrption.json"
    LOG_FILE = f"logs/{log_tag}.txt"
    logger = get_logger(LOG_FILE)
    args_dict = vars(args)

    for key, value in args_dict.items():
        formatted_key = key.ljust(20)
        formatted_value = str(value).ljust(20)
        logger.info("%s: %s", formatted_key, formatted_value)

    if encoder == "standard":
        t_embs_file, v_embs_file = "t_bert_embs_10k.npy", "v_vit_embs_10k.npy"
    elif encoder == "vilt":
        t_embs_file, v_embs_file = "vilt_embs_10k.npy", "vilt_embs_10k.npy"
    elif encoder == "blip":
        t_embs_file, v_embs_file = "blip_embs_10k.npy", "blip_embs_10k.npy"

    ## Create and train the model
    setup_seed(seed)
    (
        train_dataset,
        val_dataset,
        test_dataset,
        node2idx,
        val_structure_list,
        test_structure_list,
        val_group_list,
        test_group_list,
    ) = build_dataset(
        missing_choice_ratio,
        missing_sample_ratio,
        t_embs_file,
        v_embs_file,
        prompt_file,
        result_file,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    fixed_train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    ## logging
    if backbone == "gat":
        model = GAT(
            hidden_dim=hidden_size,
            node_type_num=len(node2idx),
            dropout=dropout,
            model_zoo_num=NUM_MODEL_ZOO,
            max_node_num=MAX_NODES,
            loss=loss,
        ).to(torch.device("cuda"))
    elif backbone == "gru":
        model = GRU(
            hidden_dim=hidden_size,
            node_type_num=len(node2idx),
            dropout=dropout,
            model_zoo_num=NUM_MODEL_ZOO,
            max_node_num=MAX_NODES,
            loss=loss,
        ).to(torch.device("cuda"))
    elif backbone == "transformer":
        model = Transformer(
            hidden_dim=hidden_size,
            node_type_num=len(node2idx),
            dropout=dropout,
            model_zoo_num=NUM_MODEL_ZOO,
            max_node_num=MAX_NODES,
            loss=loss,
        ).to(torch.device("cuda"))

    train_model(
        model,
        val_structure_list,
        test_structure_list,
        val_group_list,
        test_group_list,
        fixed_train_loader,
        train_loader,
        val_loader,
        test_loader,
        args,
    )

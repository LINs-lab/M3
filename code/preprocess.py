import io
import json
import tokenize
import networkx as nx

from tqdm import tqdm
from utils.utils import get_ori_task_types


# parse the encoded str
def parse_step(step_str, partial=False):
    tokens = list(tokenize.generate_tokens(io.StringIO(step_str).readline))
    output_var = tokens[0].string
    step_name = tokens[2].string
    parsed_result = dict(output_var=output_var, step_name=step_name)
    if partial:
        return parsed_result

    arg_tokens = [token for token in tokens[4:-3] if token.string not in [",", "="]]
    num_tokens = len(arg_tokens) // 2
    args = dict()
    for i in range(num_tokens):
        args[arg_tokens[2 * i].string] = arg_tokens[2 * i + 1].string
    parsed_result["args"] = args
    return parsed_result


## create graph from program
def create_graph_from_program(program, vqa, loc, node2idx, type2idx):
    lines = program.strip().split("\n")
    graph = nx.DiGraph()

    node_id = 0
    parsed_dict, modified_parsed_dict = {}, {}
    graph.add_node(node_id, m_idx=1, t_idx=1)
    node_id += 1

    for line in lines:
        parsed_result = parse_step(line)
        parsed_dict[node_id] = parsed_result
        node_type = parsed_result["step_name"]
        if node_type == "RESULT":
            node = "result"
            task_type = "eval"
        if node_type == "EVAL":
            node = "eval"
            task_type = "eval"
        elif node_type == "VQA":
            node = vqa
            task_type = "vqa"
        elif node_type == "LOC":
            node = loc
            task_type = "loc"
        elif "CROP" in node_type:
            node = "crop"
            task_type = "eval"
        elif node_type == "COUNT":
            node = "count"
            task_type = "eval"

        if node not in node2idx:
            node2idx[node] = len(node2idx) + 1
        if task_type not in type2idx:
            type2idx[task_type] = len(type2idx) + 1

        graph.add_node(node_id, m_idx=node2idx[node], t_idx=type2idx[task_type])
        node_id += 1

    ## get the input and output of each node
    for node_id, node_dict in parsed_dict.items():
        node_type = node_dict["step_name"]
        if node_type == "RESULT":
            input = node_dict["args"]["var"]
        if node_type == "LOC":
            input = node_dict["args"]["image"]
        elif node_type == "VQA":
            input = node_dict["args"]["image"]
        elif "CROP" in node_type:
            input = node_dict["args"]["box"]
        elif node_type == "EVAL":
            input = node_dict["args"]["expr"]
        elif node_type == "COUNT":
            input = node_dict["args"]["box"]
        output = node_dict["output_var"]
        modified_parsed_dict[node_id] = {"input": input, "output": output}

    ## create the edges
    for node_id, io in modified_parsed_dict.items():
        input, output = io.values()
        if input == "IMAGE":
            graph.add_edge(0, node_id)
        for pre_node_id in range(1, int(node_id)):
            pre_input, pre_output = modified_parsed_dict[pre_node_id].values()
            if pre_output == input or pre_output in input:
                graph.add_edge(pre_node_id, node_id)

    return graph, node2idx, type2idx


## process step by step
def process(instance_result_file, program_file):
    ## 1. load program file
    with open(program_file, "r") as file:
        ori_program_list = json.load(file)

    ori_program_list = get_ori_task_types(ori_program_list)

    ## 2. load instance result file
    with open(instance_result_file, "r") as file:
        instance_results = json.load(file)

    ## 3. process computation graph
    node2idx, type2idx, processed_graphs = {}, {}, {}

    for cnt, (id, item_list) in tqdm(enumerate(instance_results.items())):
        ## get meta data
        meta_data = ori_program_list[int(id) - 1]
        program = meta_data["program"]

        ## only valid path
        flag = False
        for item in item_list:
            y = list(item.values())[-1]
            if y == 1:
                flag = True

        ## iteration
        if flag:
            for idx, item in enumerate(item_list):
                vqa, loc, time, y = item.values()
                G, node2idx, type2idx = create_graph_from_program(
                    program, vqa, loc, node2idx, type2idx
                )
                processed_graphs[cnt * len(item_list) + idx] = nx.node_link_data(G)

    ## 4. save
    with open("data/preprocess/node2idx.json", "w") as file:
        json.dump(node2idx, file)

    with open("data/preprocess/type2idx.json", "w") as file:
        json.dump(type2idx, file)

    with open("data/preprocess/processed_graphs.json", "w") as file:
        json.dump(processed_graphs, file)


if __name__ == "__main__":
    instance_result_file = "data/gqa_model_selection_instance_results.json"
    program_file = "data/gqa_computation_graph_descrption.json"
    process(instance_result_file, program_file)

import io
import json
import tokenize
import networkx as nx

from tqdm import tqdm


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
def create_graph_from_program(program, llm, vqa, node2idx):
    graph = nx.DiGraph()
    graph.add_node(0, m_idx=1)

    if "LLM(" in program:
        if vqa not in node2idx:
            node2idx[vqa] = len(node2idx) + 1
        if llm not in node2idx:
            node2idx[llm] = len(node2idx) + 1
        graph.add_node(1, m_idx=node2idx[vqa])
        graph.add_node(2, m_idx=node2idx[llm])
        graph.add_edge(0, 1)
        graph.add_edge(1, 2)
    else:
        if vqa not in node2idx:
            node2idx[vqa] = len(node2idx) + 1
        graph.add_node(1, m_idx=node2idx[vqa])
        graph.add_edge(0, 1)

    return graph, node2idx


## process step by step
def process(instance_result_file, program_file):
    ## 1. load program file
    with open(program_file, "r") as file:
        ori_program_list = json.load(file)

    ## 2. load instance result file
    with open(instance_result_file, "r") as file:
        instance_results = json.load(file)

    ## 3. process computation graph
    node2idx, processed_graphs = {}, {}

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
                llm, vqa, _, y = item.values()
                G, node2idx = create_graph_from_program(program, llm, vqa, node2idx)
                processed_graphs[cnt * len(item_list) + idx] = nx.node_link_data(G)

    ## 4. save
    with open("data/preprocess/node2idx.json", "w") as file:
        json.dump(node2idx, file)

    with open("data/preprocess/processed_graphs.json", "w") as file:
        json.dump(processed_graphs, file)


if __name__ == "__main__":
    instance_result_file = "data/okvqa_model_selection_instance_results.json"
    program_file = "data/okvqa_computation_graph_descrption.json"
    process(instance_result_file, program_file)

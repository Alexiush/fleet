from pyvis.network import Network
from fleet import Node, VectorDSU
from tokenizers import Tokenizer

def plot_graph(root: Node, dsu: VectorDSU, tokenizer: Tokenizer, path=None):
    queue = [root]
    mapping = {}
    processed = {root}
    id_counter = 0
    net = Network(directed=True, notebook=True, cdn_resources='remote')

    def rgb_to_hex(r, g, b):
        """Converts RGB integers (0-255) to a hexadecimal color code string."""
        return f'#{r:02x}{g:02x}{b:02x}'

    def bw_to_hex(v):
        value = 255 - int(v * 255)
        return rgb_to_hex(value, value, value)

    def get_node_twin(node):
        nonlocal id_counter

        if node not in mapping:
            net.add_node(id_counter, label=id_counter, value=node.visits, color=bw_to_hex(node.value))
            mapping[node] = id_counter
            id_counter += 1

        return mapping[node]

    while len(queue) > 0:
        head = queue[0]
        queue = queue[1:]

        head_node_id = get_node_twin(head)
        for action, nodes in head.children_visits.items():
            for node_id, visits in nodes.items():
                child_node = head.children[node_id]
                child_node_id = get_node_twin(child_node)

                action_label = tokenizer.decode([action]) if action is not None else "default"
                net.add_edge(head_node_id, child_node_id, weight=visits / head.visits, label=action_label)

                if not child_node in processed:
                    queue.append(dsu.node_store[child_node])
                    processed.add(dsu.node_store[child_node])


    if path is not None:
        net.show(path)
    else:
        temp_path = ""
        net.show(temp_path)
        return net.generate_html()
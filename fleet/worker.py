from typing import List, Optional, Tuple
import torch
from fleet import Node, Trajectory, VectorDSU, PriorTree
import math

class FleetWorker:
    def __init__(
        self, rank: int, dsu: VectorDSU, root: Node, layer: int,
        threshold: Tuple[float, float], logit_dim: int, resample_temperature: float = 3.0, top_k = 32,
        prior_tree: Optional[PriorTree] = None,
        verbose: bool = False, use_reward_penalty: bool = True, return_trajectory: bool = True
    ):
        self.rank = rank

        self.dsu = dsu
        self.prior_tree = prior_tree
        self.root = root
        self.layer = layer

        self.logit_dim = logit_dim

        self.threshold = threshold
        self.resample_temperature = resample_temperature
        self.top_k = top_k

        self.offset = 0
        self.tokens = []
        self.entropies = []
        self.varentropies = []

        self.queue = []
        self.hit_threshold = False

        self.nodes = [self.root]
        self.activation_norm_cache = None
        self.proxy_token = None

        self.verbose = verbose
        self.use_reward_penalty = use_reward_penalty
        self.return_trajectory = return_trajectory

        if self.return_trajectory:
            self.trajectory = Trajectory()

    def update_prompt(self, prompt_tokens: List[int]):
        self.offset = len(prompt_tokens)
        if self.return_trajectory:
            self.trajectory.offset = len(prompt_tokens)

        for token in prompt_tokens:
            self.tokens.append(token)
            if self.return_trajectory:
                self.trajectory.tokens.append(token)

    def update_tokens(self, token: int):
        if self.hit_threshold:
            self.queue.append((self.activation_norm_cache, token))

        self.tokens.append(token)

        if self.return_trajectory:
            self.trajectory.tokens.append(token)

    def update_logits(self, activation: torch.Tensor, logits: torch.Tensor) -> bool:
        """Updates the logit statistics and tells whether threshold was hit"""
        self.activation_norm_cache = (activation / activation.norm(p=2))

        probs = torch.nn.functional.softmax(logits, dim=-1)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        entropy = (-torch.sum(probs * log_probs, dim=-1))
        scaled_entropy = entropy / math.log(self.logit_dim)
        self.entropies.extend(scaled_entropy.cpu().tolist())

        varentropy = torch.sum(probs * (log_probs + entropy.unsqueeze(-1))**2, dim=-1)
        scaled_varentropy = 4 * varentropy / math.log(self.logit_dim)**2
        self.varentropies.extend(scaled_varentropy.cpu().tolist())

        self.hit_threshold = (scaled_entropy > self.threshold[0] and scaled_varentropy > self.threshold[1]).item()
        return self.hit_threshold

    def apply_penalty(self, node: Node, logits: torch.Tensor) -> Tuple[torch.Tensor, List[int]]:
        penalties = torch.zeros_like(logits)
        magnitude = torch.max(logits) + 1e-6

        probs = torch.nn.functional.softmax(logits / self.resample_temperature, dim=-1)

        if self.verbose:
            print("Setting penalty")
            print(f"Children: {len(node.children)}, Actions: {node.actions}, Visits: {node.children_visits}")

        if (len(node.children)) == 0:
            return penalties, []

        _, indices = torch.topk(probs, self.top_k)

        indices = list(set(indices.tolist() + list(node.actions.keys())))
        top_probs = probs[indices].tolist()

        unexplored_top_probs = 0
        probs_mapping = {}
        for p, i in zip(top_probs, indices):
            probs_mapping[i] = p

            if i in node.actions:
                continue

            unexplored_top_probs += p

        node_children = [self.dsu.node_store[c] for c in node.children]

        search_bias = self.prior_tree.query(self.activation_norm_cache) if self.prior_tree else {}

        action_pucbs = {}
        for action, children in node.actions.items():
            pucbs = [
                node_children[c].upper_confidence_bound(
                    node.children_visits[action][c], use_reward_penalty=self.use_reward_penalty
                ) / (node.children_visits[action][c] + 1) for c in children # prior discounting
            ]
            all_visits = sum(node.children_visits[action].values())

            bias_score = search_bias.get(action, None)
            action_pucbs[action] = (
                sum(pucbs) / all_visits
                * probs_mapping[action]
                * math.pow(max(bias_score.mean_reward, 0.15) if bias_score else 0.5, 1.0 / node.visits)
            )

            if self.verbose:
                print(f"Action: {action}, PUCBS: {pucbs}, Visits {all_visits}")

        exploration_pucb = unexplored_top_probs * math.sqrt(node.visits) / (1 + node.visits)
        penalty_ids = [a for a in node.actions.keys()]

        action_pucbs['exploration'] = exploration_pucb
        actions_sorted = sorted(action_pucbs, key=action_pucbs.get, reverse=True)
        best_action = actions_sorted[self.rank]

        exploration = best_action == 'exploration'
        if not exploration:
            penalty_ids.remove(best_action)

        w = [magnitude for i in range(len(penalty_ids))]

        if self.verbose:
            print(f"Probs: {probs_mapping}")
            print(f"PUCB: {action_pucbs}, Exploration: {exploration_pucb}, Penalized tokens: {penalty_ids}")
            print(f"Magnitude: {w}")
            if len(penalty_ids) > 0:
                print(
                    logits[torch.tensor(penalty_ids)].tolist(),
                    logits[torch.tensor(penalty_ids)].cpu() - torch.tensor(w)
                )
            print(f"Exploration: {exploration}")

        if len(penalty_ids) == 0:
            return penalties, []

        penalties[torch.tensor(penalty_ids)] = torch.tensor(w, dtype=penalties.dtype).to(penalties.device)
        return penalties, penalty_ids

    def process_logits(self, logits: torch.Tensor) -> Tuple[torch.Tensor, List[int]]:
        node = self.dsu[self.activation_norm_cache]
        if self.return_trajectory:
            self.trajectory.states[len(self.trajectory.tokens) - 1] = self.activation_norm_cache.tolist()

        if node is not None:
            penalties, p_ids = self.apply_penalty(self.dsu.node_store[node], logits)
        else:
            penalties, p_ids = torch.zeros_like(logits), []

        return penalties, p_ids

    def register_node(self, node_ref: int, activation: torch.Tensor, token: int):
        if node_ref is None:
            node_ref = self.dsu.create_node()

        self.dsu[activation] = node_ref

        node = self.dsu.node_store[node_ref]
        self.nodes[-1].add_child(node_ref, self.proxy_token)
        self.nodes.append(node)

        self.proxy_token = token

    def finish_iteration(self, reward: float) -> Optional[Trajectory]:
        token = self.tokens[-1]
        self.queue.append((self.activation_norm_cache, token))

        for activation_norm, token in self.queue:
            node = self.dsu[activation_norm]
            self.register_node(node, activation_norm, token)

        self.entropies = self.entropies[-10000:]
        self.varentropies = self.varentropies[-10000:]

        Node.backpropagate(self.nodes, reward)

        self.offset = 0
        self.tokens = []
        self.queue = []
        self.hit_threshold = False

        self.nodes = [self.root]
        self.activation_norm_cache = None
        self.proxy_token = None

        if self.return_trajectory:
            trajectory = self.trajectory
            trajectory.reward = reward
            return trajectory
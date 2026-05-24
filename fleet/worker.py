from typing import List, Optional, Tuple
import torch
from fleet import Node, Trajectory, VectorDSU, PriorTree
import math
import random

class FleetWorker:
    def __init__(
        self, rank: int, dsu: VectorDSU, root: Node, layer: int,
        threshold: Tuple[float, float], logit_dim: int, resample_temperature: float = 3.0, top_k = 32,
        prior_tree: Optional[PriorTree] = None, exploration_weight = 1.0, strategy = "naive",
        verbose: bool = False, use_reward_penalty: bool = True, return_trajectory: bool = True
    ):
        """
        A worker class to run at each node

        :param rank: rank of the worker, used to ensure different workers explore different trajectories
        :param dsu: dsu of the nodes in the fleet graph
        :param root: root of the dsu, the starting node
        :param layer: model's layer which activations are analyzed
        :param threshold: tuple of entropy and varentropy values that signify model uncertainty
        :param logit_dim: model's logit dim to scale the entropy and varentropy values
        :param resample_temperature: temperature used to infer predictor component of PUCB
        :param top_k: estimate of viable actions, used to calculate exploration and for action-space based parameter
            scaling
        :param prior_tree: optional prior tree with bias inferred from previous trajectories
        :param exploration_weight: exploration weight coefficient to be used during PUCB calculation, if strategy is
            set to 'random' acts as a scaling parameter
        :param strategy: 'naive' or 'random':
            * 'naive' is well-suited for small values of n and forces the worker to
              always pick the n-th by modulo best action
            * 'random' scales better for higher amount of workers by randomizing their exploration factor
        :param verbose: bool, if set to `True` logs the method decisions
        :param use_reward_penalty: bool, if set to `True` additionally penalizes the nodes that appear to hit the dead end
        :param return_trajectory: bool, if set to `True` saves and returns the trajectory
        """
        self.rank = rank
        self.strategy = strategy

        self.dsu = dsu
        self.prior_tree = prior_tree
        self.root = root
        self.layer = layer

        self.logit_dim = logit_dim

        self.threshold = threshold
        self.resample_temperature = resample_temperature
        self.top_k = top_k

        if self.strategy == 'random':
            scale = math.log(self.top_k)
            random.seed(rank)
            self.exploration_weight = random.random() * scale * exploration_weight
        else:
            self.exploration_weight = exploration_weight

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
        """
        Updates the data about input tokens

        :param prompt_tokens: list of tokens passed as the input to the model
        """
        self.offset = len(prompt_tokens)
        if self.return_trajectory:
            self.trajectory.offset = len(prompt_tokens)

        for token in prompt_tokens:
            self.tokens.append(token)
            if self.return_trajectory:
                self.trajectory.tokens.append(token)

    def update_tokens(self, token: int):
        """
        Updates the data about generated tokens

        :param token: freshly generated token
        """
        if self.hit_threshold:
            self.queue.append((self.activation_norm_cache, token))

        self.tokens.append(token)

        if self.return_trajectory:
            self.trajectory.tokens.append(token)

    def update_logits(self, activation: torch.Tensor, logits: torch.Tensor) -> bool:
        """
        Updates the logit statistics and tells whether threshold was hit

        :param activation: hidden state vector used to differentiate between search states
        :param logits: logit lens of the corresponding vector used to update entropy values
        :return: bool that tells whether the threshold was hit
        """
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
        """
        Uses search data to infer the

        :param node: node representing current search state
        :param logits: logits as returned by the model
        :return: penalty vector to be subtracted from logits and list of ids of penalized actions
        """

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
                    node.children_visits[action][c],
                    use_reward_penalty=self.use_reward_penalty,
                    exploration_weight=self.exploration_weight
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

        if self.strategy == 'naive':
            best_action = actions_sorted[self.rank % len(actions_sorted)]
        else:
            best_action = actions_sorted[0]

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
        """
        Receives the logits from the model and returns the proposed penalty

        :param logits: logits as returned by the model
        :return: penalty vector to be subtracted from logits and list of ids of penalized actions
        """

        node = self.dsu[self.activation_norm_cache]['nodes']
        if self.return_trajectory:
            self.trajectory.states[len(self.trajectory.tokens) - 1] = self.activation_norm_cache.tolist()

        if node is not None:
            penalties, p_ids = self.apply_penalty(node, logits)
        else:
            penalties, p_ids = torch.zeros_like(logits), []

        return penalties, p_ids

    def register_connection(self, node_ref: Optional[int], activation: torch.Tensor, token: int):
        """
        Registers a new connection in the dsu

        :param node_ref: reference to the child node, can be None for previously unvisited states
        :param activation: activation vector to be used by dsu
        :param token: action that transitioned the state towards the registered one
        """

        node = self.dsu.node_store[node_ref]
        self.nodes[-1].add_child(node_ref, self.proxy_token)
        self.nodes.append(node)

        self.proxy_token = token

    def finish_iteration(self, reward: float) -> Optional[Trajectory]:
        """
        Uses the reward to finalize the search iteration

        :param reward: reward for the current iteration
        :return: returns iteration trajectory if requested
        """

        token = self.tokens[-1]
        self.queue.append((self.activation_norm_cache, token))

        for activation_norm, token in self.queue:
            node = self.dsu[activation_norm]['index']
            self.register_connection(node, activation_norm, token)

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
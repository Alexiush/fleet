import math
from collections import deque
from pydantic import BaseModel, Field
from typing import List, Dict, Set

from fleet import VectorDSU


class Node(BaseModel):
    children: List[int] = Field(default_factory=list)
    actions: Dict[int, Set[int]] = Field(default_factory=dict)
    children_visits: Dict[int, Dict[int, int]] = Field(default_factory=dict)
    value: float = 0
    rewards: List[float] = Field(default_factory=list)
    visits: int = 1

    def __repr__(self) -> str:
        return (
            f"<Node value={self.value}, visits={self.visits},"
        )

    def add_child(self, node: int, action: int):
        if node not in self.children:
            self.children.append(node)

        if action not in self.actions:
            action = action if action is not None else -1
            self.actions[action] = set()
            self.children_visits[action] = {}

        node_id = self.children.index(node)
        self.actions[action].add(node_id)

        if node_id not in self.children_visits[action]:
            self.children_visits[action][node_id] = 0

        self.children_visits[action][node_id] += 1

    @property
    def is_terminal(self):
        return not self.children

    def upper_confidence_bound(self, state_visits: int, use_reward_penalty=True, exploration_weight=1.0):
        """Return the UCT score"""
        if self.visits == 0:
            return self.value
        # Encourages exploitation of high-value trajectories
        average_reward = self.value / self.visits
        # Encourages exploration of less-visited trajectories
        exploration_term = math.sqrt(self.visits) / (1 + state_visits)

        # Fallacy penalty:
        # * if this state is relatively well explored
        # * and there is a clear boundary to its score
        # * as well as its predecessor does not have such boundary
        # we can pretty confidently attribute the fallacy to this particular state
        # As we use UCBs only when deciding which state to pick we can omit the parent requirement
        # because the fitting parent was penalized too

        # We will penalize the scores relatively, so very bad states will have reward of a zero
        if use_reward_penalty and len(self.rewards) > 0:
            max_reward = max(self.rewards)
            n_max = self.rewards.count(max_reward)

            p_fallacy = (1.0 - max_reward) * (n_max / len(self.rewards)) * max(0.0, 1 - math.exp(-(n_max - 1) / 10))
        else:
            p_fallacy = 0.0

        fallacy_scale = 1.0 - p_fallacy

        return (average_reward + exploration_weight * exploration_term) * fallacy_scale

    @staticmethod
    def backpropagate(nodes: List['Node'], reward: float):
        """Update the score along the trajectory"""
        for node in nodes:
            node.visits += 1
            node.rewards.append(reward)
            node.value = (node.value * (node.visits - 1) + reward) / node.visits

    def _get_all_children(self, dsu: VectorDSU):
        all_nodes = []
        nodes = deque()
        nodes.append(self)
        while nodes:
            node = nodes.popleft()
            children = [dsu.node_store[c] for c in node.children]

            all_nodes.extend(children)
            for n in children:
                nodes.append(n)
        return all_nodes
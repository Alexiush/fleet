from pydantic import BaseModel, Field
from typing import List, Dict
from fleet.vector_dsu import VectorDSU
import itertools

class Trajectory(BaseModel):
    offset: int = 0 # generation prompt tokens
    tokens: List[int] = Field(default_factory=list)
    states: Dict[int, list[float]] = Field(default_factory=dict) # dict: token index + activation vector
    reward: float = 0.0

    def as_conversation_modeling_trajectory(self):
        """Returns trajectory with prompt labels masked"""
        return {
          'input_ids': self.tokens,
          'attention_mask': [1] * len(self.tokens),
          'labels': [t if i >= self.offset else -100 for i, t in enumerate(self.tokens)]
        }

    def as_language_modeling_trajectory(self):
        """Returns trajectory with all tokens masked except for ones that fleet intervened upon"""
        return {
          'input_ids': self.tokens,
          'attention_mask': [1] * len(self.tokens),
          'labels': [t if i >= self.offset and i-1 in self.states else -100 for i, t in enumerate(self.tokens)]
        }

    def as_distribution_matching_trajectory(self, dsu, use_reward_penalty=False):
        """Returns custom data format with `input_ids` and ucb reward distribution instead of labels"""
        states_sorted = sorted(self.states, key=self.states.get)

        def get_pucbs(node_id: int) -> Dict[int, float]:
            node = dsu[node_id]

            action_pucbs = {}
            for action, children in node.actions.items():
                pucbs = [
                    node.children[c].upper_confidence_bound(
                        node.children_visits[action][c], use_reward_penalty=use_reward_penalty
                    ) / (node.children_visits[action][c] + 1) for c in children
                ]
                all_visits = sum(node.children_visits[action].values())

                action_pucbs[action] = sum(pucbs) / all_visits

            return action_pucbs

        return {
            'input_ids': self.tokens,
            'attention_mask': [1] * len(self.tokens),
            'distribution': [
                get_pucbs(i - 1) if i >= self.offset and i - 1 in self.states else -100 for i, t in enumerate(self.tokens)
            ]
        }

class ResidualCollection(BaseModel):
    dsu: VectorDSU
    trajectories: list[Trajectory] = Field(default_factory=list)

    def as_conversation_modeling_data(self):
        return [t.as_conversation_modeling_trajectory() for t in self.trajectories]

    def as_language_modeling_data(self):
        return [t.as_language_modeling_trajectory() for t in self.trajectories]

    def as_distribution_matching_data(self):
        return list(itertools.chain(*[t.as_distribution_matching_trajectory(self.dsu) for t in self.trajectories]))
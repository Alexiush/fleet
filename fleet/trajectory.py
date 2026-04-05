from pydantic import BaseModel, Field
from typing import List, Dict
from fleet import VectorDSU
import itertools

class Trajectory(BaseModel):
    offset: int = 0 # generation prompt tokens
    tokens: List[int] = Field(default_factory=list)
    states: Dict[int, list[float]] = Field(default_factory=dict) # dict: token index + activation vector
    reward: float = 0.0

    def as_conversation_modeling_trajectory(self):
        return {
          'input_ids': self.tokens,
          'attention_mask': [1] * len(self.tokens),
          'labels': [t if i >= self.offset else -100 for i, t in enumerate(self.tokens)]
        }

    def as_language_modeling_trajectory(self):
        return {
          'input_ids': self.tokens,
          'attention_mask': [1] * len(self.tokens),
          'labels': [t if i >= self.offset and i-1 in self.states else -100 for i, t in enumerate(self.tokens)]
        }

    def as_distribution_matching_trajectory(self, dsu, use_reward_penalty=False):
        states_sorted = sorted(self.states, key=self.states.get)

        examples = []
        for state in states_sorted:
            node = dsu[self.states[state]]

            action_pucbs = {}
            for action, children in node.actions.items():
                pucbs = [
                    node.children[c].upper_confidence_bound(
                        node.children_visits[action][c], use_reward_penalty=use_reward_penalty
                    ) / (node.children_visits[action][c] + 1) for c in children
                ]
                all_visits = sum(node.children_visits[action].values())

                action_pucbs[action] = sum(pucbs) / all_visits

            example = {
              'input_ids': self.tokens[:state],
              'distribution': action_pucbs
            }
            examples.append(example)

        return examples

class ResidualCollection(BaseModel):
    dsu: VectorDSU
    trajectories: list[Trajectory] = Field(default_factory=list)

    def as_conversation_modeling_data(self):
        return [t.as_conversation_modeling_trajectory() for t in self.trajectories]

    def as_language_modeling_data(self):
        return [t.as_language_modeling_trajectory() for t in self.trajectories]

    def as_distribution_matching_data(self):
        return list(itertools.chain(*[t.as_distribution_matching_trajectory(self.dsu) for t in self.trajectories]))
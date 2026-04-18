from pydantic import BaseModel, Field, ConfigDict, field_serializer, field_validator
from typing import List, Dict, Any, Union
import torch
import copy
from fleet import Node

class VectorDSU(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    threshold: float = 0.9
    metric: str = 'dot'

    # set_id -> the centroid-nearest string
    canonical_vec: List[torch.Tensor] = Field(default_factory=list)
    # set_id -> your associated data
    data_store: List[Any] = Field(default_factory=list)
    # node_id -> ref to node behind that id
    node_store: Dict[int, Node] = Field(default_factory=dict)

    def create_node(self):
        new_id = max(self.node_store.keys(), default=-1) + 1

        node = Node()
        self.node_store[new_id] = node
        return new_id

    @field_serializer('canonical_vec')
    def serialize_canonical_vec(self, vecs: List[torch.Tensor]):
        return [t.tolist() for t in vecs]

    @field_validator('canonical_vec', mode='before')
    @classmethod
    def deserialize_canonical_vec(cls, v):
        if not v: return v
        if isinstance(v, list) and len(v) > 0 and not isinstance(v[0], torch.Tensor):
            return [torch.tensor(t) for t in v]
        return v

    def add_tag(self, tag_vecs: Union[torch.Tensor|List[torch.Tensor]]) -> List[torch.Tensor]:
        """Adds tag, unions if similar, and updates the cluster representative."""
        if not isinstance(tag_vecs, list):
            tag_vecs = [tag_vecs]

        reprs = []

        for t in tag_vecs:
            if len(self.canonical_vec) > 0:
                t = t.to(self.canonical_vec[0].device)

            # Compare against current canonical representatives
            reps = self.canonical_vec

            if len(reps) == 0:
                sims = torch.zeros([1], device=t.device)
            elif self.metric == 'dot':
                sims = torch.stack(reps) @ t.unsqueeze(dim=0).T
            else:
                raise ValueError("Unknown metric")

            best_sim, best_sim_id = torch.max(sims), torch.argmax(sims)

            if best_sim > self.threshold:
                target_root = best_sim_id.item()
                canonical_vec = self.canonical_vec[target_root]
                node_ref = self.data_store[target_root]
                visits = self.node_store[node_ref].visits
                self.canonical_vec[target_root] = (canonical_vec * visits + t) / (visits + 1)
                reprs.append(target_root)
            else:
                # New cluster
                target_root = len(self.canonical_vec)
                self.canonical_vec.append(t)
                self.data_store.append(self.create_node())
                reprs.append(target_root)

        return reprs

    def __setitem__(self, tag: torch.Tensor, value: Any):
        root = self.add_tag(tag)[0]
        self.data_store[root] = value

    def __getitem__(self, tag: torch.Tensor) -> Any:
        root = self.add_tag(tag)[0]  # Auto-discover or create
        return self.data_store[root]

    def vector_count(self) -> int:
        return len(self.canonical_vec)

    def list_sets(self):
        """Returns list of (Representative, Data, All Members)"""
        return [
            {
                "representative": self.canonical_vec[root],
                "data": self.data_store[root],
            }
            for root in range(len(self.canonical_vec))
        ]

    def to(self, device) -> 'VectorDSU':
        """Creates an independent replica of the DSU, moving all tensors to the target device."""
        replica = VectorDSU(
            threshold=self.threshold,
            metric=self.metric,
        )

        replica.canonical_vec = [v.to(device) for v in self.canonical_vec]
        replica.data_store = copy.deepcopy(self.data_store)
        replica.node_store = copy.deepcopy(self.node_store)

        return replica
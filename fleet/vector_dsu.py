from pydantic import BaseModel, Field, ConfigDict, field_serializer, field_validator
from typing import List, Dict, Any, Union
import torch
import copy
from fleet import Node

class VectorDSU(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    threshold: float = 0.9

    canonical_vec: List[torch.Tensor] = Field(default_factory=list)
    data_store: Dict[str, List[Any]] = Field(default_factory=lambda: {
      'nodes': [],
    })

    @property
    def node_store(self):
        """The getter: accessed via circle.radius"""
        return self.data_store['nodes']

    def create_node(self):
        new_id = len(self.node_store)

        node = Node()
        for key, values_list in self.data_store.items():
            values_list.append(None)
        self.data_store['nodes'][new_id] = node

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

    def add_tag(self, tag_vecs: Union[torch.Tensor | List[torch.Tensor]]) -> List[torch.Tensor]:
        """Adds tag, unions if similar, and updates the cluster representative."""
        if not isinstance(tag_vecs, list):
            tag_vecs = [tag_vecs]

        reprs = []

        for t in tag_vecs:
            if len(self.canonical_vec) > 0:
                t = t.to(self.canonical_vec[0].device)

            reps = self.canonical_vec
            if len(reps) == 0:
                sims = torch.zeros([1], device=t.device)
            else:
                sims = torch.stack(reps) @ t.unsqueeze(dim=0).T

            best_sim, best_sim_id = torch.max(sims), torch.argmax(sims)

            if best_sim > self.threshold:
                target_root = best_sim_id.item()
                canonical_vec = self.canonical_vec[target_root]
                node_ref = target_root
                visits = self.node_store[node_ref].visits
                new_centroid = (canonical_vec * visits + t) / (visits + 1)
                new_centroid = new_centroid / new_centroid.norm(p=2)
                self.canonical_vec[target_root] = new_centroid
                reprs.append(target_root)
            else:
                target_root = len(self.canonical_vec)
                self.canonical_vec.append(t)
                self.create_node()
                reprs.append(target_root)

        return reprs

    def __setitem__(self, keys: tuple[torch.Tensor, str], value: Any):
        tag, key = keys
        root = self.add_tag(tag)[0]
        self.data_store[key][root] = value

    def __getitem__(self, query: int | torch.Tensor | tuple[torch.Tensor, str] | tuple[int, str]) -> Any:
        need_resolve_root = False
        need_specific_key = False

        if isinstance(query, tuple):
            if isinstance(query[0], torch.Tensor):
                tag, key = query
                need_resolve_root = True
            else:
                root, key = query

            need_specific_key = True
        elif isinstance(query, torch.Tensor):
            tag = query
            need_resolve_root = True
        else:
            root = query

        if need_resolve_root:
            root = self.add_tag(tag)[0]

        values = {k: v[root] for k, v in self.data_store.items()}
        values['index'] = root

        if need_specific_key:
            return values[key]
        else:
            return values

    @property
    def vector_count(self) -> int:
        return len(self.canonical_vec)

    def list_sets(self):
        """Returns list of (Representative, Data, All Members)"""
        return [
            {
                "representative": self.canonical_vec[root],
                "data": self[root],
            }
            for root in range(len(self.canonical_vec))
        ]

    def to(self, device) -> 'VectorDSU':
        """Creates an independent replica of the DSU, moving all tensors to the target device."""
        replica = VectorDSU(
            threshold=self.threshold
        )

        replica.canonical_vec = [v.to(device) for v in self.canonical_vec]
        replica.data_store = copy.deepcopy(self.data_store)

        return replica
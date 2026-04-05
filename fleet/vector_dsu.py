from pydantic import BaseModel, Field, ConfigDict, field_serializer, field_validator
from typing import List, Dict, Any, Union
import torch
import copy

from fleet import Node

class VectorDSU(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    threshold: float = 0.9
    duplicate_threshold: float = 0.98
    max_cluster_size: int = 10
    metric: str = 'dot'

    # set_id -> list of tags in this set
    cluster_members: List[List[torch.Tensor]] = Field(default_factory=list)
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

    @field_serializer('cluster_members')
    def serialize_cluster_members(self, clusters: List[List[torch.Tensor]]):
        return [[t.tolist() for t in cluster] for cluster in clusters]

    @field_serializer('canonical_vec')
    def serialize_canonical_vec(self, vecs: List[torch.Tensor]):
        return [t.tolist() for t in vecs]

    @field_validator('cluster_members', mode='before')
    @classmethod
    def deserialize_cluster_members(cls, v):
        if not v:
            return v

        if isinstance(v, list) and len(v) > 0 and isinstance(v[0], list):
            if len(v[0]) > 0 and not isinstance(v[0][0], torch.Tensor):
                return [[torch.tensor(t) for t in cluster] for cluster in v]
        return v

    @field_validator('canonical_vec', mode='before')
    @classmethod
    def deserialize_canonical_vec(cls, v):
        if not v: return v
        if isinstance(v, list) and len(v) > 0 and not isinstance(v[0], torch.Tensor):
            return [torch.tensor(t) for t in v]
        return v

    def quantize_cluster(self, cluster: List[torch.Tensor], iterations=3) -> List[torch.Tensor]:
        X = torch.stack(cluster)
        num_artificials = self.max_cluster_size // 2

        indices = torch.randperm(X.size(0), device=X.device)[:num_artificials]
        centroids = X[indices]

        for _ in range(iterations):
            sims = X @ centroids.T
            assignments = torch.argmax(sims, dim=1)

            new_centroids = []
            for k in range(num_artificials):
                cluster_k = X[assignments == k]
                if len(cluster_k) > 0:
                    mean_vec = cluster_k.mean(dim=0)
                    new_centroids.append(mean_vec / mean_vec.norm())
                else:
                    new_centroids.append(centroids[k])

            centroids = torch.stack(new_centroids)

        return [centroids[i] for i in range(num_artificials)]

    def extend_cluster(self, cluster: List[torch.Tensor], vector: torch.Tensor) -> List[torch.Tensor]:
        if self.metric == 'dot':
            member_sims = torch.stack(cluster) @ vector.unsqueeze(dim=0).T
        else:
            raise ValueError("Unknown metric")

        member_sims = member_sims.squeeze()
        if member_sims.dim() == 0:
            member_sims = member_sims.unsqueeze(0)

        max_member_sim, max_member_idx = torch.max(member_sims, dim=0)

        if max_member_sim > self.duplicate_threshold:
            return cluster

        cluster.append(vector)

        if len(cluster) > self.max_cluster_size:
            cluster = self.quantize_cluster(cluster)

        return cluster

    def add_tag(self, tag_vecs: Union[torch.Tensor|List[torch.Tensor]]) -> List[torch.Tensor]:
        """Adds tag, unions if similar, and updates the centroid representative."""
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
                self.cluster_members[target_root] = self.extend_cluster(self.cluster_members[target_root], t)
                self._update_representative(target_root)
                reprs.append(target_root)
            else:
                # New cluster
                target_root = len(self.canonical_vec)
                self.cluster_members.append([t])
                self.canonical_vec.append(t)
                self.data_store.append(None)
                reprs.append(target_root)

        return reprs

    def _update_representative(self, root_id: int):
        """Finds the tag in the cluster closest to the mathematical mean."""
        members = self.cluster_members[root_id]
        if len(members) <= 2:
            return  # Not enough data to shift the centroid meaningfully

        member_vecs = members
        centroid = torch.stack(member_vecs).mean(dim=0)

        # Find member with the highest similarity to the centroid
        if self.metric == 'dot':
            sims = centroid.unsqueeze(dim=0) @ torch.stack(member_vecs).T
        else:
            raise ValueError("Unknown metric")

        best_idx = torch.argmax(sims)
        self.canonical_vec[root_id] = members[best_idx]

    def __setitem__(self, tag: torch.Tensor, value: Any):
        root = self.add_tag(tag)[0]
        self.data_store[root] = value

    def __getitem__(self, tag: torch.Tensor) -> Any:
        root = self.add_tag(tag)[0]  # Auto-discover or create
        return self.data_store[root]

    def vector_count(self) -> int:
        return sum([len(members) for members in self.cluster_members])

    def list_sets(self):
        """Returns list of (Representative, Data, All Members)"""
        return [
            {
                "representative": self.canonical_vec[root],
                "data": self.data_store[root],
                "members": self.cluster_members[root]
            }
            for root in range(len(self.cluster_members))
        ]

    def to(self, device) -> 'VectorDSU':
        """Creates an independent replica of the DSU, moving all tensors to the target device."""
        replica = VectorDSU(
            threshold=self.threshold,
            duplicate_threshold=self.duplicate_threshold,
            max_cluster_size=self.max_cluster_size,
            metric=self.metric,
        )

        replica.canonical_vec = [v.to(device) for v in self.canonical_vec]
        replica.cluster_members = [[v.to(device) for v in cluster] for cluster in self.cluster_members]
        replica.data_store = copy.deepcopy(self.data_store)
        replica.node_store = copy.deepcopy(self.node_store)

        return replica
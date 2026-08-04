from pydantic import BaseModel, Field, ConfigDict, field_serializer, field_validator
from typing import List, Dict, Optional
from sklearn.cluster import AgglomerativeClustering
import torch
import numpy as np
from fleet.vector_dsu import VectorDSU
import copy

class ActionPrior(BaseModel):
    visits: int
    mean_reward: float
    variance: float

class PriorTree(BaseModel):
  model_config = ConfigDict(arbitrary_types_allowed=True)

  centroids: torch.Tensor
  radia: List[float]
  statistics: List[Dict[int, ActionPrior]] = Field(default_factory=list)

  @field_serializer('centroids')
  def serialize_cluster_members(self, centroids: torch.Tensor):
      return centroids.tolist()

  @field_validator('centroids', mode='before')
  @classmethod
  def deserialize_cluster_members(cls, centroids):
      return torch.Tensor(centroids)

  @property
  def tree_size(self):
    return self.centroids.shape[0]

  @property
  def tree_height(self):
    return len(set(self.radia))

  def query(self, vector: torch.Tensor) -> Dict[int, ActionPrior]:
    """Vectorized query for priors on various prior tree levels with vector as a key"""
    if self.centroids.numel() == 0:
      return {}

    distances = 1.0 - (vector @ self.centroids.T)
    hits = torch.where(distances.cpu() < torch.Tensor(self.radia))[0]

    # tree is sorted, by going from start to end we can overwrite the data
    current_priors = {}
    for index in hits:
      for action, stats in self.statistics[index].items():
        current_priors[action] = stats

    return current_priors

  def to(self, device) -> 'PriorTree':
    """Creates an independent replica of the PriorTree, moving all tensors to the target device."""
    replica = PriorTree(
      centroids = self.centroids.to(device),
      radia=copy.deepcopy(self.radia),
      statistics=copy.deepcopy(self.statistics)
    )

    return replica

def merge_metadata(metadata, indices):
  """
  Merges multiple metadata entries into a single one

  :param metadata: list of state metadata dictionaries
  :param indices: metadata entries to merge
  :return: merged metadata dictionary
  """
  aggregated_rewards = {}
  for i in indices:
      for action, rewards in metadata[i].items():
        if action not in aggregated_rewards:
          aggregated_rewards[action] = []
        aggregated_rewards[action].extend(rewards)

  return aggregated_rewards


class AgglomerativePriorTreeBuilder:
  def __init__(self, initial_radius=0.15, base_variance_threshold=0.08, min_visits=10):
    """
    Hierarchically clusters the hidden states and extracts priors from the resulting superclusters.

    :param initial_radius: initial raddius of prior tree hierarchical clusters
    :param base_variance_threshold: acceptable variance threshold for the lowest level of a tree
    :param min_visits: minimum number of metadata entries needed for action to be considered a useful prior
    """
    self.initial_radius = initial_radius
    self.base_variance_threshold = base_variance_threshold
    self.min_visits = min_visits

  def build_from_dsus(self, dsus: List[VectorDSU]) -> PriorTree:
    """Extracts metadata from the list of dsus and turns it into a prior tree"""
    if len(dsus) == 0:
      return PriorTree(
        centroids=torch.empty(0),
        statistics=[],
        radia=[]
      )

    vectors = []
    metadata = []

    for dsu in dsus:
      for root_id, vec in enumerate(dsu.canonical_vec):
        node = dsu.node_store[root_id]
        action_rewards = {}

        for action, children in node.actions.items():
          rewards = []
          for child_id in children:
            child_node = dsu.node_store[child_id]
            rewards.extend(child_node.rewards)
          if rewards:
            action_rewards[action] = rewards

        if action_rewards:
          vectors.append(vec)
          metadata.append(action_rewards)

    return self.build_level(vectors, metadata, 1)

  def metadata_to_priors(self, metadata, level):
    """Extracts useful priors from dsu metadata"""
    statistics = {}

    variance_tolerance = self.base_variance_threshold

    total_visits = sum(len(r) for r in metadata.values())
    min_visits = max(self.min_visits , int(total_visits * 0.05))

    for action, rewards in metadata.items():
      variance = float(np.var(rewards))
      if variance <= variance_tolerance and len(rewards) >= min_visits:
        statistics[action] = ActionPrior(
          visits=len(rewards),
          mean_reward=float(np.mean(rewards)),
          variance=variance
        )

    return statistics

  def build_level(
      self,
      centroids, metadata, level,
      old_centroids=None, old_metadata=None, old_levels=None
    ) -> Optional[PriorTree]:
    """Iteratively clusters the data till it becomes a single cluster"""

    if old_centroids is None:
      old_centroids = []
    if old_metadata is None:
      old_metadata = []
    if old_levels is None:
      old_levels = []

    radius = self.initial_radius * np.power(1.4, level)

    vectors = torch.stack(centroids, dim=0)
    X_tensor = torch.clamp(1.0 - (vectors @ vectors.T), min=0.0, max=2.0)
    X = X_tensor.cpu().numpy()
    np.fill_diagonal(X, 0.0)

    clustering = AgglomerativeClustering(
      metric='precomputed',
      linkage='complete',
      distance_threshold=radius,
      n_clusters=None
    ).fit(X)

    new_node_centroids = []
    new_node_metadata = []

    for i in range(clustering.n_clusters_):
      node_ids = clustering.labels_ == i

      cluster_centroid = vectors[node_ids].mean(dim=0)
      cluster_centroid = cluster_centroid / cluster_centroid.norm(p=2)
      new_node_centroids.append(cluster_centroid)

      cluster_metadata = merge_metadata(metadata, np.where(node_ids)[0])
      new_node_metadata.append(cluster_metadata)

    if clustering.n_clusters_ > 1:
      return self.build_level(
        new_node_centroids,
        new_node_metadata,
        level + 1,
        new_node_centroids + old_centroids,
        new_node_metadata + old_metadata,
        ([level] * len(new_node_centroids)) + old_levels
      )
    else :
      all_centroids = new_node_centroids + old_centroids
      all_metadata = new_node_metadata + old_metadata
      all_levels = ([level] * len(new_node_centroids)) + old_levels

      all_statistics = [self.metadata_to_priors(metadata, level) for metadata, level in zip(all_metadata, all_levels)]

      relevant_centroids = []
      relevant_statistics = []
      relevant_levels = []

      for c, s, l in zip(all_centroids, all_statistics, all_levels):
        if len(s.keys()) == 0:
          continue

        relevant_centroids.append(c)
        relevant_statistics.append(s)
        relevant_levels.append(l)

      if len(relevant_centroids) == 0:
        return None

      return PriorTree(
        centroids=torch.stack(relevant_centroids, dim=0),
        statistics=relevant_statistics,
        radia=[self.initial_radius * np.power(1.4, level) for level in relevant_levels]
      )

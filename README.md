# Fleet 🚀

Fleet (or Fast Logit Entropy Enhanced Trajectories) is a method
for Best-of-N scaling of LLMs that provides:
* More sample-effective solution generation than sampling by temperature
* With interpretable hyperparameters
* Token-level decision attribution (and useful trajectories for finetuning)
* Highly composable
* Can run in distributed fashion 

> [!NOTE]
> It also requires white-box access, but only to hidden states (so can easily run on vLLM!)

Let your personal fleet of local models fight your problems!  

### Installation

via pip:
`pip install fleet`

### How it works?

Fleet is based on a heuristic that logits with high entropy and high varentropy 
mean that model is uncertain (higher entropy → more uncertainty).

Fleet treats such states as branching points and uses online clustering to
keep a mapping between clusters of similar branching points and their metadata
(which tokens were selected, what state was next and what was the result). 
It allows to evaluate these states in MCTS fashion and select best action to take.
To be compatible with other sampling techniques fleet does not select the best 
token, but penalizes any other encountered tokens (which are likely to be the
most probable ones).

### Hyperparameters selection

The most important hyperparameters are initial threshold and index of the layer
from which we get the hidden state. 
There is an example notebook in `examples/hyperparameters_test` that shows how
to derive the optimal value and reasoning behind it.

### How to run

Fleet is not bound to specific model or worker backend 
(although it uses torch to work with tensors). The library provides primitives 
for master and workers. There are examples on how to run fleet with `transformers`,
`nnsight`, `vllm`, `ray` they also show how to use special features like 
extracting priors, finetuning datasets from produced trajectories or visualizing the 
search.



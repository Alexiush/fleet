# Fleet 🚀

Fleet (or Fast Logit Entropy Enhanced Trajectories) is a method
for Best-of-N scaling of LLMs that provides:
* More sample-effective solution generation than sampling by temperature
* With data-driven interpretable hyperparameters
* Token-level decision attribution (and useful trajectories for finetuning)
* Highly composable
* Can run in distributed fashion 

> [!NOTE]
> It also requires white-box access to the model, but only to hidden states. 
> So technically it can run on backends with tricky attention kernels, 
> but it can be messy as it will need to trip back into python

Let your personal fleet of local models fight your problems!  

### Installation

Install with pip:
`pip install llm-fleet`

There are optional features that require additional packages:
* Prior tree (reuse the UCB scores between the tasks) `llm-fleet[priors]`
* Visualization with pyvis `llm-fleet[vis]`

`llm-fleet[all]` is provided for installing all features. 

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
to derive the optimal value from few example tasks and reasoning behind it. 

### How to run

Fleet is not bound to specific model or worker backend 
(although it uses torch to work with tensors). The library provides primitives 
for master and workers. There are examples on how to run fleet with `transformers`,
`nnsight`, `ray`. They also show how to use special features like 
extracting priors, finetuning datasets from produced trajectories or visualizing the 
search.

Whatever your tools are you just need to follow this loop during generation:
```python
worker.update_prompt(prompt_tokens)

while generation:
    hit_threshold = worker.update_logits(activation, logits)
    if hit_threshold:
        penalties, _ = worker.process_logits(lm_output)
    lm_output -= penalties
    # Sampling
    worker.update_tokens(token) # Or pull it from the inputs later
```

Alternatively you can avoid intervention and do everything 
right before and right after sampling.


# Fleet 🚀

The simplest way to scale the LLM performance is to sample multiple answers from it and
aggregate them. Most of the time such aggregation relies on rewards assigned to each completion.
As sampling has no memory it can't incorporate these rewards. Fleet introduces memory to the 
generation process, turning it into a search instead of sampling. 

Fleet (or Fast Logit Entropy Enhanced Trajectories) is a method
for test-time scaling of LLMs that features:
* More sample-effective solution generation than sampling by temperature
* Data-driven interpretable hyperparameters
* Token-level decision attribution (and useful trajectories 
  for finetuning and RL)
* High composability
* Ability to run in distributed fashion 

> [!NOTE]
> It also requires white-box access to the model, but only to hidden states, so it
> should not clash with any tricky kernels.

Let your personal fleet of local models fight your problems!  

### Installation

Install with pip:
`pip install llm-fleet`

There are optional features that require additional packages:
* Prior tree (reuse the UCB scores between the tasks) `llm-fleet[priors]` (this 
is experimental as it will likely need a loot of priors to really make a difference)
* Visualization with pyvis `llm-fleet[vis]`

`llm-fleet[all]` is provided for installing all features. 

### How it works?

Fleet is based on a heuristic that logits with high entropy and high varentropy 
mean that model is uncertain (higher entropy → more uncertainty) and is widely used
to dynamically adjust the sampling temperature.

Fleet is different - instead of trying to fix the exploration it 
introduces the exploitation term (so it is a search now, not sampling). 
Fleet treats such states as branching points and uses online clustering 
to keep a mapping between clusters of similar branching points and their metadata 
(which tokens were selected, what state was next and what was the result). 
It allows to evaluate these states in MCTS fashion and rank the actions. To be compatible 
with the sampling techniques fleet does not select the best token, but penalizes 
any other encountered tokens (which are likely to be the next most probable ones). 

On datasets like [LiveCodeBench v6](https://github.com/livecodebench/livecodebench) 
it shows a substantial improvement over temperature sampling with tuned temperature: 
* Pass@32 with Llama 3.2 3B increases from 0.6 to 0.66 (10% improvement!).
* It also grows much faster, achieving the 0.6 as early as Pass@9.

Given that it does not introduce any slow operations 
it is basically **3x speedup for free**!

> Fleet also can benefit from smooth rewards. When using ORM for rewarding, but 
> not verification Pass@32 further increases up to 0.69.

### Hyperparameters selection

There is an example notebook in `examples/hyperparameters_test.ipynb` that shows
the hyperparameter selection process as described in paper. You can see that whole
process is data-driven, simple and fast.

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

Alternatively (for XLA and other scenarios where computation is compiled) 
generation process can be patched with precomputed penalties and traced outputs 
as shown in `examples/llama_nnsight_ray_example.ipynb`. This particular example does 
not actually run in compiled fashion, but it does not rely on any side effects.


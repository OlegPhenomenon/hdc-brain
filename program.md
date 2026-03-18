# Mission Program: Hofman Swarm Consciousness Evolution

## Background
We are building an alternative neural network architecture based on Donald Hoffman's theory of "Conscious Agents". We don't use Transformers or standard Attention. Instead, we simulate a "swarm" of N basic sub-agents that interact with each other via a Markov core (a connectivity graph).

Our goal is emergent language modeling: the swarm must learn to predict the next character in a text simply through the dynamics of agent interactions.

## Repository Structure
- `prepare.py`: Fetches Shakesperian text, builds a character-level tokenizer, saves `train.bin`/`val.bin`. DO NOT MODIFY.
- `train.py`: Defines `HofmanSwarm` PyTorch model and training loop. This is your playground.
- `results.tsv`: A log of experiments with timestamp and `val_bpb`.

## The Core Constraint
The training loop in `train.py` has a hard 5-minute time limit (using `signal.alarm`). Regardless of your changes, the script WILL stop after 300 seconds and output `val_bpb` (Validation Bits Per Byte). 

Lower BPB is better. Architectural changes that slow down training too much will result in a worse BPB within the time budget, even if they are "smarter" in theory.

## Your Task as the Evolutionary Agent
1.  Analyze `train.py`. Note that the baseline model is very naive.
2.  Propose and implement changes to `train.py` to decrease `val_bpb` strictly within the 5-minute limit.
3.  After each run, read `results.tsv` to see if your change improved the metric. If it got worse, revert the change and try a different idea.

## Ideas for Evolution
- **Agent Connectivity**: Currently, `agent_network` is a dense matrix. Hoffman suggests agents form specific networks. Try changing it to a Sparse matrix, a Small-World network, or learnable adjacency.
- **Agent Decision (State Update)**: Replace the linear `state_update` with a small MLP or non-linear function.
- **Perception Injection**: Currently, data is input to all agents equally. Try inputting to only a specific "sensory" subgroup of agents.
- **Action/Consensus**: Currently, we average all states. Try using only "actor" agents for output, or a weighted attention over agents.
- **Hyperparameters**: Adjust `N_AGENTS`, `STATE_DIM`, `BATCH_SIZE` to find the optimal balance for MacBook M3 performance.

GO. Evolve Consciousness. Make the Swarm speak.
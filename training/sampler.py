from typing import Callable

import torch


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    inputs: torch.Tensor, # (batch_size, num_tokens)
    max_new_tokens: int,
    strategy: str,
    temperature: float = 0.6,
    eos_id: int = 7,
    **decode_kwargs
) -> torch.Tensor:

    decode_strategy = get_generation_strategy(strategy)

    for _ in range(max_new_tokens):

        # crop current context
        idx_cond = (
            inputs
            if inputs.size(1) <= model.args.context_length
            else inputs[:, - model.args.context_length :]
        )

        logits = model.forward(idx_cond)

        # last time step, (batch, n_token, vocab_size) -> (batch, vocab_size), temperature scaling
        logits = logits[:, -1, :] / temperature

        idx_next = decode_strategy(logits, **decode_kwargs)

        # stop at end-of-sequence token
        if idx_next == eos_id:
            break

        inputs = torch.cat((inputs, idx_next), dim=1)

    return inputs


def get_generation_strategy(strategy: str) -> Callable[..., torch.Tensor]:

    strategies = {
        "greedy_decode": greedy_decode,
        "multinomial_sampling": multinomial_sampling,
        "top_k_sampling": top_k_sampling,
        "top_p_sampling": top_p_sampling,
        "beam_decode": None
    }

    if strategy not in strategies:
        raise ValueError(
            f"Strategy {strategy} is not implemented or does not exist! Availables strategies: {strategies.keys()}."
        )

    return strategies[strategy]


def greedy_decode(probs: torch.Tensor) -> torch.Tensor:

    # get the idx of the vocab entry with the highest logits value
    idx_next = torch.argmax(probs, dim=-1, keepdim=True)  # (batch, 1)

    return idx_next


def multinomial_sampling(logits: torch.Tensor) -> torch.Tensor:

    # probability of each token in vocabulary
    probs = torch.softmax(logits, dim=-1)

    # get the idx of the vocab entry by multinomial sampling
    idx_next = torch.multinomial(probs, num_samples=1)

    return idx_next


def top_k_sampling(logits: torch.Tensor, top_k: int = 3) -> torch.Tensor:

    top_logits, top_pos = torch.topk(logits, top_k)

    # select top k possible tokens, assign -inf to all others in batch
    logits = torch.where(
        condition=logits < top_logits[:, -1],
        input=torch.tensor(float('-inf')),
        other=logits
    )

    # probability of each token in vocabulary
    probs = torch.softmax(logits, dim=-1)

    # get the idx of the vocab entry by multinomial sampling
    idx_next = torch.multinomial(probs, num_samples=1)

    return idx_next


def top_p_sampling(logits: torch.Tensor, top_p: float = 0.9) -> torch.Tensor:

    assert 0.0 < top_p < 1.0, "top_p must be between 0 and 1."

    # probability of each token in vocabulary
    probs = torch.softmax(logits, dim=-1)

    # sort probabilities in descending order
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)

    # create cumulative sum of elements
    probs_sum = torch.cumsum(probs_sort, dim=-1)

    # mark tokens having values over top_p
    mask = probs_sum - probs_sort > top_p
    probs_sort[mask] = 0.0

    # renormalize remaining probabilities
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))

    # get the idx of the probabilites by multinomial sampling
    idx_next = torch.multinomial(probs_sort, num_samples=1)

    # get original index
    idx_next = torch.gather(probs_idx, -1, idx_next)

    return idx_next


def beam_search():
    raise NotImplementedError
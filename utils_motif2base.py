import torch

BASE2ID = {'A':0, 'C':1, 'G':2, 'T':3}

@torch.no_grad()
def build_stencil_from_detok(detok, k_max=5, device='cuda', dtype=torch.float32, length_fair=True):
    """
    Build a fixed stencil M that 'paints' each token's bases across offsets.
    Returns M of shape [V, k_max, 4] aligned with token IDs (0..V-1).

    - detok.id2token: dict {id: token_str}
    - detok.pad_token_id: int
    - detok.base_tokens: ('A','U','C','G')
    """
    V = detok.get_vocab_size()
    pad_id = detok.pad_token_id
    vocab_strings = [detok.id2token[i] for i in range(V)]  # id order

    M = torch.zeros(V, k_max, 4, device=device, dtype=dtype)
    lens = torch.ones(V, 1, device=device, dtype=dtype)    # default 1 to avoid div-by-zero

    for t in range(V):
        if t == pad_id:
            continue  # pad paints nothing
        tok = vocab_strings[t].upper().replace('U', 'T')

        # sanity: skip anything not in A/U/C/G
        usable = True
        for ch in tok:
            if ch not in BASE2ID:
                usable = False
                break
        if not usable:
            continue

        k = min(len(tok), k_max)
        lens[t, 0] = max(k, 1)

        # paint each offset r with the base channel for tok[r]
        for r in range(k):
            b = BASE2ID[tok[r]]
            M[t, r, b] = 1.0

    if length_fair:
        # scale each token’s offsets by 1/k so 5-mers don’t dominate 3-mers
        M = M * (1.0 / lens).unsqueeze(-1)

    return M  # [V, k_max, 4]

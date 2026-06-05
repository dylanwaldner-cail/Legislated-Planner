import torch

actions = torch.load('checkpoints/point_maze/actions.pth', map_location='cpu')
seq_lengths = torch.load('checkpoints/point_maze/seq_lengths.pth', map_location='cpu')
states = torch.load('checkpoints/point_maze/states.pth', map_location='cpu')

for name, obj in [('actions', actions), ('seq_lengths', seq_lengths), ('states', states)]:
    print(f"\n=== {name} ===")
    print(f"  type: {type(obj)}")
    if isinstance(obj, torch.Tensor):
        print(f"  shape: {obj.shape}")
        print(f"  dtype: {obj.dtype}")
        print(f"  min/max: {obj.min().item():.4f} / {obj.max().item():.4f}")
        print(f"  sample (first 8): {obj.flatten()[:8].tolist()}")
    elif isinstance(obj, (list, tuple)):
        print(f"  len: {len(obj)}")
        print(f"  first element type: {type(obj[0])}")
        if isinstance(obj[0], torch.Tensor):
            print(f"  first element shape: {obj[0].shape}")
            print(f"  first element sample: {obj[0].flatten()[:8].tolist()}")
    elif isinstance(obj, dict):
        print(f"  keys: {list(obj.keys())}")
        for k, v in obj.items():
            print(f"    [{k}]: type={type(v)}", end='')
            if isinstance(v, torch.Tensor):
                print(f", shape={v.shape}, dtype={v.dtype}")
            else:
                print(f", value={v}")

# ── Cross-reference to determine frame-level vs transition-level ──────────────
print("\n=== STRUCTURE ANALYSIS ===")

def total_entries(obj):
    if isinstance(obj, torch.Tensor):
        return obj.shape[0]
    elif isinstance(obj, (list, tuple)):
        return len(obj)
    return None

n_actions = total_entries(actions)
n_states  = total_entries(states)

if isinstance(seq_lengths, torch.Tensor):
    sl = seq_lengths
elif isinstance(seq_lengths, (list, tuple)):
    sl = torch.tensor(seq_lengths)
else:
    sl = None

if sl is not None:
    print(f"  num sequences:          {len(sl)}")
    print(f"  seq_lengths unique:     {sl.unique().tolist()}")
    print(f"  seq_lengths sum:        {sl.sum().item()}")
    print(f"  actions total entries:  {n_actions}")
    print(f"  states  total entries:  {n_states}")

    sum_sl = sl.sum().item()
    n_seq  = len(sl)

    print()
    if n_actions is not None:
        if sum_sl == n_actions:
            print("  [actions] sum(seq_lengths) == n_actions → per-FRAME actions")
        elif sum_sl == n_actions + n_seq:
            print("  [actions] sum(seq_lengths) == n_actions + n_seq → likely (s,a,s') with terminal state")
        else:
            print(f"  [actions] no clean match: sum_sl={sum_sl}, n_actions={n_actions}, n_seq={n_seq}")

    if n_states is not None:
        if sum_sl == n_states:
            print("  [states]  sum(seq_lengths) == n_states → per-FRAME states (no terminal append)")
        elif sum_sl + n_seq == n_states:
            print("  [states]  n_states == sum_sl + n_seq → states include one terminal state per episode (s_0..s_T)")
        else:
            print(f"  [states]  no clean match: sum_sl={sum_sl}, n_states={n_states}, n_seq={n_seq}")

    if n_actions is not None and n_states is not None:
        diff = n_states - n_actions
        print(f"\n  states - actions = {diff}")
        if diff == n_seq:
            print("  → Each episode has one more state than actions: classic (s,a,s') transition format")
        elif diff == 0:
            print("  → states and actions same length: likely frame-aligned (s_t, a_t) pairs, no terminal state stored")
        else:
            print(f"  → Difference {diff} doesn't match n_seq={n_seq}; may be ragged or a different encoding")

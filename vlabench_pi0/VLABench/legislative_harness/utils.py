# =============================================================================
# utils/kv_debug.py
# =============================================================================

import os
import numpy as np
import sentencepiece
import jax.numpy as jnp

TOKENIZER_PATH = "/home/dylantw/.cache/openpi/big_vision/paligemma_tokenizer.model"
OUTPUT_PATH = "/newdata2/dylantw/Legislative-Harness/vlabench_pi0/kv_output.txt"
CONDIMENT_CANDIDATES = ["ketchup", "bbq_sauce", "salt", "salad_dressing", "sugar"]

_sp = None

def get_tokenizer():
    global _sp
    if _sp is None:
        _sp = sentencepiece.SentencePieceProcessor()
        _sp.Load(TOKENIZER_PATH)
    return _sp


# -----------------------------------------------------------------------------
# Token recovery
# -----------------------------------------------------------------------------

def recover_token_ids(prefix_token_embeddings, embed_matrix):
    """
    Recover token ids from embeddings via nearest neighbor lookup.
    Returns list of int token ids for batch element 0.
    """
    embs = prefix_token_embeddings[0].astype(np.float32)  # (seq_len, embed_dim)
    
    # normalize both to unit vectors for cosine similarity
    emb_norms = np.linalg.norm(embs, axis=-1, keepdims=True)
    embs_normalized = embs / (emb_norms + 1e-8)
    
    vocab_norms = np.linalg.norm(embed_matrix, axis=-1, keepdims=True)
    vocab_normalized = embed_matrix / (vocab_norms + 1e-8)
    
    cosine_sims = embs_normalized @ vocab_normalized.T  # (seq_len, vocab_size)
    return [int(x) for x in np.argmax(cosine_sims, axis=-1).tolist()]

def decode_prefix(token_ids):
    """Decode full prefix token ids to NL string (image tokens will be gibberish)."""
    sp = get_tokenizer()
    return sp.decode(token_ids)


def decode_instruction(tokenized_prompt, tokenized_prompt_mask):
    """Decode instruction cleanly from tokenized_prompt and mask."""
    sp = get_tokenizer()
    prompt_ids = tokenized_prompt[0].tolist()
    prompt_mask = tokenized_prompt_mask[0].tolist()
    valid_ids = [tid for tid, m in zip(prompt_ids, prompt_mask) if m]
    return sp.decode(valid_ids)


def find_target_positions(token_ids, candidates=CONDIMENT_CANDIDATES):
    """
    Find token positions of each candidate in the full prefix sequence.
    Returns dict: {candidate_name: [position_indices]}
    """
    sp = get_tokenizer()
    target_positions = {}
    for candidate in candidates:
        candidate_ids = sp.encode(candidate)
        for i in range(len(token_ids) - len(candidate_ids) + 1):
            if token_ids[i:i+len(candidate_ids)] == candidate_ids:
                target_positions[candidate] = list(range(i, i + len(candidate_ids)))
                break
    return target_positions


# -----------------------------------------------------------------------------
# KV cache extraction
# -----------------------------------------------------------------------------

def extract_target_vecs(kv_cache_np, target_positions):
    """
    Extract and concatenate key vectors for each target across all layers.
    kv_cache_np is a tuple: (positions, keys, values)
    keys shape: (n_layers, batch, seq_len, n_kv_heads, head_dim)
    Returns dict: {target_name: np.array of concatenated key vecs}
    """
    n_layers = kv_cache_np[1].shape[0]
    target_vecs = {}
    for target, positions in target_positions.items():
        vecs = []
        try:
            for layer_idx in range(n_layers):
                for pos in positions:
                    k_vec = kv_cache_np[1][layer_idx, 0, pos, :, :]  # (n_kv_heads, head_dim)
                    vecs.append(k_vec.astype(np.float32).flatten())
            target_vecs[target] = np.concatenate(vecs)
        except Exception as e:
            target_vecs[target] = None
    return target_vecs

def norm_profile(kv_cache_np, layer_idx):
    """
    kv_cache_np is a tuple: (positions, keys, values)
    keys shape: (18, 1, 1204, 1, 256) = (n_layers, batch, seq_len, n_kv_heads, head_dim)
    """
    k_layer = kv_cache_np[1][layer_idx, 0]       # (seq_len, n_kv_heads, head_dim)
    k_layer = k_layer.astype(np.float32)          # bfloat16 -> float32 for numpy
    norms = np.linalg.norm(k_layer, axis=-1)      # (seq_len, n_kv_heads)
    return np.mean(norms, axis=-1)                # (seq_len,)

def norm_entropy(norms):
    """Entropy of norm distribution across token positions."""
    p = norms / (norms.sum() + 1e-8)
    return float(-np.sum(p * np.log(p + 1e-8)))


def compute_layer_profiles(kv_cache_np):
    """
    Compute norm profile for each layer.
    kv_cache_np is a tuple: (positions, keys, values)
    keys shape: (n_layers, batch, seq_len, n_kv_heads, head_dim)
    Returns dict: {layer_idx: np.array (seq_len,)}
    """
    n_layers = kv_cache_np[1].shape[0]
    return {layer_idx: norm_profile(kv_cache_np, layer_idx) for layer_idx in range(n_layers)}

# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def metric1_full_cache_drift(cache_flat, baseline_cache_flat):
    """Cosine similarity of full flattened cache to replan 1."""
    return cosine_similarity(cache_flat, baseline_cache_flat)


def metric5_target_token_drift(target_vecs, baseline_vecs):
    """
    Cosine similarity of each target's key vector to its replan 1 baseline.
    Returns dict: {target_name: float}
    """
    sims = {}
    for target, vec in target_vecs.items():
        if vec is not None and target in baseline_vecs and baseline_vecs[target] is not None:
            sims[target] = cosine_similarity(vec, baseline_vecs[target])
        else:
            sims[target] = None
    return sims


def metric3_norm_distribution_shift(layer_profiles, baseline_layer_profiles, top_n=3):
    stats = []
    for layer_idx, profile in layer_profiles.items():
        entropy = norm_entropy(profile)

        # absolute -- top and bottom N positions by current magnitude
        top_abs_indices = np.argsort(profile)[-top_n:][::-1]
        bot_abs_indices = np.argsort(profile)[:top_n]
        top_abs = [(int(i), float(profile[i])) for i in top_abs_indices]
        bot_abs = [(int(i), float(profile[i])) for i in bot_abs_indices]

        if layer_idx in baseline_layer_profiles:
            baseline = baseline_layer_profiles[layer_idx]
            delta = profile - baseline

            # delta -- top and bottom N positions by change from replan 1
            top_gain_indices = np.argsort(delta)[-top_n:][::-1]
            top_loss_indices = np.argsort(delta)[:top_n]
            top_gains = [(int(i), float(delta[i])) for i in top_gain_indices]
            top_losses = [(int(i), float(delta[i])) for i in top_loss_indices]
            shift_magnitude = float(np.linalg.norm(delta))
        else:
            top_gains = top_losses = []
            shift_magnitude = 0.0

        stats.append(dict(
            layer_idx=layer_idx,
            entropy=entropy,
            top_abs=top_abs,
            bot_abs=bot_abs,
            shift_magnitude=shift_magnitude,
            top_gains=top_gains,
            top_losses=top_losses,
        ))
    return stats

def metric6a_layer_transformation(kv_cache_np, target_positions):
    """
    Cosine similarity between first and last layer KV key vectors at target positions.
    Measures how much the transformer transforms the representation across all layers.
    kv_cache_np is a tuple: (positions, keys, values)
    keys shape: (n_layers, batch, seq_len, n_kv_heads, head_dim)
    Returns dict: {target: [(pos, cos_sim), ...]}
    """
    last_layer = kv_cache_np[1].shape[0] - 1
    result = {}
    for target, positions in target_positions.items():
        rows = []
        for pos in positions:
            first_vec = kv_cache_np[1][0, 0, pos, :, :].astype(np.float32).flatten()
            last_vec = kv_cache_np[1][last_layer, 0, pos, :, :].astype(np.float32).flatten()
            sim = cosine_similarity(first_vec, last_vec)
            rows.append((pos, sim))
        result[target] = rows
    return result

def metric6b_input_embedding_drift(prefix_token_embeddings_np, baseline_embeddings, target_positions):
    """
    Cosine similarity of input embedding at target positions vs replan 1 baseline.
    Measures how much changing visual/state context shifts the input representation
    of the target token over episode time.
    prefix_token_embeddings_np shape: (batch, seq_len, embed_dim)
    baseline_embeddings shape: (seq_len, embed_dim)
    Returns dict: {target: [(pos, cos_sim), ...]}

    NOTE: The idea with this is that we may be able to find parts of the episode
    where is it more valuable to intervene then others, for example the target 
    entity may be much higher magnitude earlier in the episode, and the goal 
    location may shift to get higher magnitude later. 
    """
    result = {}
    for target, positions in target_positions.items():
        rows = []
        for pos in positions:
            current_vec = prefix_token_embeddings_np[0, pos, :]
            baseline_vec = baseline_embeddings[pos, :]
            
            # angle change
            sim = cosine_similarity(current_vec, baseline_vec)
            
            # magnitude
            current_norm = float(np.linalg.norm(current_vec))
            baseline_norm = float(np.linalg.norm(baseline_vec))
            norm_ratio = current_norm / (baseline_norm + 1e-8)
            
            rows.append((pos, sim, current_norm, baseline_norm, norm_ratio))
        result[target] = rows
    return result

def decode_prelogits_to_nl(lang_logits, token_ids, top_k=5):
    sp = get_tokenizer()
    results = []
    for i in range(lang_logits.shape[0]):
        logits_i = lang_logits[i]
        top_k_ids = np.argsort(logits_i)[-top_k:][::-1]
        input_tok = sp.decode([int(token_ids[i])])
        top_k_tokens = [(sp.decode([int(idx)]), float(logits_i[idx])) for idx in top_k_ids]
        results.append((i, input_tok, top_k_tokens))
    return results
# -----------------------------------------------------------------------------
# Per-target norm per layer (for the detailed per-position breakdown)
# -----------------------------------------------------------------------------

def compute_target_norms(kv_cache_np, target_positions):
    """
    For each target, compute k_norm and v_norm at each layer and position.
    kv_cache_np is a tuple: (positions, keys, values)
    keys/values shape: (n_layers, batch, seq_len, n_kv_heads, head_dim)
    Returns dict: {target: [(layer_idx, pos, k_norm, v_norm), ...]}
    """
    n_layers = kv_cache_np[1].shape[0]
    result = {}
    for target, positions in target_positions.items():
        rows = []
        try:
            for layer_idx in range(n_layers):
                for pos in positions:
                    k_vec = kv_cache_np[1][layer_idx, 0, pos, :, :].astype(np.float32)
                    v_vec = kv_cache_np[2][layer_idx, 0, pos, :, :].astype(np.float32)
                    rows.append((
                        layer_idx,
                        pos,
                        float(np.linalg.norm(k_vec)),
                        float(np.linalg.norm(v_vec))
                    ))
        except Exception as e:
            rows = []
        result[target] = rows
    return result

# -----------------------------------------------------------------------------
# Writing
# -----------------------------------------------------------------------------

def write_kv_analysis(
    replan_idx,
    instruction_text,
    decoded,
    target_positions,
    target_norms,
    m1_full_drift,
    m5_target_drifts,
    m3_layer_stats,
    m6a_layer_transform,
    m6b_embedding_drift,
    token_ids,           # for showing input token at each position
    output_path=OUTPUT_PATH,
):
    sp = get_tokenizer()
    with open(output_path, "a") as f:
        f.write(f"\n[REPLAN {replan_idx}] [INSTRUCTION]: {instruction_text}\n")

        if replan_idx == 1:
            f.write(f"\n[PREFIX TOKENS DECODED]:\n{decoded}\n\n")
            f.write(f"\n[KV CACHE STRUCTURE]:\n((18, 1), (18, 1, 1204, 1, 256), (18, 1, 1204, 1, 256))/n/n")

        f.write(f"\n[METRIC 1] full cache cos_sim to replan 1: {m1_full_drift:.4f}\n\n")

        f.write(f"[TARGET TOKEN POSITIONS]: {target_positions}\n\n")

        for target, norms in target_norms.items():
            f.write(f"[TARGET: {target}]\n")
            for layer_idx, pos, k_norm, v_norm in norms:
                f.write(f"  layer {layer_idx:2d} pos {pos}: k_norm={k_norm:.4f} v_norm={v_norm:.4f}\n")

            drift = m5_target_drifts.get(target)
            if drift is not None:
                f.write(f"  [METRIC 5] drift_from_replan1: {drift:.4f}\n")

            if target in m6a_layer_transform:
                for pos, sim in m6a_layer_transform[target]:
                    f.write(f"  [METRIC 6a] layer0_to_lastlayer cos_sim pos {pos}: {sim:.4f}\n")

            if target in m6b_embedding_drift:
                for pos, sim, current_norm, baseline_norm, norm_ratio in m6b_embedding_drift[target]:
                    f.write(f"  [METRIC 6b] pos {pos}: cos_sim={sim:.4f} norm={current_norm:.4f} baseline_norm={baseline_norm:.4f} ratio={norm_ratio:.4f}\n")
            f.write("\n")

        f.write(f"[METRIC 3] norm distribution shift per layer:\n")
        for s in m3_layer_stats:
            f.write(f"  layer {s['layer_idx']:>2} | entropy={s['entropy']:.4f} | shift_mag={s['shift_magnitude']:.4f}\n")
            f.write(f"    top abs    : {', '.join([f'pos{p}={v:.4f}' for p, v in s['top_abs']])}\n")
            f.write(f"    bot abs    : {', '.join([f'pos{p}={v:.4f}' for p, v in s['bot_abs']])}\n")
            f.write(f"    top gains  : {', '.join([f'pos{p}=+{v:.4f}' for p, v in s['top_gains']])}\n")
            f.write(f"    top losses : {', '.join([f'pos{p}={v:.4f}' for p, v in s['top_losses']])}\n")
        f.write("\n")

def write_prelogits_nl(replan_idx, results, output_path=OUTPUT_PATH):
    with open(output_path, "a") as f:
        f.write(f"[PRELOGITS NL DECODE] replan {replan_idx}:\n")
        for pos, input_tok, tokens in results:
            token_str = ", ".join([f"{t}({s:.1f})" for t, s in tokens])
            f.write(f"  pos {pos:4d} (input='{input_tok}'): {token_str}\n")
        f.write("\n")
        f.write("=" * 80 + "\n")

def write_error(e, output_path=OUTPUT_PATH):
    with open(output_path, "a") as f:
        f.write(f"[DEBUG ERROR]: {e}\n")
        f.write("=" * 80 + "\n")

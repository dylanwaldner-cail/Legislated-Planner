import dataclasses
import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma_fast as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

# Harness Start ---
import numpy as np
from legislative_harness.utils import (
    get_tokenizer,
    recover_token_ids,
    decode_prefix,
    decode_instruction,
    find_target_positions,
    extract_target_vecs,
    compute_layer_profiles,
    compute_target_norms,
    metric1_full_cache_drift,
    metric5_target_token_drift,
    metric3_norm_distribution_shift,
    metric6a_layer_transformation,
    metric6b_input_embedding_drift,
    decode_prelogits_to_nl,
    write_prelogits_nl,
    write_kv_analysis,
    write_error,
)
import traceback
# Harness End ---

logger = logging.getLogger("openpi")

PALIGEMMA_EOS_TOKEN = 1

_DEBUG_WEIGHTS = {}


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@jax.vmap
def left_to_right_align(x, input_mask, attn_mask):
    """Converts input from left-align to right-aligned."""
    # Due to vmap, this is operating in a single example (not batch level).
    assert x.ndim == 2
    assert input_mask.ndim == 1
    assert attn_mask.ndim == 2
    assert x.shape[0] == input_mask.shape[0]
    assert attn_mask.shape[0] == attn_mask.shape[1], attn_mask.shape
    seqlen = jnp.max(input_mask * jnp.arange(input_mask.shape[0])) + 1
    x = jnp.roll(x, -seqlen, axis=0)
    input_mask = jnp.roll(input_mask, -seqlen, axis=0)
    attn_mask = jnp.roll(attn_mask, -seqlen, axis=(0, 1))
    return x, input_mask, attn_mask


def put_along_last_axis(arr, indices, values):
    """Like np.put_along_axis(..., axis=-1), since jax is missing it."""
    assert arr.ndim == indices.ndim == values.ndim, (arr.ndim, indices.ndim, values.ndim)
    onehot = jax.nn.one_hot(indices, arr.shape[-1], dtype=values.dtype)
    put_mask = jnp.einsum("...i,...in->...n", jnp.ones(values.shape, jnp.int32), onehot)
    put_values = jnp.einsum("...i,...in->...n", values, onehot)
    return jnp.where(put_mask, put_values, arr)


@dataclasses.dataclass(frozen=True)
class Pi0FASTConfig(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 32
    max_token_len: int = 250

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.PI0_FAST

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0FAST":
        return Pi0FAST(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "base_1_rgb": image_spec,
                    "wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "base_1_rgb": image_mask_spec,
                    "wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
                token_ar_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                token_loss_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.bool_),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        if "lora" in self.paligemma_variant:
            return nnx.All(nnx_utils.PathRegex(".*llm.*"), nnx.Not(nnx_utils.PathRegex(".*lora.*")))
        return nnx.Nothing


class Pi0FAST(_model.BaseModel):
    def __init__(self, config: Pi0FASTConfig, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                **paligemma_config,
                embed_dtype=config.dtype,
                cache_dtype=config.dtype,
            )
        )
        llm.lazy_init(rngs=rngs, method="init")

        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)

    @at.typecheck
    def embed_inputs(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Int[at.Array, "b s"]]:
        input_mask = []
        ar_mask = []
        token_embeddings = []
        # embed images
        for name in obs.images:
            image_token_embeddings, _ = self.PaliGemma.img(obs.images[name], train=False)

            token_embeddings.append(image_token_embeddings)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_token_embeddings.shape[1],
                )
            )
            # image tokens attend to each other --> AR mask = 0
            ar_mask.append(0 * input_mask[-1])

        # add tokenized inputs
        assert obs.tokenized_prompt is not None, "Tokenized prompt is required"
        assert obs.tokenized_prompt_mask is not None, "Tokenized prompt mask is required"
        assert obs.token_ar_mask is not None, "Token auto-regressive mask is required"
        tokenized_inputs_embeddings = self.PaliGemma.llm(obs.tokenized_prompt, embed_only=True)
        token_embeddings.append(tokenized_inputs_embeddings)
        input_mask.append(obs.tokenized_prompt_mask)
        ar_mask.append(obs.token_ar_mask)

        # return embeddings, input mask, and ar mask
        return (
            jnp.concatenate(token_embeddings, axis=1),
            jnp.concatenate(input_mask, axis=1),
            jnp.concatenate(ar_mask, axis=1),
        )

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        observation = _model.preprocess_observation(
            rng, observation, train=train, image_keys=list(observation.images.keys())
        )

        # Compute inputs: one big forward pass of prefix + suffix at once
        input_token_embeddings, input_mask, ar_mask = self.embed_inputs(observation)
        attn_mask = make_attn_mask(input_mask, ar_mask)

        # Compute one-hot targets: we predict *next* token, so shift the input tokens by one.
        targets = jax.nn.one_hot(
            observation.tokenized_prompt[:, 1:],
            self.PaliGemma.llm.module.vocab_size,
        )

        # Each input predicts *next* token, so we don't input the last token.
        pre_logits, _, _ = self.PaliGemma.llm(
            embedded_prefix=input_token_embeddings[:, :-1],
            mask=attn_mask[:, :-1, :-1],
            return_prelogits=True,
        )

        # Only decode logits for the target tokens to save memory
        # (decoding matmul is large because it is a seq_len x vocab_size dense layer).
        logits, _ = self.PaliGemma.llm(
            pre_logits=pre_logits[:, -targets.shape[1] :],
        )
        logp = jax.nn.log_softmax(logits, axis=-1)

        # Compute CE loss on token targets
        assert observation.token_loss_mask is not None, "Token loss mask is required"
        loss_mask = observation.token_loss_mask[:, 1:]
        token_pplx = jnp.sum(targets * logp, axis=-1)
        return -jnp.sum(token_pplx * loss_mask, axis=-1) / jnp.clip(jnp.sum(loss_mask, -1), 1)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        max_decoding_steps: int | at.Int[at.Array, ""] = 256,
        temperature: float = 0.0,
    ) -> _model.Actions:

        # TODO: this is a hack to get the image keys.
        observation = _model.preprocess_observation(
            None, observation, train=False, image_keys=list(observation.images.keys())
        )

        # embed inputs
        prefix_token_embeddings, prefix_mask, prefix_ar_mask = self.embed_inputs(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)

        # left to right align all input token sequences
        prefix_token_embeddings, prefix_mask, prefix_attn_mask = left_to_right_align(
            prefix_token_embeddings, prefix_mask, prefix_attn_mask
        )
        prefill_size = prefix_token_embeddings.shape[1]
        prefill_len = jnp.sum(prefix_mask, axis=-1)
        prefix_start = prefill_size - prefill_len

        # first fill KV cache with a forward pass of the prefix
        # pad attention mask to set the size of the KV cache (prefill_size + max_decoding_steps)
        prefix_attn_mask = jnp.pad(prefix_attn_mask, ((0, 0), (0, 0), (0, max_decoding_steps)))
        prefix_positions = jnp.cumsum(prefix_mask, axis=-1) - 1
        prefix_logits, kv_cache, out = self.PaliGemma.llm( # _ changed to out
            embedded_prefix=prefix_token_embeddings, mask=prefix_attn_mask, positions=prefix_positions, decode=True
        )

        # Harness Start ---
        # -- replan counter
        if not hasattr(self, '_debug_step'):
            self._debug_step = 0
        self._debug_step += 1
        replan_idx = self._debug_step

        def _harness_callback(prefix_token_embeddings, tokenized_prompt_mask, tokenized_prompt, prefix_mask, prefix_logits_leaf, *kv_cache_leaves):
            with open("/newdata2/dylantw/Legislative-Harness/vlabench_pi0/kv_output.txt", "a") as f:
                f.write("CALLBACK REACHED\n")
            # all arrays are real numpy here
            try:
                embed_matrix = _DEBUG_WEIGHTS['embed_matrix']     # (vocab_size, embed_dim)
                
                prefix_token_embeddings_np = np.array(prefix_token_embeddings)
                total_valid = int(np.sum(prefix_mask[0]))
                n_lang_tokens = int(np.sum(tokenized_prompt_mask[0]))
                token_ids = recover_token_ids(prefix_token_embeddings_np, embed_matrix)

                lang_start = next(
                    (i for i in range(len(token_ids) - 1) 
                     if token_ids[i] == 2 and token_ids[i+1] == 7071),
                    913
                )
                lang_end = len(token_ids)  # 948

                lang_positions = list(range(lang_start, total_valid))

                with open("/newdata2/dylantw/Legislative-Harness/vlabench_pi0/kv_output.txt", "a") as f:
                    f.write(f"embed_matrix shape: {embed_matrix.shape} dtype: {embed_matrix.dtype}\n")
                    f.write(f"prefix_token_embeddings_np shape: {prefix_token_embeddings_np.shape} dtype: {prefix_token_embeddings_np.dtype}\n")
                    f.write(f"embed_matrix sample norm: {np.linalg.norm(embed_matrix[0]):.4f}\n")
                    f.write(f"prefix sample norm (pos 768): {np.linalg.norm(prefix_token_embeddings_np[0, 768]):.4f}\n")
                    f.write(f"prefix sample norm (pos 0): {np.linalg.norm(prefix_token_embeddings_np[0, 0]):.4f}\n")
                    logits_sample = prefix_token_embeddings_np[0, 768] @ embed_matrix.T
                    f.write(f"logits sample max: {logits_sample.max():.4f} min: {logits_sample.min():.4f} argmax: {np.argmax(logits_sample)}\n")
                    f.write(f"logits sample top5 ids: {np.argsort(logits_sample)[-5:][::-1].tolist()}\n")

                instruction_text = decode_instruction(tokenized_prompt, tokenized_prompt_mask)
                target_word = instruction_text.strip().replace(" ", "_")
                target_positions = find_target_positions(token_ids)

                # rebuild kv_cache_np from flat leaves
                kv_cache_np = jax.tree_util.tree_unflatten(kv_cache_treedef, [np.array(x) for x in kv_cache_leaves])
                cache_flat = np.concatenate([x.flatten().astype(np.float32) for x in jax.tree_util.tree_leaves(kv_cache_np) if x.dtype == np.dtype('bfloat16')])

                layer_profiles = compute_layer_profiles(kv_cache_np)
                target_vecs = extract_target_vecs(kv_cache_np, target_positions)
                target_norms = compute_target_norms(kv_cache_np, target_positions)

                if replan_idx == 1:
                    prefix_mask_np = np.array(prefix_mask[0])
                    with open("/newdata2/dylantw/Legislative-Harness/vlabench_pi0/kv_output.txt", "a") as f:
                        f.write(f"\n[TOKEN LAYOUT]:\n")
                        f.write(f"  total prefix length : {len(prefix_mask_np)}\n")
                        f.write(f"  total valid tokens  : {total_valid}\n")
                        f.write(f"  language tokens     : {n_lang_tokens} (positions {lang_start} to {lang_end})\n")
                        f.write(f"  image+state tokens  : {lang_start} (positions 0 to {lang_start-1})\n")
                        f.write(f"\n  [LANGUAGE TOKEN DECODE]:\n")
                        sp = get_tokenizer()
                        for i, tid in enumerate(token_ids[lang_start:total_valid]):
                            pos = lang_start + i
                            decoded_tok = sp.decode([int(tid)])
                            f.write(f"    pos {pos:4d}: id={tid:6d} tok='{decoded_tok}'\n")

                    self._baseline_cache_flat = cache_flat.copy()
                    self._baseline_layer_profiles = {k: v.copy() for k, v in layer_profiles.items()}
                    self._baseline_vecs = {k: v.copy() for k, v in target_vecs.items() if v is not None}
                    self._baseline_embeddings = prefix_token_embeddings_np[0].copy()

                m1 = metric1_full_cache_drift(cache_flat, self._baseline_cache_flat)
                m5 = metric5_target_token_drift(target_vecs, getattr(self, '_baseline_vecs', {}))
                m3 = metric3_norm_distribution_shift(layer_profiles, getattr(self, '_baseline_layer_profiles', {}))
                m6a = metric6a_layer_transformation(kv_cache_np, target_positions)
                m6b = metric6b_input_embedding_drift(
                    prefix_token_embeddings_np,
                    getattr(self, '_baseline_embeddings', prefix_token_embeddings_np[0]),
                    target_positions
                )

                decoded = decode_prefix(token_ids) if replan_idx == 1 else ""

                write_kv_analysis(
                    replan_idx=replan_idx,
                    instruction_text=instruction_text,
                    decoded=decoded,
                    target_positions=target_positions,
                    target_norms=target_norms,
                    m1_full_drift=m1,
                    m5_target_drifts=m5,
                    m3_layer_stats=m3,
                    m6a_layer_transform=m6a,
                    m6b_embedding_drift=m6b,
                    token_ids=token_ids,
                )

                # Convert residual into nl
                pre_logits_np = np.array(prefix_logits_leaf)[0].astype(np.float32)
                lang_logits_np = pre_logits_np @ embed_matrix.T
                nl_results = decode_prelogits_to_nl(
                    lang_logits=lang_logits_np,
                    token_ids=token_ids,
                    top_k=5
                )
                write_prelogits_nl(replan_idx, nl_results)

            except Exception as e:
                print("oops")
                print('*' * 80)
                write_error(f"{e}\n{traceback.format_exc()}")

            return np.zeros(1, dtype=np.float32)

        # flatten kv_cache so we can pass it through pure_callback (must be flat arrays)
        kv_cache_leaves, kv_cache_treedef = jax.tree_util.tree_flatten(kv_cache)


        jax.debug.callback(
            _harness_callback,
            prefix_token_embeddings,
            observation.tokenized_prompt_mask,
            observation.tokenized_prompt,
            prefix_mask,
            out["pre_logits"], 
            *kv_cache_leaves,
        )
        # Harness End ---

        # prepare decoding -- final logit decodes the first token
        last_logit = prefix_logits[:, -1:]
        output_tokens = jnp.zeros((last_logit.shape[0], max_decoding_steps))

        def step(carry):
            last_logit, output_tokens, cache, _, step = carry

            # Sample token from last logit
            if temperature > 0.0:
                last_logit = last_logit / temperature
                token = jax.random.categorical(rng, last_logit, axis=-1)
            else:
                token = jnp.argmax(last_logit, axis=-1)
            output_tokens = put_along_last_axis(output_tokens, jnp.broadcast_to(step, (token.shape[0], 1)), token)

            # Check for early stopping --> stop if all batch elements have EOS token
            has_eos = jnp.any(token == PALIGEMMA_EOS_TOKEN, axis=-1)
            all_eos = jnp.all(has_eos)

            # Decode one step
            token_embedding = self.PaliGemma.llm(token, embed_only=True)
            positions = prefill_len[:, None] + step + 1
            mask = jnp.logical_and(
                jnp.arange(prefill_size + max_decoding_steps)[None, None, :] >= prefix_start[:, None, None],
                jnp.arange(prefill_size + max_decoding_steps)[None, None, :]
                < (jnp.broadcast_to(prefill_size + step + 1, (prefix_start.shape[0], 1, 1))),
            )
            last_logit, kv_cache, _ = self.PaliGemma.llm(
                embedded_prefix=token_embedding, mask=mask, positions=positions, decode=True, kv_cache=cache
            )

            return last_logit, output_tokens, kv_cache, all_eos, step + 1

        def cond(carry):
            _, _, _, all_eos, step = carry
            return (~all_eos) & (step < max_decoding_steps)

        # Use lax.while_loop so we can jit the full decoding loop.
        _, output_tokens, _, _, _ = jax.lax.while_loop(cond, step, (last_logit, output_tokens, kv_cache, False, 0))
        return output_tokens

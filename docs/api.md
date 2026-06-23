# API Reference

## save_delta_from_paths

```python
dt.save_delta_from_paths(
    out_path,
    finetuned_dir,
    base_dir,
    strategy="sparse",
    prefetch=2,
    **kwargs,
) -> str
```

Streaming delta save. Peak RAM is O(prefetch tensors), not O(two full models).

**Args:**

| Parameter | Type | Description |
|---|---|---|
| `out_path` | `str \| Path` | Output `.wdelta` file path |
| `finetuned_dir` | `str \| Path` | Folder containing fine-tuned safetensors shards |
| `base_dir` | `str \| Path` | Folder containing base safetensors shards |
| `strategy` | `str` | `"sparse"`, `"quantized"`, or `"int4"` |
| `prefetch` | `int` | Number of tensor pairs to prefetch (default 2) |
| `**kwargs` | | Strategy-specific options (see below) |

**Strategy kwargs:**

| Strategy | kwarg | default | description |
|---|---|---|---|
| `sparse` | `sparsity` | `0.9` | Fraction of weights to zero out |
| `int4` | `outlier_fraction` | `0.01` | Fraction of weights stored as float16 outliers |

**Returns:** SHA-256 hex hash of the base model.

---

## load_delta_from_paths

```python
dt.load_delta_from_paths(
    path,
    base_dir,
    verify=True,
) -> Dict[str, np.ndarray]
```

Reconstruct a fine-tuned model from a `.wdelta` file and a base model directory. Loads each base shard once — O(n_shards) file opens rather than O(n_tensors × n_shards).

**Args:**

| Parameter | Type | Description |
|---|---|---|
| `path` | `str \| Path` | Path to the `.wdelta` file |
| `base_dir` | `str \| Path` | Folder containing base safetensors shards |
| `verify` | `bool` | SHA-256 verify base before reconstructing (default `True`) |

**Returns:** Reconstructed state dict as `Dict[str, np.ndarray]`.

---

## inspect

```python
dt.inspect(path) -> dict
```

Return metadata from a `.wdelta` file without loading the base model.

**Args:**

| Parameter | Type | Description |
|---|---|---|
| `path` | `str \| Path` | Path to the `.wdelta` file |

**Returns:**
```python
{
    "path": "checkpoint.wdelta",
    "size_mb": 294.2,
    "parent_hash": "e1810a...",
    "strategy": "int4",
    "n_tensors": 290,
    "tensors": {
        "model.embed_tokens.weight": {"shape": [151936, 896], "dtype": "float32"},
        ...
    }
}
```

---

## save_delta

```python
dt.save_delta(
    path,
    finetuned,
    base,
    strategy="sparse",
    **kwargs,
) -> str
```

Compute and save the delta between `finetuned` and `base`. Loads both models fully into RAM — for models larger than ~3B use `save_delta_from_paths` instead.

**Args:**

| Parameter | Type | Description |
|---|---|---|
| `path` | `str \| Path` | Output `.wdelta` file path |
| `finetuned` | `Dict[str, np.ndarray \| Tensor]` | Fine-tuned state dict |
| `base` | `Dict[str, np.ndarray \| Tensor]` | Base state dict |
| `strategy` | `str` | `"sparse"`, `"quantized"`, or `"int4"` |

**Returns:** SHA-256 hex hash of the base model.

---

## load_delta

```python
dt.load_delta(
    path,
    base,
    verify=True,
) -> Dict[str, np.ndarray]
```

Reconstruct a fine-tuned model from a `.wdelta` file and a base state dict. Requires the full base loaded in RAM — for large models use `load_delta_from_paths` instead.

**Args:**

| Parameter | Type | Description |
|---|---|---|
| `path` | `str \| Path` | Path to the `.wdelta` file |
| `base` | `Dict[str, np.ndarray \| Tensor]` | Base state dict |
| `verify` | `bool` | SHA-256 verify base before reconstructing (default `True`) |

**Returns:** Reconstructed state dict as `Dict[str, np.ndarray]`.
# Format Spec

## Overview

`.wdelta` is a binary format for storing compressed weight deltas between a base model and a fine-tuned model.

## Layout

```
[0:7]      Magic bytes: b'wdelta\x00'
[7:11]     Version:     uint32 little-endian (currently 1)
[11:15]    Header len:  uint32 little-endian (bytes of JSON that follow)
[15:15+H]  JSON header (UTF-8)
[15+H:-32] Binary payload: concatenated numpy arrays in order of tensor index
[-32:]     SHA-256 checksum of all preceding bytes
```

## JSON header

```json
{
  "parent_hash": "<sha256 hex of base model>",
  "strategy": "sparse | quantized | int4",
  "tensors": {
    "<tensor_name>": {
      "strategy": "...",
      "shape": [...],
      "dtype": "...",
      "<array_field>": {"_ref": "<field_name>"}
    }
  }
}
```

Array fields are replaced with `{"_ref": "<field_name>"}` references — the actual array data lives in the binary payload section.

## Binary payload

Each array is serialised as:

```
[4 bytes]  tensor name length (uint32 LE)
[N bytes]  tensor name (UTF-8)
[4 bytes]  field name length (uint32 LE)
[N bytes]  field name (UTF-8)
[4 bytes]  dtype string length (uint32 LE)
[N bytes]  dtype string (UTF-8, e.g. "float16", "uint8")
[4 bytes]  ndim (uint32 LE)
[8×ndim]   shape dimensions (uint64 LE each)
[8 bytes]  data length in bytes (uint64 LE)
[N bytes]  raw array bytes
```

## Compression strategies

### sparse

Keeps the top `(1 - sparsity)` fraction of delta weights by magnitude. Stores indices and values in CSR style.

**Payload arrays:** `indices` (int64), `values` (float32)

**Metadata:** `sparsity`, `shape`, `dtype`

---

### quantized

1-bit sign mask with a learned per-row float16 scale (BitDelta-style).

Reconstructed weight: `scale[row] × sign[row, col]`

**Payload arrays:** `scales` (float16), `packed_signs` (uint8)

**Metadata:** `shape`, `dtype`, `n_elements`, `n_cols`

---

### int4

Outlier extraction + 4-bit quantization:

1. Compute absolute delta magnitudes
2. Extract top-k% outliers → stored as float16
3. Quantize remaining weights into 4-bit unsigned integers via asymmetric min-max scaling
4. Bit-pack pairs of int4 values into uint8 bytes

**Payload arrays:** `outlier_idx` (int64), `outlier_vals` (float16), `scale` (float16), `zero_point` (float16), `packed` (uint8)

**Metadata:** `shape`, `dtype`, `outlier_fraction`, `n_elements`, `n_outliers`

## Checksum

The final 32 bytes are a SHA-256 digest of all preceding content. Verified on read — if the file is corrupted or truncated, `read_wdelta` raises `ValueError`.

## Versioning

The version field is currently `1`. Future breaking changes to the format will increment this. `read_wdelta` raises `ValueError` on unsupported versions.
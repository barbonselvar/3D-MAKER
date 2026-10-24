Synthetic outputs written in the documented formats of
`nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader,nounits`
and `rocm-smi --showproductname --showmeminfo vram --json`. They were **not** captured on real
hardware (no GPU was available); replace them with captured outputs once a GPU host exists.

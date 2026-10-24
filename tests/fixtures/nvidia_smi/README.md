Synthetic outputs written in the documented formats of
`nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader,nounits`
and `rocm-smi --showproductname --showmeminfo vram --json`. They were **not** captured on real
hardware (no GPU was available); replace them with captured outputs once a GPU host exists.

The `rtx4050_laptop_win*.txt` and `banner_win_*.txt` files imitate a Windows 11 laptop with an
RTX 4050 Laptop GPU (6 GB, WDDM driver model). They are written by hand from the documented
formats, **not captured** from that laptop. Replace them with real captures from the target machine
(`nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader,nounits`
and plain `nvidia-smi`).

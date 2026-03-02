# FAQ (English)

## Why does RL work better than SFT here?
At very low parameter counts, RL focuses on correctness reward while SFT also has to model formatting noise.

## Why can results vary between runs?
Seed placement, dataset sampling order, and quantization mode mismatch are common causes.

## Why no code is extracted from outputs?
Check prompt format consistency and ensure extraction logic is reused from `utils.py`.

## How to speed up debugging?
Lower `MAX_SAMPLES`, use fewer validation samples, and start with baseline testing.

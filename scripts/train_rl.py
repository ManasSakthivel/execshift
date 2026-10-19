"""Stage 3: train all PPO members: python scripts/train_rl.py --impact sqrt linear transient.

Thin entry point; the implementation is ``adexec.pipeline.training``."""

from adexec.pipeline.training import main

if __name__ == "__main__":
    raise SystemExit(main())

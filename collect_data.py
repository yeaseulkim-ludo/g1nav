"""Generate G1Nav training episodes using the WBC locomotion policy."""
import argparse
import sys
sys.path.insert(0, "/home/ludo-us/work/g1nav")

from code.data.collector import collect_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    collect_dataset(n_episodes=args.n_episodes, seed=args.seed)

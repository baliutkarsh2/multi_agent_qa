"""
Runs 3 random videos from aitw_videos/ by turning them into goals.
"""
import random
from glob import glob
from pathlib import Path
import agents
from runners.run_example import run

def _goal_from_video(path: Path) -> str:
    return f"Reproduce the UI flow shown in {path.name}"

def main() -> None:
    vids = glob("aitw_videos/*.mp4")
    for vid in random.sample(vids, min(3, len(vids))):
        goal = _goal_from_video(Path(vid))
        print("\n[AITW] Goal:", goal)
        run(goal)

if __name__ == "__main__":
    main() 
"""
Example runner wiring all agents together.
"""
import os
import sys
from pathlib import Path

# Add project root to path and load environment variables first
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables before importing other modules
from core.env_loader import load_env_file
load_env_file()

import agents
import core.logging_config
from core.episode import EpisodeContext
from core.registry import get_agent
from env.android_interface import AndroidDevice
from core.message_bus import subscribe, Message

class App:
    def __init__(self, goal: str, serial: str = None):
        self.goal = goal
        self.device = AndroidDevice(serial)
        self.episode = EpisodeContext(user_goal=goal)
        self.is_done = False
        
        # Initialize agents
        self.planner = get_agent("llm_planner")()
        self.executor = get_agent("llm_executor")(self.device)
        self.verifier = get_agent("llm_verifier")(self.device)
        self.supervisor = get_agent("llm_supervisor")()
        
        # Subscribe to completion message
        subscribe("episode_done", self.on_episode_done)

    def on_episode_done(self, msg: Message):
        self.is_done = True
        print(f"Episode is done. Reason: {msg.payload.get('reason')}")

    def run(self):
        print(f"Goal: {self.goal}")
        
        # Start the first planning step
        ui = self.device.get_ui_tree()
        self.planner.act(self.goal, ui, self.episode)
        
        # Keep the app running until the episode is marked as done
        while not self.is_done:
            pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--goal", required=True)
    parser.add_argument("--serial", help="emulator-5554")
    args = parser.parse_args()
    
    app = App(args.goal, args.serial)
    app.run()

if __name__ == "__main__":
    import argparse
    main() 
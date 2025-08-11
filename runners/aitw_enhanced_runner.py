"""
Enhanced Android in the Wild (AITW) Runner
Integrates with official Google Research dataset and provides comprehensive evaluation.
"""

import os
import sys
import time
import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import asdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables first
from core.env_loader import load_env_file
load_env_file()

import agents
import core.logging_config
from core.episode import EpisodeContext
from core.registry import get_agent
from core.message_bus import subscribe, Message, publish
from env.android_interface import AndroidDevice
from evaluation.aitw_evaluator import AITWEvaluator, VideoTrace, AgentTrace, AITWScore
from evaluation.aitw_data_loader import AITWDataLoader
from core.logging_config import get_logger

log = get_logger("AITW-ENHANCED-RUNNER")

class AITWEnhancedRunner:
    """Enhanced runner that integrates with official AITW dataset."""
    
    def __init__(self, 
                 video_dir: str = "aitw_videos", 
                 dataset_name: str = "google_apps",
                 num_episodes: int = 3,
                 use_official_dataset: bool = True):
        self.video_dir = Path(video_dir)
        self.dataset_name = dataset_name
        self.num_episodes = num_episodes
        self.use_official_dataset = use_official_dataset
        self.evaluator = AITWEvaluator()
        self.results = []
        
        # Initialize data loader for official dataset
        if self.use_official_dataset:
            try:
                self.data_loader = AITWDataLoader(dataset_name)
                log.info(f"Initialized AITW data loader for dataset: {dataset_name}")
            except Exception as e:
                log.warning(f"Could not initialize AITW data loader: {e}")
                self.data_loader = None
                self.use_official_dataset = False
        else:
            self.data_loader = None
        
        # Create video directory if it doesn't exist
        self.video_dir.mkdir(exist_ok=True)
        
        # Initialize device (mock mode for evaluation)
        self.device = AndroidDevice()
        
    def setup_environment(self):
        """Set up the environment for AITW evaluation."""
        print("🔧 Setting up AITW Environment")
        print("=" * 50)
        
        if self.use_official_dataset and self.data_loader:
            print("📊 Using official Google Research AITW dataset")
            
            # Check if TensorFlow is available
            try:
                import tensorflow as tf
                print("✅ TensorFlow available")
            except ImportError:
                print("❌ TensorFlow not available")
                print("   Installing TensorFlow...")
                if self.data_loader.install_dependencies():
                    print("✅ TensorFlow installed successfully")
                else:
                    print("❌ Failed to install TensorFlow")
                    self.use_official_dataset = False
            
            # Check if Google Research repository is available
            if not Path("./google-research").exists():
                print("📥 Cloning Google Research repository...")
                if self.data_loader.setup_google_research_repo():
                    print("✅ Repository cloned successfully")
                else:
                    print("❌ Failed to clone repository")
                    self.use_official_dataset = False
        else:
            print("📱 Using local test videos")
            
        print(f"🎯 Evaluation mode: {'Official Dataset' if self.use_official_dataset else 'Local Videos'}")
        print("✅ Environment setup completed")
    
    def get_evaluation_data(self) -> List[Union[VideoTrace, Dict[str, Any]]]:
        """Get evaluation data from either official dataset or local videos."""
        if self.use_official_dataset and self.data_loader:
            return self._get_official_dataset_episodes()
        else:
            return self._get_local_video_episodes()
    
    def _get_official_dataset_episodes(self) -> List[Dict[str, Any]]:
        """Get episodes from the official AITW dataset."""
        log.info("Loading episodes from official AITW dataset...")

        try:
            episodes = self.data_loader.get_sample_episodes(self.num_episodes)
            log.info(f"Loaded {len(episodes)} episodes from official dataset")

            # Convert to VideoTrace format for compatibility
            video_traces = []
            for episode in episodes:
                trace = VideoTrace(
                    episode_id=episode['episode_id'],
                    app_package=episode['app_package'],
                    steps=episode['steps'],
                    metadata=episode['metadata']
                )
                video_traces.append(trace)

            return video_traces

        except Exception as e:
            log.error(f"Error loading official dataset episodes: {e}")
            log.info("Falling back to demo episodes...")
            return self._get_demo_episodes()
    
    def _get_demo_episodes(self) -> List[VideoTrace]:
        """Get demo episodes for testing purposes."""
        log.info("Loading demo episodes...")
        
        # Create demo episodes that mimic AITW structure
        demo_episodes = [
            {
                'episode_id': 'demo_gmail_001',
                'app_package': 'com.google.android.gm',
                'steps': [
                    {'action': 'launch', 'ui_element': 'gmail_icon', 'timestamp': 0},
                    {'action': 'tap', 'ui_element': 'compose_button', 'timestamp': 1},
                    {'action': 'type', 'ui_element': 'to_field', 'timestamp': 2, 'text': 'test@example.com'},
                    {'action': 'type', 'ui_element': 'subject_field', 'timestamp': 3, 'text': 'Test Email'},
                    {'action': 'type', 'ui_element': 'body_field', 'timestamp': 4, 'text': 'This is a test email.'},
                    {'action': 'tap', 'ui_element': 'send_button', 'timestamp': 5}
                ],
                'metadata': {
                    'episode_id': 'demo_gmail_001',
                    'app_package': 'com.google.android.gm',
                    'actions': ['launch', 'tap', 'type', 'type', 'type', 'tap'],
                    'unique_actions': ['launch', 'tap', 'type']
                }
            },
            {
                'episode_id': 'demo_chrome_001',
                'app_package': 'com.android.chrome',
                'steps': [
                    {'action': 'launch', 'ui_element': 'chrome_icon', 'timestamp': 0},
                    {'action': 'tap', 'ui_element': 'address_bar', 'timestamp': 1},
                    {'action': 'type', 'ui_element': 'address_bar', 'timestamp': 2, 'text': 'google.com'},
                    {'action': 'tap', 'ui_element': 'go_button', 'timestamp': 3},
                    {'action': 'tap', 'ui_element': 'search_box', 'timestamp': 4},
                    {'action': 'type', 'ui_element': 'search_box', 'timestamp': 5, 'text': 'android automation'}
                ],
                'metadata': {
                    'episode_id': 'demo_chrome_001',
                    'app_package': 'com.android.chrome',
                    'actions': ['launch', 'tap', 'type', 'tap', 'tap', 'type'],
                    'unique_actions': ['launch', 'tap', 'type']
                }
            },
            {
                'episode_id': 'demo_maps_001',
                'app_package': 'com.google.android.apps.maps',
                'steps': [
                    {'action': 'launch', 'ui_element': 'maps_icon', 'timestamp': 0},
                    {'action': 'tap', 'ui_element': 'search_bar', 'timestamp': 1},
                    {'action': 'type', 'ui_element': 'search_bar', 'timestamp': 2, 'text': 'Purdue University'},
                    {'action': 'tap', 'ui_element': 'search_result', 'timestamp': 3},
                    {'action': 'tap', 'ui_element': 'directions_button', 'timestamp': 4}
                ],
                'metadata': {
                    'episode_id': 'demo_maps_001',
                    'app_package': 'com.google.android.apps.maps',
                    'actions': ['launch', 'tap', 'type', 'tap', 'tap'],
                    'unique_actions': ['launch', 'tap', 'type']
                }
            }
        ]
        
        # Convert to VideoTrace format for compatibility
        video_traces = []
        for episode in demo_episodes:
            trace = VideoTrace(
                episode_id=episode['episode_id'],
                app_package=episode['app_package'],
                steps=episode['steps'],
                metadata=episode['metadata']
            )
            video_traces.append(trace)
        
        log.info(f"Created {len(video_traces)} demo episodes")
        return video_traces
    
    def _get_local_video_episodes(self) -> List[VideoTrace]:
        """Get episodes from local test videos (fallback)."""
        log.info("Loading episodes from local test videos...")
        
        # This would use the existing local video logic
        # For now, return empty list as placeholder
        return []
    
    def run_official_dataset_evaluation(self) -> List[AITWScore]:
        """Run evaluation using the official AITW dataset."""
        print("🚀 Running Official AITW Dataset Evaluation")
        print("=" * 60)
        
        if not self.data_loader:
            print("❌ AITW data loader not available")
            return []
        
        # Get episodes from official dataset
        episodes = self.data_loader.get_sample_episodes(self.num_episodes)
        
        if not episodes:
            print("❌ No episodes found in dataset")
            return []
        
        print(f"📊 Found {len(episodes)} episodes for evaluation")
        
        results = []
        
        for i, episode in enumerate(episodes):
            print(f"\n📱 Evaluating Episode {i+1}/{len(episodes)}")
            print(f"   ID: {episode['episode_id']}")
            print(f"   App: {episode['app_package']}")
            print(f"   Steps: {len(episode['steps'])}")
            
            # Generate task prompt from episode
            task_prompt = self._generate_task_from_episode(episode)
            print(f"   Task: {task_prompt}")
            
            # Run multi-agent system
            print("   🤖 Running multi-agent system...")
            agent_trace = self._run_multi_agent_system(task_prompt, episode)
            
            # Evaluate results
            print("   📊 Evaluating results...")
            score = self.evaluator.evaluate_episode(episode, agent_trace)
            
            # Log results
            self._log_evaluation_results(episode, task_prompt, agent_trace, score)
            
            results.append(score)
            
            # Add delay between episodes
            if i < len(episodes) - 1:
                time.sleep(2)
        
        # Generate final report
        self._generate_final_report(results, episodes)
        
        return results
    
    def _generate_task_from_episode(self, episode: Dict[str, Any]) -> str:
        """Generate a natural language task description from episode data."""
        app_package = episode.get('app_package', 'the app')
        actions = episode.get('metadata', {}).get('actions', [])
        
        # Simple task generation based on actions
        if 'launch' in str(actions).lower():
            return f"Launch and navigate through {app_package}"
        elif 'tap' in str(actions).lower():
            return f"Interact with {app_package} by tapping various elements"
        elif 'type' in str(actions).lower():
            return f"Use {app_package} to enter text and perform actions"
        else:
            return f"Complete the user interaction flow in {app_package}"
    
    def _run_multi_agent_system(self, task_prompt: str, episode_data: Dict[str, Any]) -> AgentTrace:
        """Run the multi-agent system to reproduce the episode."""
        episode_id = f"aitw_{episode_data['episode_id']}"
        
        # Initialize episode context
        episode_context = EpisodeContext(id=episode_id, user_goal=task_prompt)
        
        # Set up message handlers
        episode_done = False
        execution_reports = []
        
        def on_episode_done(msg: Message):
            nonlocal episode_done
            episode_done = True
            log.info(f"Episode {episode_id} completed: {msg.payload.get('reason', 'Unknown')}")
        
        def on_exec_report(msg: Message):
            if msg.payload.get("episode_id") == episode_id:
                execution_reports.append(msg.payload)
        
        # Subscribe to messages
        subscribe("episode_done", on_episode_done)
        subscribe("exec-report", on_exec_report)
        
        try:
            # Start the episode
            publish(Message("AITW-RUNNER", "episode_start", {
                "episode_id": episode_id,
                "user_goal": task_prompt
            }))
            
            # Wait for completion or timeout
            timeout = 60  # 60 seconds timeout
            start_time = time.time()
            
            while not episode_done and (time.time() - start_time) < timeout:
                time.sleep(0.1)
            
            if not episode_done:
                log.warning(f"Episode {episode_id} timed out after {timeout} seconds")
                publish(Message("AITW-RUNNER", "episode_done", {
                    "reason": "Timeout reached"
                }))
            
        finally:
            # Unsubscribe from messages
            # Note: In a real implementation, you'd want proper cleanup
            
            pass
        
        # Create agent trace
        agent_trace = AgentTrace(
            episode_id=episode_id,
            actions=execution_reports,
            ui_states=[],  # Empty for now
            timestamps=[time.time() - start_time if 'start_time' in locals() else 0],
            task_completion=episode_done,
            success_rate=1.0 if episode_done else 0.0,
            duration=time.time() - start_time if 'start_time' in locals() else 0
        )
        
        return agent_trace
    
    def _log_evaluation_results(self, episode: Dict[str, Any], task_prompt: str, 
                               agent_trace: AgentTrace, score: AITWScore):
        """Log detailed evaluation results."""
        log.info(f"Episode {episode['episode_id']} evaluation completed:")
        log.info(f"  Task: {task_prompt}")
        log.info(f"  Task Completion: {agent_trace.task_completion}")
        log.info(f"  Duration: {agent_trace.duration:.2f}s")
        log.info(f"  Accuracy Score: {score.accuracy_score:.3f}")
        log.info(f"  Robustness Score: {score.robustness_score:.3f}")
        log.info(f"  Generalization Score: {score.generalization_score:.3f}")
    
    def _generate_final_report(self, results: List[AITWScore], episodes: List[Dict[str, Any]]):
        """Generate comprehensive evaluation report."""
        if not results:
            return
        
        # Calculate aggregate scores
        avg_accuracy = sum(r.accuracy_score for r in results) / len(results)
        avg_robustness = sum(r.robustness_score for r in results) / len(results)
        avg_generalization = sum(r.generalization_score for r in results) / len(results)
        avg_duration = sum(r.average_duration for r in results) / len(results)
        success_rate = sum(1 for r in results if r.task_completion_rate > 0) / len(results)
        
        # Create report
        report = {
            "evaluation_summary": {
                "total_episodes": len(episodes),
                "dataset_name": self.dataset_name,
                "evaluation_timestamp": time.time(),
                "system_version": "multi_agent_qa_v1.0_enhanced",
                "data_source": "official_aitw_dataset" if self.use_official_dataset else "local_videos"
            },
            "aggregate_scores": {
                "average_accuracy": avg_accuracy,
                "average_robustness": avg_robustness,
                "average_generalization": avg_generalization,
                "task_completion_rate": success_rate,
                "average_duration": avg_duration
            },
            "individual_results": []
        }
        
        # Add individual results
        for i, (episode, result) in enumerate(zip(episodes, results)):
            episode_result = {
                "episode_id": episode['episode_id'],
                "app_package": episode['app_package'],
                "num_steps": len(episode['steps']),
                "scores": {
                    "accuracy_score": result.accuracy_score,
                    "robustness_score": result.robustness_score,
                    "generalization_score": result.generalization_score,
                    "task_completion_rate": result.task_completion_rate,
                    "duration": result.average_duration
                }
            }
            report["individual_results"].append(episode_result)
        
        # Save report
        report_path = Path("logs") / "aitw_enhanced_evaluation_report.json"
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📊 Evaluation Report saved to: {report_path}")
        
        # Print summary
        print("\n🎯 Evaluation Summary")
        print("=" * 40)
        print(f"Total Episodes: {len(episodes)}")
        print(f"Dataset: {self.dataset_name}")
        print(f"Average Accuracy: {avg_accuracy:.3f}")
        print(f"Average Robustness: {avg_robustness:.3f}")
        print(f"Average Generalization: {avg_generalization:.3f}")
        print(f"Success Rate: {success_rate:.1%}")
        print(f"Average Duration: {avg_duration:.2f}s")
    
    def run_demo(self):
        """Run a demonstration of the enhanced AITW runner."""
        print("🚀 AITW Enhanced Runner Demo")
        print("=" * 50)
        
        # Setup environment
        self.setup_environment()
        
        # Run evaluation
        if self.use_official_dataset:
            results = self.run_official_dataset_evaluation()
        else:
            print("📱 Running local video evaluation...")
            # This would use the existing local evaluation logic
            results = []
        
        print(f"\n✅ Demo completed! Evaluated {len(results)} episodes")
        return results

def main():
    """Main entry point for the enhanced AITW runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced AITW Runner")
    parser.add_argument("--dataset", default="google_apps", 
                       choices=["general", "google_apps", "install", "single", "web_shopping"],
                       help="AITW dataset to use")
    parser.add_argument("--episodes", type=int, default=3, 
                       help="Number of episodes to evaluate")
    parser.add_argument("--local-only", action="store_true",
                       help="Use only local test videos")
    parser.add_argument("--demo", action="store_true",
                       help="Run in demo mode")
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = AITWEnhancedRunner(
        dataset_name=args.dataset,
        num_episodes=args.episodes,
        use_official_dataset=not args.local_only
    )
    
    if args.demo:
        runner.run_demo()
    else:
        # Run evaluation
        if runner.use_official_dataset:
            results = runner.run_official_dataset_evaluation()
        else:
            print("📱 Running local video evaluation...")
            # This would use the existing local evaluation logic
            results = []

if __name__ == "__main__":
    main()

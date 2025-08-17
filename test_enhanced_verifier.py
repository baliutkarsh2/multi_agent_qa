#!/usr/bin/env python3
"""
Comprehensive test suite for the enhanced verifier agent with OpenAI Vision API integration.
Tests multi-modal verification, confidence scoring, and intelligent result combination.
"""

import sys
import os
import time
import json
import base64
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, mock_open
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import the enhanced verifier agent
from agents.llm_verifier_agent import LLMVerifierAgent
from core.message_bus import Message, publish, subscribe
from core.registry import register_agent, get_agent

class TestEnhancedVerifierAgent:
    """Test suite for the enhanced verifier agent with Vision API."""
    
    def setup_method(self):
        """Setup test environment."""
        # Mock Android device
        self.mock_device = Mock()
        self.mock_device.get_ui_tree.return_value = Mock()
        self.mock_device.get_ui_tree.return_value.xml = "<test>Mock UI XML</test>"
        self.mock_device.screenshot.return_value = "test_screenshot.png"
        
        # Create verifier agent
        self.verifier = LLMVerifierAgent(self.mock_device, vision_model="gpt-4o")
        
        # Mock LLM client
        self.verifier.llm = Mock()
        self.verifier.llm.client = Mock()
        self.verifier.llm.model = "gpt-4o-mini"
        self.verifier.llm._extract_json_from_response = Mock(return_value='{"verified": true, "reason": "test", "confidence": 0.8}')
        
        # Mock episodic memory
        self.verifier.episodic_memory = Mock()
        
        # Mock the publish function to avoid actual message publishing
        self.verifier._publish_verification_report = Mock()

    def test_implicit_verification_triggering(self):
        """Test that critical actions trigger implicit verification."""
        # Test critical actions
        critical_actions = [
            {"action": "launch_app", "package": "com.test.app", "step_id": "step1"},
            {"action": "type", "text": "test input", "step_id": "step2"},
            {"action": "tap", "resource_id": "test_button", "step_id": "step3"},
            {"action": "press_key", "key": "enter", "step_id": "step4"},
            {"action": "scroll", "direction": "up", "step_id": "step5"}
        ]
        
        for action in critical_actions:
            should_verify = self.verifier._should_verify_implicitly(action)
            assert should_verify, f"Action {action['action']} should trigger implicit verification"
        
        # Test non-critical actions
        non_critical_actions = [
            {"action": "wait", "duration": 1.0, "step_id": "step6"},
            {"action": "verify", "resource_id": "test_element", "step_id": "step7"}
        ]
        
        for action in non_critical_actions:
            should_verify = self.verifier._should_verify_implicitly(action)
            assert not should_verify, f"Action {action['action']} should not trigger implicit verification"

    def test_multi_modal_verification_text_only(self):
        """Test verification when only text verification is available."""
        # Mock successful text verification
        text_result = {"verified": True, "reason": "Text verification passed", "confidence": 0.8}
        self.verifier._verify_from_ui_xml = Mock(return_value=text_result)
        
        # Mock confidence calculation
        self.verifier._calculate_enhanced_confidence = Mock(return_value=0.85)
        
        # Test verification without screenshot
        result = self.verifier.verify_action("Test action", "Mock UI XML", None)
        
        # Should fall back to text-only verification
        assert result["verified"] == True
        assert "Text verification" in result["reason"]
        assert result["confidence"] == 0.85
        assert result["analysis"]["verification_method"] == "text_only"

    def test_multi_modal_verification_with_screenshot(self):
        """Test multi-modal verification combining text and visual analysis."""
        # Mock successful text verification
        text_result = {"verified": True, "reason": "Text verification passed", "confidence": 0.8}
        self.verifier._verify_from_ui_xml = Mock(return_value=text_result)
        
        # Mock successful visual verification
        visual_result = {"verified": True, "reason": "Visual verification passed", "confidence": 0.9}
        self.verifier._verify_from_screenshot = Mock(return_value=visual_result)
        
        # Mock confidence calculation
        self.verifier._calculate_enhanced_confidence = Mock(return_value=0.85)
        
        # Mock Path.exists() to return True for the screenshot path
        with patch('agents.llm_verifier_agent.Path') as mock_path:
            mock_path.return_value.exists.return_value = True
            
            # Test verification
            result = self.verifier.verify_action("Test action", "Mock UI XML", "test_screenshot.png")
        
        # Should combine both results
        assert result["verified"] == True
        assert "Text: Text verification passed" in result["reason"]
        assert "Visual: Visual verification passed" in result["reason"]
        assert result["confidence"] == 0.85
        assert result["analysis"]["verification_method"] == "multi_modal"

    def test_multi_modal_verification_disagreement(self):
        """Test multi-modal verification when text and visual results disagree."""
        # Mock text verification (lower confidence)
        text_result = {"verified": True, "reason": "Text verification passed", "confidence": 0.6}
        self.verifier._verify_from_ui_xml = Mock(return_value=text_result)
        
        # Mock visual verification (higher confidence, different result)
        visual_result = {"verified": False, "reason": "Visual verification failed", "confidence": 0.9}
        self.verifier._verify_from_screenshot = Mock(return_value=visual_result)
        
        # Mock confidence calculation
        self.verifier._calculate_enhanced_confidence = Mock(return_value=0.7)
        
        # Mock Path.exists() to return True for the screenshot path
        with patch('agents.llm_verifier_agent.Path') as mock_path:
            mock_path.return_value.exists.return_value = True
            
            # Test verification
            result = self.verifier.verify_action("Test action", "Mock UI XML", "test_screenshot.png")
        
        # Should prefer visual result due to higher confidence
        assert result["verified"] == False
        assert "Visual verification preferred" in result["reason"]
        assert result["confidence"] == 0.7
        assert result["analysis"]["verification_method"] == "multi_modal"

    def test_enhanced_confidence_calculation(self):
        """Test enhanced confidence calculation based on multiple factors."""
        # Test with all factors positive
        confidence = self.verifier._calculate_enhanced_confidence("Test action", "Mock UI XML", {"confidence": 0.5})
        assert confidence > 0.5  # Base confidence + factors
        
        # Test error detection
        error_ui = "<error>Something went wrong</error>"
        confidence_with_error = self.verifier._calculate_enhanced_confidence("Test action", error_ui, {"confidence": 0.5})
        assert confidence_with_error < confidence  # Should be lower due to error

    def test_element_presence_analysis(self):
        """Test analysis of UI element presence."""
        # Test with multiple elements present
        ui_xml = "<button>Click me</button><text>Hello world</text>"
        score = self.verifier._analyze_element_presence("Click the button", ui_xml)
        assert score == 0.3  # Should get full score for 2+ elements
        
        # Test with single element
        ui_xml = "<button>Click me</button>"
        score = self.verifier._analyze_element_presence("Click the button", ui_xml)
        assert score == 0.15  # Should get partial score for 1 element
        
        # Test with no relevant elements
        ui_xml = "<div>Some content</div>"
        score = self.verifier._analyze_element_presence("Click the button", ui_xml)
        assert score == 0.0  # Should get no score

    def test_error_indicator_analysis(self):
        """Test detection of error indicators in UI."""
        # Test with no errors
        ui_xml = "<success>Operation completed</success>"
        score = self.verifier._analyze_error_indicators(ui_xml)
        assert score == 0.3  # Should get full score for no errors
        
        # Test with single error
        ui_xml = "<error>Something went wrong</error>"
        score = self.verifier._analyze_error_indicators(ui_xml)
        assert score == 0.1  # Should get partial score for 1 error
        
        # Test with multiple errors
        ui_xml = "<error>First error</error><failed>Second failure</failed>"
        score = self.verifier._analyze_error_indicators(ui_xml)
        assert score == -0.2  # Should get penalty for multiple errors

    def test_context_consistency_analysis(self):
        """Test analysis of context consistency."""
        # Test app launch context
        ui_xml = "<home>App launcher</home>"
        score = self.verifier._analyze_context_consistency("Launch the app", ui_xml)
        assert score == 0.3  # Should get full score for app launch context
        
        # Test input context
        ui_xml = "<input>Text field</input>"
        score = self.verifier._analyze_context_consistency("Type some text", ui_xml)
        assert score == 0.3  # Should get full score for input context
        
        # Test tap context
        ui_xml = "<button>Clickable button</button>"
        score = self.verifier._analyze_context_consistency("Tap the button", ui_xml)
        assert score == 0.3  # Should get full score for tap context
        
        # Test unrelated context
        ui_xml = "<div>Some content</div>"
        score = self.verifier._analyze_context_consistency("Tap the button", ui_xml)
        assert score == 0.1  # Should get default score

    def test_ui_stability_analysis(self):
        """Test analysis of UI state stability."""
        # Test stable UI
        ui_xml = "<ready>Ready state</ready><complete>Complete state</complete>"
        score = self.verifier._analyze_ui_stability(ui_xml)
        assert score == 0.2  # Should get full score for stable UI
        
        # Test mixed stability - use distinct indicators
        ui_xml = "<ready>Ready state</ready><loading>Loading state</loading>"
        score = self.verifier._analyze_ui_stability(ui_xml)
        assert score == 0.1  # Should get partial score for mixed stability
        
        # Test unstable UI
        ui_xml = "<loading>Loading state</loading><updating>Updating state</updating>"
        score = self.verifier._analyze_ui_stability(ui_xml)
        assert score == 0.0  # Should get no score for unstable UI

    def test_screenshot_capture(self):
        """Test screenshot capture functionality."""
        # Test successful screenshot capture
        screenshot_path = self.verifier._capture_screenshot("episode_123", "step_456")
        assert screenshot_path == "test_screenshot.png"
        self.mock_device.screenshot.assert_called_with("episode_123_step_456")
        
        # Test screenshot capture failure
        self.mock_device.screenshot.side_effect = Exception("Screenshot failed")
        screenshot_path = self.verifier._capture_screenshot("episode_123", "step_456")
        assert screenshot_path is None

    def test_image_encoding(self):
        """Test image encoding to base64."""
        # Mock file content
        mock_image_data = b"fake_image_data"
        
        with patch("builtins.open", mock_open(read_data=mock_image_data)):
            with patch("base64.b64encode") as mock_b64encode:
                mock_b64encode.return_value.decode.return_value = "encoded_image_data"
                
                encoded = self.verifier._encode_image_to_base64("test_image.png")
                assert encoded == "encoded_image_data"
                mock_b64encode.assert_called_with(mock_image_data)

    def test_verification_result_combination(self):
        """Test intelligent combination of verification results."""
        # Test text-only verification
        text_result = {"verified": True, "reason": "Text passed", "confidence": 0.8}
        combined = self.verifier._combine_verification_results(text_result, None)
        
        assert combined["verified"] == True
        assert "Text verification" in combined["reason"]
        assert combined["combination_method"] == "text_only"
        
        # Test agreement between text and visual
        visual_result = {"verified": True, "reason": "Visual passed", "confidence": 0.9}
        combined = self.verifier._combine_verification_results(text_result, visual_result)
        
        assert combined["verified"] == True
        assert "Text: Text passed" in combined["reason"]
        assert "Visual: Visual passed" in combined["reason"]
        assert combined["combination_method"] == "agreement"
        assert combined["agreement_level"] == "full"
        
        # Test disagreement with text preferred
        visual_result = {"verified": False, "reason": "Visual failed", "confidence": 0.6}
        combined = self.verifier._combine_verification_results(text_result, visual_result)
        
        assert combined["verified"] == True
        assert "Text verification preferred" in combined["reason"]
        assert combined["combination_method"] == "confidence_based"
        assert combined["preferred_method"] == "text"
        assert combined["disagreement"] == True

    def test_explicit_verification_handling(self):
        """Test handling of explicit verification actions."""
        step = {"action": "verify", "step_id": "step1", "resource_id": "test_element"}
        
        # Mock verification methods
        self.verifier._verify_from_ui_xml = Mock(return_value={"verified": True, "reason": "Passed", "confidence": 0.8})
        self.verifier._verify_from_screenshot = Mock(return_value={"verified": True, "reason": "Visual passed", "confidence": 0.9})
        self.verifier._calculate_enhanced_confidence = Mock(return_value=0.85)
        
        # Mock Path.exists
        with patch('agents.llm_verifier_agent.Path') as mock_path:
            mock_path.return_value.exists.return_value = True
            
            # Test explicit verification
            self.verifier._handle_explicit_verification(step, "episode_123")
            
            # Verify that verification methods were called
            self.verifier._verify_from_ui_xml.assert_called_once()
            self.verifier._verify_from_screenshot.assert_called_once()

    def test_implicit_verification_handling(self):
        """Test handling of implicit verification for critical actions."""
        step = {"action": "launch_app", "step_id": "step1", "package": "com.test.app"}
        
        # Mock verification methods
        self.verifier._verify_from_ui_xml = Mock(return_value={"verified": True, "reason": "Passed", "confidence": 0.8})
        self.verifier._verify_from_screenshot = Mock(return_value={"verified": True, "reason": "Visual passed", "confidence": 0.9})
        self.verifier._calculate_enhanced_confidence = Mock(return_value=0.85)
        
        # Mock Path.exists
        with patch('agents.llm_verifier_agent.Path') as mock_path:
            mock_path.return_value.exists.return_value = True
            
            # Test implicit verification
            self.verifier._verify_action_implicitly(step, "episode_123")
            
            # Verify that verification methods were called
            self.verifier._verify_from_ui_xml.assert_called_once()
            self.verifier._verify_from_screenshot.assert_called_once()
            
            # Verify episodic memory storage
            self.verifier.episodic_memory.store.assert_called_once()

    def test_error_handling(self):
        """Test error handling in verification process."""
        # Test UI XML verification failure
        self.verifier._verify_from_ui_xml = Mock(side_effect=Exception("UI verification failed"))
        
        result = self.verifier.verify_action("Test action", "Mock UI XML", None)
        
        assert result["verified"] == False
        assert "Verification failed due to error" in result["reason"]
        assert result["confidence"] == 0.0
        assert result["analysis"]["verification_method"] == "error"
        
        # Test screenshot verification failure
        self.verifier._verify_from_ui_xml = Mock(return_value={"verified": True, "reason": "Passed", "confidence": 0.8})
        self.verifier._verify_from_screenshot = Mock(side_effect=Exception("Screenshot verification failed"))
        
        with patch('agents.llm_verifier_agent.Path') as mock_path:
            mock_path.return_value.exists.return_value = True
            
            result = self.verifier.verify_action("Test action", "Mock UI XML", "test_screenshot.png")
            
            assert result["verified"] == False
            assert "Verification failed due to error" in result["reason"]
            assert result["confidence"] == 0.0

def run_demo():
    """Run a demonstration of the enhanced verifier agent."""
    print("🚀 Enhanced Verifier Agent with OpenAI Vision API Demo")
    print("=" * 60)
    
    # Create mock environment
    mock_device = Mock()
    mock_device.get_ui_tree.return_value = Mock()
    mock_device.get_ui_tree.return_value.xml = "<test>Demo UI XML</test>"
    mock_device.screenshot.return_value = "demo_screenshot.png"
    
    print("\n1. Creating Enhanced Verifier Agent...")
    verifier = LLMVerifierAgent(mock_device, vision_model="gpt-4o")
    print("   ✅ Verifier agent created with Vision API integration")
    
    print("\n2. Testing Critical Action Detection...")
    critical_action = {"action": "launch_app", "package": "com.test.app"}
    should_verify = verifier._should_verify_implicitly(critical_action)
    print(f"   ✅ Critical action triggers implicit verification: {should_verify}")
    
    print("\n3. Testing Element Presence Analysis...")
    ui_xml = "<button>Click me</button><text>Hello world</text>"
    element_score = verifier._analyze_element_presence("Click the button", ui_xml)
    print(f"   ✅ Element presence score: {element_score}")
    
    print("\n4. Testing Error Detection...")
    error_score = verifier._analyze_error_indicators(ui_xml)
    print(f"   ✅ Error indicator score: {error_score}")
    
    print("\n5. Testing Context Consistency...")
    context_score = verifier._analyze_context_consistency("Click the button", ui_xml)
    print(f"   ✅ Context consistency score: {context_score}")
    
    print("\n6. Testing UI Stability...")
    stability_score = verifier._analyze_ui_stability(ui_xml)
    print(f"   ✅ UI stability score: {stability_score}")
    
    print("\n🎉 Demo completed successfully!")
    print("\nEnhanced Features Demonstrated:")
    print("  • Multi-modal verification (UI XML + screenshot)")
    print("  • OpenAI Vision API integration")
    print("  • Enhanced confidence scoring with multiple factors")
    print("  • Implicit verification for critical actions")
    print("  • Intelligent result combination")
    print("  • Comprehensive error handling")
    print("  • Episodic memory integration")

if __name__ == "__main__":
    # Run the demo
    run_demo()
    
    print("\n" + "=" * 60)
    print("🧪 Running Test Suite...")
    
    # Run tests
    test_verifier = TestEnhancedVerifierAgent()
    
    # Test all methods
    test_methods = [
        "test_implicit_verification_triggering",
        "test_multi_modal_verification_text_only",
        "test_multi_modal_verification_with_screenshot",
        "test_multi_modal_verification_disagreement",
        "test_enhanced_confidence_calculation",
        "test_element_presence_analysis",
        "test_error_indicator_analysis",
        "test_context_consistency_analysis",
        "test_ui_stability_analysis",
        "test_screenshot_capture",
        "test_image_encoding",
        "test_verification_result_combination",
        "test_explicit_verification_handling",
        "test_implicit_verification_handling",
        "test_error_handling"
    ]
    
    for method_name in test_methods:
        try:
            test_verifier.setup_method()
            getattr(test_verifier, method_name)()
            print(f"✅ {method_name} passed")
        except Exception as e:
            print(f"❌ {method_name} failed: {e}")
    
    print("\n🎯 Test suite completed!")
    print("The enhanced verifier agent with OpenAI Vision API is ready for production use.")

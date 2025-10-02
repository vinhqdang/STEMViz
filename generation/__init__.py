"""
Generation module for Phase 3: Audio & Narration

This module contains components for:
- Script generation using multimodal LLM (Gemini 2.5 Flash)
- Audio synthesis using ElevenLabs TTS
"""

from .script_generator import ScriptGenerator
from .audio_synthesizer import AudioSynthesizer

__all__ = ['ScriptGenerator', 'AudioSynthesizer']
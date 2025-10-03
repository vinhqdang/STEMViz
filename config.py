from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path
from typing import Optional


class Settings(BaseSettings):
    # API Keys
    openrouter_api_key: str = Field(..., env="OPENROUTER_API_KEY")
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    google_api_key: str = Field(..., env="GOOGLE_API_KEY")

    # OpenRouter Configuration
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # Model Selection
    reasoning_model: str = "openai/gpt-4o"  # GPT-4o via OpenRouter for reasoning
    multimodal_model: str = "gemini-2.0-flash-exp"  # Gemini 2.0 Flash Experimental for multimodal analysis

    # OpenAI TTS Configuration
    tts_provider: str = "openai"
    tts_model: str = "tts-1"  # or "tts-1-hd" for higher quality
    tts_voice: str = "alloy"  # Options: alloy, echo, fable, onyx, nova, shimmer
    tts_speed: float = 1.0

    # Paths
    output_dir: Path = Path("output")

    @property
    def scenes_dir(self) -> Path:
        return self.output_dir / "scenes"

    @property
    def animations_dir(self) -> Path:
        return self.output_dir / "animations"

    @property
    def audio_dir(self) -> Path:
        return self.output_dir / "audio"

    @property
    def scripts_dir(self) -> Path:
        return self.output_dir / "scripts"

    @property
    def final_dir(self) -> Path:
        return self.output_dir / "final"

    @property
    def analyses_dir(self) -> Path:
        return self.output_dir / "analyses"

    @property
    def rendering_dir(self) -> Path:
        return self.output_dir / "rendering"

    @property
    def generation_dir(self) -> Path:
        return self.output_dir / "generation"

    # Manim Settings
    manim_quality: str = "p"  # Production quality (1080p60)
    manim_background_color: str = "#0f0f0f"
    manim_frame_rate: int = 60
    manim_render_timeout: int = 300  # seconds
    manim_max_retries: int = 3
    manim_max_scene_duration: float = 30.0  # seconds
    manim_total_video_duration_target: float = 120.0  # seconds

    # Animation generation settings
    animation_reasoning_effort: Optional[str] = None
    animation_temperature: float = 0.7
    animation_max_retries_per_scene: int = 3
    animation_enable_simplification: bool = True

    # Script Generation Settings
    script_generation_temperature: float = 0.5
    script_generation_max_retries: int = 3
    script_generation_timeout: int = 180  # seconds

    # Audio Synthesis Settings
    tts_max_retries: int = 3
    tts_timeout: int = 120  # seconds

    # Video Settings
    video_codec: str = "libx264"
    video_preset: str = "medium"
    video_crf: int = 23  # Constant Rate Factor (0-51, lower = higher quality)
    audio_codec: str = "aac"
    audio_bitrate: str = "128k"

    # Subtitle Settings
    subtitle_burn_in: bool = True  # Burn subtitles into video (hard subs)
    subtitle_font_size: int = 24
    subtitle_font_color: str = "white"
    subtitle_background: bool = True
    subtitle_background_opacity: float = 0.5
    subtitle_position: str = "bottom"  # top, center, bottom

    # Video Composition Settings
    video_composition_max_retries: int = 3
    video_composition_timeout: int = 600  # seconds

    # LLM Settings
    llm_temperature: float = 1.0
    llm_max_retries: int = 3
    llm_timeout: int = 120  # seconds

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    def create_directories(self):
        """Create all output directories if they don't exist"""
        for dir_path in [
            self.output_dir,
            self.scenes_dir,
            self.animations_dir,
            self.audio_dir,
            self.scripts_dir,
            self.final_dir,
            self.analyses_dir,
            self.rendering_dir,
            self.generation_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)


# Initialize settings and create directories
settings = Settings()
settings.create_directories()

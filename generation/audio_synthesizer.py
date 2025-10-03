import logging
import time
import tempfile
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime

from openai import OpenAI
from pydantic import BaseModel, Field


class AudioSegment(BaseModel):
    """Single audio segment from TTS synthesis"""
    text: str
    start_time: float  # seconds
    end_time: float    # seconds
    audio_path: Optional[str] = None
    duration: Optional[float] = None
    file_size: Optional[float] = None  # MB


class AudioResult(BaseModel):
    """Result of audio synthesis"""
    success: bool
    audio_path: Optional[str] = None
    audio_segments: List[AudioSegment] = Field(default_factory=list)
    total_duration: Optional[float] = None
    file_size_mb: Optional[float] = None
    error_message: Optional[str] = None

    # Metadata
    generation_time: Optional[float] = None
    voice_settings: Optional[Dict[str, Any]] = None
    model_used: str = "tts-1"


class AudioSynthesizer:
    """
    Audio Synthesizer: Converts timestamped scripts into synchronized speech audio
    using OpenAI Text-to-Speech API.
    """

    def __init__(
        self,
        api_key: str,
        output_dir: Path,
        model: str = "tts-1",
        voice: str = "alloy",
        speed: float = 1.0,
        max_retries: int = 3,
        timeout: int = 120
    ):
        self.api_key = api_key
        self.output_dir = Path(output_dir)
        self.model = model
        self.voice = voice
        self.speed = speed
        self.max_retries = max_retries
        self.timeout = timeout

        # Initialize OpenAI client
        self.client = OpenAI(api_key=api_key)

        # Ensure output directories exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "segments").mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.logger = logging.getLogger(self.__class__.__name__)

    def execute(self, script_path: str, target_duration: Optional[float] = None) -> AudioResult:
        """
        Convert SRT script to synchronized audio

        Args:
            script_path: Path to SRT script file
            target_duration: Optional target duration to match video

        Returns:
            AudioResult with generated audio file and metadata
        """
        start_time = time.time()
        self.logger.info(f"Starting audio synthesis for script: {script_path}")

        try:
            # Validate script file
            script_file = Path(script_path)
            if not script_file.exists():
                raise FileNotFoundError(f"Script file not found: {script_path}")

            # Parse SRT file
            subtitles = self._parse_srt_file(script_file)
            if not subtitles:
                raise ValueError("No subtitles found in SRT file")

            self.logger.info(f"Parsed {len(subtitles)} subtitles from script")

            # Generate audio for each subtitle
            audio_segments = self._generate_audio_segments(subtitles)

            if not audio_segments:
                raise ValueError("Failed to generate audio for any subtitles")

            # Concatenate audio segments
            final_audio_path = self._concatenate_audio_segments(audio_segments, script_file.stem)

            # Validate audio duration
            actual_duration = self._get_audio_duration(final_audio_path)
            if target_duration and actual_duration:
                # Add silence padding if needed
                if actual_duration < target_duration:
                    final_audio_path = self._add_silence_padding(
                        final_audio_path, target_duration - actual_duration
                    )
                    actual_duration = target_duration

            # Calculate file size
            file_size_mb = final_audio_path.stat().st_size / (1024 * 1024)

            generation_time = time.time() - start_time
            self.logger.info(f"Audio synthesis completed in {generation_time:.2f}s")
            self.logger.info(f"Generated audio: {actual_duration:.2f}s, {file_size_mb:.2f}MB")

            return AudioResult(
                success=True,
                audio_path=str(final_audio_path),
                audio_segments=audio_segments,
                total_duration=actual_duration,
                file_size_mb=file_size_mb,
                generation_time=generation_time,
                model_used=self.model,
                voice_settings={
                    "voice": self.voice,
                    "model": self.model,
                    "speed": self.speed
                }
            )

        except Exception as e:
            self.logger.error(f"Audio synthesis failed: {e}")
            return AudioResult(
                success=False,
                error_message=str(e),
                generation_time=time.time() - start_time,
                model_used=self.model
            )

    def _parse_srt_file(self, script_file: Path) -> List[Dict[str, Any]]:
        """Parse SRT file and extract subtitles with timing"""

        subtitles = []

        with open(script_file, 'r', encoding='utf-8') as f:
            content = f.read()

        blocks = content.strip().split('\n\n')

        for block in blocks:
            lines = block.strip().split('\n')
            if len(lines) < 3:
                continue

            try:
                # Parse sequence number
                sequence = int(lines[0].strip())

                # Parse timestamps
                timestamp_line = lines[1].strip()
                if '-->' not in timestamp_line:
                    continue

                start_time_str, end_time_str = [t.strip() for t in timestamp_line.split('-->')]

                # Convert timestamp to seconds
                start_time = self._timestamp_to_seconds(start_time_str)
                end_time = self._timestamp_to_seconds(end_time_str)

                # Parse text (may be multiple lines)
                text = ' '.join(line.strip() for line in lines[2:] if line.strip())

                subtitles.append({
                    'sequence': sequence,
                    'start_time': start_time,
                    'end_time': end_time,
                    'text': text,
                    'duration': end_time - start_time
                })

            except Exception as e:
                self.logger.warning(f"Error parsing subtitle block: {e}")
                continue

        return subtitles

    def _normalize_timestamp(self, timestamp: str) -> str:
        """Normalize malformed timestamps to proper HH:MM:SS,mmm format"""

        timestamp = timestamp.strip()
        timestamp = timestamp.replace('.', ',').replace(':', ',')
        parts = timestamp.split(',')

        if len(parts) == 3:
            minutes, seconds, milliseconds = parts
            hours = "00"
        elif len(parts) == 4:
            hours, minutes, seconds, milliseconds = parts
        else:
            raise ValueError(f"Cannot parse timestamp: {timestamp}")

        hours = int(hours)
        minutes = int(minutes)
        seconds = int(seconds)
        milliseconds = int(milliseconds)

        if minutes >= 60:
            hours += minutes // 60
            minutes = minutes % 60

        if seconds >= 60:
            minutes += seconds // 60
            seconds = seconds % 60

        if milliseconds >= 1000:
            seconds += milliseconds // 1000
            milliseconds = milliseconds % 1000

        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

    def _timestamp_to_seconds(self, timestamp: str) -> float:
        """Convert SRT timestamp (HH:MM:SS,mmm) to seconds"""

        normalized = self._normalize_timestamp(timestamp)
        normalized = normalized.replace(',', ':')
        parts = normalized.split(':')

        if len(parts) != 4:
            raise ValueError(f"Invalid timestamp format: {timestamp}")

        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = int(parts[2])
        milliseconds = int(parts[3])

        return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000

    def _seconds_to_timestamp(self, seconds: float) -> str:
        """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)"""

        hours = int(seconds // 3600)
        remaining = seconds % 3600
        minutes = int(remaining // 60)
        seconds = int(remaining % 60)
        milliseconds = int((remaining % 1) * 1000)

        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

    def _generate_audio_segments(self, subtitles: List[Dict[str, Any]]) -> List[AudioSegment]:
        """Generate audio for each subtitle using OpenAI TTS"""

        audio_segments = []

        for subtitle in subtitles:
            self.logger.info(f"Generating audio for subtitle {subtitle['sequence']}: {subtitle['text'][:50]}...")

            for attempt in range(self.max_retries):
                try:
                    # Generate audio using OpenAI TTS API
                    response = self.client.audio.speech.create(
                        model=self.model,
                        voice=self.voice,
                        input=subtitle['text'],
                        speed=self.speed
                    )

                    # Save audio segment to file
                    segment_path = self._save_audio_segment(response, subtitle['sequence'])

                    # Get actual duration
                    actual_duration = self._get_audio_duration(segment_path)

                    audio_segment = AudioSegment(
                        text=subtitle['text'],
                        start_time=subtitle['start_time'],
                        end_time=subtitle['end_time'],
                        audio_path=str(segment_path),
                        duration=actual_duration,
                        file_size=segment_path.stat().st_size / (1024 * 1024)
                    )

                    audio_segments.append(audio_segment)
                    self.logger.info(f"Generated audio segment: {actual_duration:.2f}s")
                    break

                except Exception as e:
                    self.logger.warning(f"TTS failed for subtitle {subtitle['sequence']} (attempt {attempt + 1}): {e}")
                    if attempt < self.max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                    else:
                        self.logger.error(f"Failed to generate audio for subtitle {subtitle['sequence']}")
                        break

        return audio_segments

    def _save_audio_segment(self, response, sequence: int) -> Path:
        """Save audio segment to file"""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"segment_{sequence:03d}_{timestamp}.mp3"
        filepath = self.output_dir / "segments" / filename

        # Write audio bytes to file
        response.stream_to_file(filepath)

        return filepath

    def _concatenate_audio_segments(self, audio_segments: List[AudioSegment], script_stem: str) -> Path:
        """Concatenate audio segments with proper timing based on SRT timestamps"""

        if not audio_segments:
            raise ValueError("No audio segments to concatenate")

        self.logger.info(f"Concatenating {len(audio_segments)} audio segments with timing")

        audio_segments.sort(key=lambda s: s.start_time)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{script_stem}_audio_{timestamp}.mp3"
        output_path = self.output_dir / output_filename

        silence_files = []
        concat_list = []

        try:
            current_time = 0.0

            for i, segment in enumerate(audio_segments):
                if not segment.audio_path:
                    continue

                gap = segment.start_time - current_time

                if gap > 0.05:
                    silence_path = self.output_dir / f"silence_{timestamp}_{i}.mp3"
                    silence_files.append(silence_path)
                    self._create_silence_file(silence_path, gap)
                    concat_list.append(f"file '{silence_path.resolve()}'\n")
                    self.logger.debug(f"Added {gap:.2f}s silence before segment {i+1}")

                concat_list.append(f"file '{Path(segment.audio_path).resolve()}'\n")
                current_time = segment.start_time + (segment.duration or 0)

            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as concat_file:
                concat_file.writelines(concat_list)
                concat_file_path = concat_file.name

            cmd = [
                "ffmpeg", "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", concat_file_path,
                "-c:a", "libmp3lame",
                "-b:a", "192k",
                "-ar", "44100",
                str(output_path)
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout + 60
            )

            if result.returncode == 0 and output_path.exists():
                self.logger.info(f"Successfully concatenated audio with timing: {output_filename}")
                return output_path
            else:
                self.logger.error(f"FFmpeg concatenation failed: {result.stderr}")
                raise RuntimeError(f"Audio concatenation failed: {result.stderr}")

        except subprocess.TimeoutExpired:
            raise RuntimeError("Audio concatenation timed out")
        finally:
            try:
                Path(concat_file_path).unlink()
            except:
                pass
            for silence_file in silence_files:
                try:
                    silence_file.unlink()
                except:
                    pass

    def _create_silence_file(self, output_path: Path, duration: float):
        """Create a silent audio file of specified duration"""
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", "anullsrc=r=44100:cl=mono",
            "-t", str(duration),
            "-c:a", "libmp3lame",
            "-b:a", "192k",
            "-ar", "44100",
            str(output_path)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to create silence file: {result.stderr}")

    def _add_silence_padding(self, audio_path: Path, padding_duration: float) -> Path:
        """Add silence padding to match target duration"""

        self.logger.info(f"Adding {padding_duration:.2f}s silence padding")

        # Generate new filename
        original_name = audio_path.stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        padded_filename = f"{original_name}_padded_{timestamp}.mp3"
        padded_path = self.output_dir / padded_filename

        # Create silence file
        silence_path = self.output_dir / f"silence_{timestamp}.mp3"
        silence_cmd = [
            "ffmpeg",
            "-y",
            "-f", "lavfi",
            "-i", f"anullsrc=r=44100:cl=mono",
            "-t", str(padding_duration),
            "-c:a", "mp3",
            str(silence_path)
        ]

        subprocess.run(silence_cmd, capture_output=True, check=True)

        # Concatenate original audio with silence
        concat_cmd = [
            "ffmpeg",
            "-y",
            "-i", str(audio_path),
            "-i", str(silence_path),
            "-filter_complex", "[0:a][1:a]concat=n=2:v=0:a=1[out]",
            "-map", "[out]",
            "-c:a", "mp3",
            str(padded_path)
        ]

        subprocess.run(concat_cmd, capture_output=True, check=True)

        # Clean up silence file
        silence_path.unlink()

        self.logger.info(f"Added silence padding: {padded_filename}")
        return padded_path

    def _get_audio_duration(self, audio_path: Path) -> Optional[float]:
        """Get audio duration using ffprobe"""

        try:
            cmd = [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "default=nokey=1:noprint_wrappers=1",
                "-show_entries", "format=duration",
                str(audio_path)
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                return float(result.stdout.strip())
            else:
                self.logger.warning(f"ffprobe failed for {audio_path}: {result.stderr}")
                return None

        except Exception as e:
            self.logger.warning(f"Could not get audio duration for {audio_path}: {e}")
            return None

    def cleanup_temp_files(self):
        """Clean up temporary audio segment files"""

        segments_dir = self.output_dir / "segments"
        if segments_dir.exists():
            for file_path in segments_dir.glob("*.mp3"):
                try:
                    file_path.unlink()
                    self.logger.debug(f"Cleaned up temporary file: {file_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to clean up {file_path}: {e}")

    def get_synthesis_stats(self) -> Dict[str, Any]:
        """Get statistics about audio synthesis performance"""
        return {
            "voice": self.voice,
            "model": self.model,
            "speed": self.speed,
            "max_retries": self.max_retries,
            "timeout": self.timeout,
            "output_dir": str(self.output_dir)
        }

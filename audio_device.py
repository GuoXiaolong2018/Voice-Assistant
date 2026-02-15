"""Audio input utilities built on top of PyAudio."""

from __future__ import annotations

import logging
import math
import struct
import tempfile
import time
from dataclasses import dataclass
from typing import Optional

import pyaudio


LOGGER = logging.getLogger(__name__)


def _rms(frame: bytes) -> float:
    if not frame:
        return 0.0
    count = len(frame) // 2
    if count == 0:
        return 0.0
    fmt = f"<{count}h"
    samples = struct.unpack(fmt, frame)
    mean_square = sum(sample * sample for sample in samples) / count
    return math.sqrt(mean_square)


@dataclass
class AudioConfig:
    sample_rate: int
    frame_length: int
    device_index: Optional[int] = None
    silence_threshold: Optional[float] = 500.0  # If None, will use adaptive threshold
    max_phrase_duration: float = 6.0
    silence_duration: float = 1
    wait_for_silence_timeout: float = 0.3  # Max time to wait for silence before recording
    skip_initial_frames: int = 5  # Number of frames to skip at recording start to avoid echo
    required_silent_frames: int = 2  # Number of consecutive silent frames required
    no_speech_timeout: float = 5.0  # Timeout if no speech detected within this time
    adaptive_silence_threshold: bool = False  # Whether to use adaptive silence threshold
    adaptive_threshold_multiplier: float = 2.0  # Multiplier for adaptive threshold calculation


class AudioStream:
    """Manages continuous microphone streaming and phrase recording."""

    def __init__(self, config: AudioConfig) -> None:
        self._config = config
        self._pa = pyaudio.PyAudio()
        self._stream: Optional[pyaudio.Stream] = None
        self._interrupted = False  # 中断标志，用于及时响应 Ctrl+C
    
    def set_interrupted(self, interrupted: bool = True) -> None:
        """设置中断标志，用于及时退出阻塞操作"""
        self._interrupted = interrupted
        # 如果设置为中断，立即停止流以便快速退出阻塞的 read() 操作
        if interrupted and self._stream is not None:
            try:
                if self._stream.is_active():
                    self._stream.stop_stream()
            except Exception:
                pass

    @staticmethod
    def list_input_devices() -> list[dict[str, str | int]]:
        pa = pyaudio.PyAudio()
        devices = []
        try:
            for idx in range(pa.get_device_count()):
                info = pa.get_device_info_by_index(idx)
                if int(info.get("maxInputChannels", 0)) > 0:
                    devices.append(
                        {
                            "index": idx,
                            "name": info.get("name", "Unknown"),
                            "sample_rate": int(info.get("defaultSampleRate", 0)),
                        }
                    )
        finally:
            pa.terminate()
        return devices

    @staticmethod
    def select_best_device(target_sample_rate: Optional[int] = None) -> Optional[int]:
        """
        Automatically select the best audio input device.
        Prioritizes USB microphones, otherwise uses the default device.
        
        Args:
            target_sample_rate: Preferred sample rate (optional). If provided,
                               will try to select a device that supports it.
        
        Returns:
            Device index or None for default device
        """
        devices = AudioStream.list_input_devices()
        if not devices:
            LOGGER.warning("No input devices found")
            return None
        
        # Log all available devices
        LOGGER.info("Available audio input devices:")
        for dev in devices:
            LOGGER.info("  [%d] %s (rate=%d)", dev["index"], dev["name"], dev["sample_rate"])
        
        # Helper function to check if device supports the target sample rate
        def check_device_compatibility(device_idx: int) -> bool:
            if target_sample_rate is None:
                return True
            try:
                pa = pyaudio.PyAudio()
                try:
                    is_supported = pa.is_format_supported(
                        target_sample_rate,
                        input_device=device_idx,
                        input_channels=1,
                        input_format=pyaudio.paInt16
                    )
                    return is_supported
                finally:
                    pa.terminate()
            except (ValueError, OSError) as e:
                LOGGER.debug("Device [%d] does not support rate %d: %s", 
                           device_idx, target_sample_rate, e)
                return False
        
        # First priority: USB microphones
        usb_devices = [
            dev for dev in devices 
            if "usb" in str(dev["name"]).lower()
        ]
        
        # Try to find a compatible USB device first
        if usb_devices:
            for dev in usb_devices:
                if check_device_compatibility(dev["index"]):
                    LOGGER.info("Selected compatible USB microphone: [%d] %s", dev["index"], dev["name"])
                    return dev["index"]
            
            # USB device found but not compatible - log and continue to try other devices
            LOGGER.warning("Found USB microphone(s) but none support sample rate %d Hz", target_sample_rate)
            LOGGER.info("Searching for alternative compatible devices...")
        
        # Try external microphones (excluding built-in/internal)
        # Prefer pipewire/default devices as they often route to actual hardware
        external_devices = [
            dev for dev in devices
            if not any(keyword in str(dev["name"]).lower() 
                      for keyword in ["built-in", "internal", "内置", "内部", "usb"])
        ]
        
        # Sort external devices: prefer pipewire and default devices
        def device_priority(dev):
            name_lower = str(dev["name"]).lower()
            if "pipewire" in name_lower:
                return 0  # Highest priority
            elif "default" in name_lower:
                return 1
            elif "sysdefault" in name_lower:
                return 3  # Lower priority (often problematic)
            else:
                return 2
        
        external_devices.sort(key=device_priority)
        
        if external_devices:
            for dev in external_devices:
                if check_device_compatibility(dev["index"]):
                    LOGGER.info("Selected compatible external microphone: [%d] %s", dev["index"], dev["name"])
                    return dev["index"]
        
        # Try any device that supports the target sample rate
        if target_sample_rate:
            for dev in devices:
                if check_device_compatibility(dev["index"]):
                    LOGGER.info("Selected compatible microphone: [%d] %s", dev["index"], dev["name"])
                    return dev["index"]
        
        # Last resort: use the first USB device even if incompatible
        if usb_devices:
            selected = usb_devices[0]
            LOGGER.warning("No compatible devices found. Using USB microphone [%d] %s anyway", 
                         selected["index"], selected["name"])
            return selected["index"]
        
        # Absolute fallback: use the first available device
        selected = devices[0]
        LOGGER.info("Using fallback microphone: [%d] %s", selected["index"], selected["name"])
        return selected["index"]

    def open(self) -> None:
        if self._stream is not None:
            return
        config = self._config
        
        # Log device information
        if config.device_index is not None:
            try:
                device_info = self._pa.get_device_info_by_index(config.device_index)
                LOGGER.info(
                    "Opening audio stream (device=[%d] %s, rate=%s, frame=%s)",
                    config.device_index,
                    device_info.get("name", "Unknown"),
                    config.sample_rate,
                    config.frame_length,
                )
            except Exception as e:
                LOGGER.warning("Could not get device info: %s", e)
                LOGGER.info(
                    "Opening audio stream (device=%s, rate=%s, frame=%s)",
                    config.device_index,
                    config.sample_rate,
                    config.frame_length,
                )
        else:
            LOGGER.info(
                "Opening audio stream (device=default, rate=%s, frame=%s)",
                config.sample_rate,
                config.frame_length,
            )
        
        self._stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=config.sample_rate,
            input=True,
            frames_per_buffer=config.frame_length,
            input_device_index=config.device_index,
        )
        
        # Test initial audio levels to detect if microphone is receiving audio
        LOGGER.debug("Testing initial audio levels...")
        test_frames = []
        for _ in range(5):
            try:
                frame = self._stream.read(config.frame_length, exception_on_overflow=False)
                test_frames.append(frame)
            except Exception as e:
                LOGGER.warning("Error reading test frames: %s", e)
                break
        
        if test_frames:
            levels = [_rms(f) for f in test_frames]
            max_test_level = max(levels)
            avg_test_level = sum(levels) / len(levels)
            LOGGER.info("Initial audio test: avg=%.1f, max=%.1f", avg_test_level, max_test_level)
            
            if max_test_level < 5.0:
                LOGGER.warning(
                    "⚠️  Very low audio levels detected (max=%.1f). "
                    "Microphone may not be receiving audio. "
                    "Please check:\n"
                    "  1. System audio settings - ensure USB microphone is selected as input\n"
                    "  2. Run './check_audio_setup.sh' for detailed diagnostics\n"
                    "  3. Try using pavucontrol to set USB mic as default source",
                    max_test_level
                )

    def read_frame(self) -> bytes:
        """
        读取一帧音频数据
        
        注意：这是一个阻塞调用，会等待音频数据可用。
        如果设置了 _interrupted 标志或流已停止，会立即返回空数据。
        """
        if self._stream is None:
            raise RuntimeError("Audio stream has not been opened")
        
        # 如果已经中断，检查流状态并立即返回
        if self._interrupted:
            # 尝试停止流以便 read() 能立即返回
            try:
                if self._stream.is_active():
                    self._stream.stop_stream()
            except Exception:
                pass
            # 返回静音帧以便快速退出
            return b'\x00' * (self._config.frame_length * 2)
        
        # PyAudio 的 read() 会阻塞直到数据可用
        # 如果流被停止，read() 可能会抛出异常或返回不完整数据
        try:
            # 检查流是否仍然活跃
            if not self._stream.is_active():
                return b'\x00' * (self._config.frame_length * 2)
            return self._stream.read(self._config.frame_length, exception_on_overflow=False)
        except (OSError, SystemError) as e:
            # 流可能已被关闭或停止
            if self._interrupted:
                LOGGER.debug("Stream read interrupted: %s", e)
                return b'\x00' * (self._config.frame_length * 2)
            raise

    def calibrate_silence_threshold(self, calibration_duration: float = 0.5) -> float:
        """
        Calibrate silence threshold by sampling ambient noise.
        
        This method should be called after TTS finishes speaking (e.g., after saying "我在")
        to measure the ambient noise level and compute an adaptive silence threshold.
        
        Args:
            calibration_duration: Duration in seconds to sample ambient noise (default 0.5s)
            
        Returns:
            Computed silence threshold based on ambient noise
        """
        if self._stream is None:
            raise RuntimeError("Audio stream has not been opened")
        
        config = self._config
        frame_duration = config.frame_length / config.sample_rate
        num_frames = int(calibration_duration / frame_duration)
        
        LOGGER.info("Calibrating silence threshold over %.2fs (%d frames)...", 
                   calibration_duration, num_frames)
        
        # Collect ambient noise samples
        noise_levels = []
        for _ in range(num_frames):
            if self._interrupted:
                LOGGER.info("Interrupted during silence threshold calibration")
                break
            # 在读取帧之前再次检查中断
            if self._interrupted:
                break
            frame = self.read_frame()
            level = _rms(frame)
            noise_levels.append(level)
        
        if not noise_levels:
            LOGGER.warning("No noise samples collected, using default threshold")
            return config.silence_threshold or 500.0
        
        # Calculate statistics
        avg_noise = sum(noise_levels) / len(noise_levels)
        max_noise = max(noise_levels)
        
        # Use average noise level with a multiplier to avoid false positives
        # The multiplier ensures speech is significantly louder than ambient noise
        adaptive_threshold = avg_noise * config.adaptive_threshold_multiplier
        
        LOGGER.info("Ambient noise: avg=%.1f, max=%.1f, computed threshold=%.1f", 
                   avg_noise, max_noise, adaptive_threshold)
        
        # Update the config to use the adaptive threshold
        self._config.silence_threshold = adaptive_threshold
        
        return adaptive_threshold

    def wait_for_silence(self, max_wait: Optional[float] = None) -> bool:
        """
        Wait for the environment to become silent before recording.
        This helps avoid recording the tail end of TTS playback.
        
        Args:
            max_wait: Maximum time to wait for silence (seconds).
                     If None, uses configured value.
            
        Returns:
            True if silence was achieved, False if timeout
        """
        if self._stream is None:
            raise RuntimeError("Audio stream has not been opened")
        
        if max_wait is None:
            max_wait = self._config.wait_for_silence_timeout
        
        start_time = time.time()
        consecutive_silent_frames = 0
        required_silent_frames = self._config.required_silent_frames
        
        LOGGER.info("Waiting for silence (max %.1fs, threshold=%.1f, required_frames=%d)...", 
                   max_wait, self._config.silence_threshold, required_silent_frames)
        
        while time.time() - start_time < max_wait:
            if self._interrupted:
                LOGGER.info("Interrupted while waiting for silence")
                return False
            # 在读取帧之前再次检查中断
            if self._interrupted:
                return False
            frame = self.read_frame()
            level = _rms(frame)
            
            if level < self._config.silence_threshold:
                consecutive_silent_frames += 1
                if consecutive_silent_frames >= required_silent_frames:
                    LOGGER.info("Silence achieved after %.2fs", time.time() - start_time)
                    return True
            else:
                consecutive_silent_frames = 0
                LOGGER.debug("Audio level: %.1f (above threshold)", level)
        
        LOGGER.warning("Silence wait timeout after %.2fs - proceeding anyway", time.time() - start_time)
        return True  # Proceed anyway after timeout

    def record_phrase(self, wait_for_silence: bool = True) -> Optional[str]:
        """
        Record a phrase from the microphone.
        
        Args:
            wait_for_silence: If True, wait for silence before starting recording
                            to avoid capturing TTS playback echoes
        
        Returns:
            Path to the recorded audio file, or None if no audio was captured.
            Returns "TIMEOUT" (string) if no speech detected within no_speech_timeout.
        """
        if self._stream is None:
            raise RuntimeError("Audio stream has not been opened")

        config = self._config
        
        # Wait for silence to avoid recording TTS echoes
        if wait_for_silence:
            if self._interrupted:
                return None
            self.wait_for_silence()
            if self._interrupted:
                return None
        
        # Skip initial frames to avoid any residual echo
        if config.skip_initial_frames > 0:
            LOGGER.debug("Skipping first %d frames to avoid echo...", config.skip_initial_frames)
            for i in range(config.skip_initial_frames):
                if self._interrupted:
                    return None
                self.read_frame()  # Discard
        
        frames: list[bytes] = []
        silence_start: Optional[float] = None
        start_time = time.time()
        speech_detected = False  # Track if any speech has been detected

        LOGGER.info("Recording phrase222 (max %.1fs, no-speech timeout %.1fs)…", 
                   config.max_phrase_duration, config.no_speech_timeout)

        frame_count = 0
        max_level_seen = 0.0
        min_level_seen = float('inf')
        zero_level_count = 0

        while time.time() - start_time < config.max_phrase_duration:
            # 检查中断标志
            if self._interrupted:
                LOGGER.info("Interrupted while recording phrase")
                return None
            # Check if we've exceeded the no-speech timeout
            if not speech_detected and (time.time() - start_time) >= config.no_speech_timeout:
                LOGGER.info("No speech detected within %.1fs - timeout", config.no_speech_timeout)
                LOGGER.warning("Audio levels during timeout: min=%.1f, max=%.1f, zero_frames=%d", 
                             min_level_seen, max_level_seen, zero_level_count)
                if max_level_seen < 10.0:
                    LOGGER.error("CRITICAL: Maximum audio level (%.1f) is extremely low! "
                               "Microphone may not be receiving audio. Check device selection and PipeWire configuration.", 
                               max_level_seen)
                return "TIMEOUT"
            
            # 在读取帧之前检查中断（read_frame 可能阻塞一个帧的时间）
            if self._interrupted:
                LOGGER.info("Interrupted while recording phrase")
                return None
            
            frame = self.read_frame()
            frames.append(frame)
            frame_count += 1

            level = _rms(frame)
            max_level_seen = max(max_level_seen, level)
            min_level_seen = min(min_level_seen, level)
            if level < 1.0:
                zero_level_count += 1
            
            # Log audio levels periodically for debugging
            if frame_count % 1 == 0:  # Every 20 frames (~1.6 seconds at 16kHz)
                LOGGER.debug("Audio level: %.1f (threshold=%.1f, max_seen=%.1f)", 
                           level, config.silence_threshold, max_level_seen)
                LOGGER.info("Audio level: %.1f (threshold=%.1f, max_seen=%.1f)", 
                           level, config.silence_threshold, max_level_seen)
            
            if level < config.silence_threshold:
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start >= config.silence_duration:
                    # Only end recording if we've detected speech before
                    if speech_detected:
                        LOGGER.debug("Silence detected after speech, ending recording")
                        break
            else:
                speech_detected = True  # Mark that we've detected speech
                silence_start = None

        if not frames or not speech_detected:
            LOGGER.warning("No audio captured for phrase")
            LOGGER.warning("Audio levels: min=%.1f, max=%.1f, zero_frames=%d/%d", 
                         min_level_seen, max_level_seen, zero_level_count, frame_count)
            if max_level_seen < 10.0:
                LOGGER.error("CRITICAL: Maximum audio level (%.1f) is extremely low! "
                           "Microphone may not be receiving audio. Possible causes:\n"
                           "  1. Wrong device selected (check PipeWire default source)\n"
                           "  2. USB microphone not routed through PipeWire\n"
                           "  3. Microphone muted or volume too low\n"
                           "  4. Device permissions issue", max_level_seen)
            return None

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp_path = tmp.name
        tmp.close()

        import wave

        with wave.open(tmp_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(self._pa.get_sample_size(pyaudio.paInt16))
            wf.setframerate(config.sample_rate)
            wf.writeframes(b"".join(frames))

        LOGGER.info("Saved recorded phrase to %s", tmp_path)
        return tmp_path

    def close(self) -> None:
        if self._stream is not None:
            LOGGER.info("Closing audio stream")
            self._interrupted = True  # 设置中断标志以快速退出阻塞操作
            try:
                # 立即停止流，这会让正在阻塞的 read() 操作立即返回
                if self._stream.is_active():
                    self._stream.stop_stream()
            except Exception as e:
                LOGGER.debug("Error stopping stream: %s", e)
            try:
                self._stream.close()
            except Exception as e:
                LOGGER.debug("Error closing stream: %s", e)
            self._stream = None

    def terminate(self) -> None:
        self.close()
        if self._pa is not None:
            self._pa.terminate()

    def __enter__(self) -> "AudioStream":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.terminate()




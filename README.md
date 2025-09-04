# ffmpeg_python_assemble_videos


Auto Video Assembler
--------------------

Windows 10/11 compatible Python script that automates assembling a video with this sequence:

    [Intro (optional)] + Main + [all B-roll] + [Outro (optional)]

Features
- Uses FFmpeg/FFprobe for speed and quality.
- Simple Tkinter UI with checkboxes to enable/disable Intro/Outro/Logo/Music.
- Logo overlay (default top-left; configurable exact x/y or presets).
- Background music with automatic ducking (sidechain compression) under dialog.
- Basic crossfade transitions between segments (configurable duration).
- Exports timestamped MP4 to ./4_Export.
- Watches input folders for new files (optional; requires `watchdog`).
- Configurable via UI and persisted to `config.json` (no code edits needed for paths).
- Optional Google Drive upload using `rclone` (if installed and configured) or Drive API (commented template).

Folders (relative to this script by default)
- 1_Main/              # main talking/review video (pick first video file by name)
- 2_B-Roll/            # all b‑roll clips (sorted by file name)
- 3_Assets/            # contains intro.mp4, outro.mp4, logo.png, music.mp3 (default names; customizable)
- 4_Export/            # output

Dependencies
- Python 3.8+
- FFmpeg (ffmpeg + ffprobe available on PATH)
- Optional: watchdog (for watch mode) → pip install watchdog
- Optional: rclone (for Google Drive upload) → configure a remote named "gdrive"

Author: Naseem Amjad
Email: naseem@technologist.com

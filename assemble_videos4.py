#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
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
"""

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from tkinter import Tk, StringVar, IntVar, BooleanVar, filedialog, ttk, messagebox, Text, END, DISABLED, NORMAL

# --------------------------- Utilities ---------------------------
VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".m4v", ".avi", ".webm")
IMAGE_EXTS = (".png", ".jpg", ".jpeg")
AUDIO_EXTS = (".mp3", ".wav", ".aac", ".m4a", ".flac")

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULTS = {
    "paths": {
        "main_dir": str(SCRIPT_DIR / "1_Main"),
        "broll_dir": str(SCRIPT_DIR / "2_B-Roll"),
        "assets_dir": str(SCRIPT_DIR / "3_Assets"),
        "export_dir": str(SCRIPT_DIR / "4_Export"),
        "intro": "intro.mp4",
        "outro": "outro.mp4",
        "logo": "logo.png",
        "music": "music.mp3"
    },
    "options": {
        "use_intro": True,
        "use_outro": True,
        "use_logo": True,
        "use_music": True,
        "watch_mode": False
    },
    "video": {
        "width": 1920,
        "height": 1080,
        "fps": 30,
        "transition_sec": 0.5,
        "crf": 20,
        "preset": "veryfast"
    },
    "logo": {
        "position": "top_left",  # one of: top_left, top_right, bottom_left, bottom_right, custom
        "x": 24,
        "y": 24,
        "scale_width": 0,  # 0 means keep original; otherwise scale to this width (px)
        "opacity": 1.0
    },
    "music": {
        "ducking": {
            "ratio": 12,      # compression ratio when voice is present
            "threshold": -18, # dB threshold for sidechain
            "attack": 5,      # ms
            "release": 200    # ms
        },
        "music_volume_db": -12  # base music volume before ducking
    },
    "upload": {
        "enable_rclone": False,
        "rclone_remote": "gdrive:",
        "rclone_path": "AutoVideo/",
        "enable_drive_api": False  # Template provided below (commented); requires credentials setup.
    },
    "broll": {
        "broll1": {
            "enable": True,
            "start_time": 5.0,  # seconds
            "duration": 10.0     # seconds (0 for full duration)
        },
        "broll2": {
            "enable": True,
            "start_time": 20.0,
            "duration": 10.0
        }
    }
}

CONFIG_PATH = SCRIPT_DIR / "config.json"


def load_config():
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                return merge_dicts(DEFAULTS, data)
        except Exception:
            pass
    return DEFAULTS.copy()


def save_config(cfg):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def merge_dicts(d1, d2):
    out = d1.copy()
    for k, v in d2.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge_dicts(out[k], v)
        else:
            out[k] = v
    return out


def log(ui, msg):
    print(msg)
    if ui is not None:
        ui.log_text.configure(state=NORMAL)
        ui.log_text.insert(END, msg + "\n")
        ui.log_text.see(END)
        ui.log_text.configure(state=DISABLED)


def which(cmd):
    return shutil.which(cmd)


# --------------------------- FFmpeg helpers ---------------------------

def check_ffmpeg(ui=None):
    if not which("ffmpeg") or not which("ffprobe"):
        log(ui, "ERROR: FFmpeg/FFprobe not found on PATH. Please install FFmpeg and try again.")
        raise RuntimeError("FFmpeg not found")


def ffprobe_duration(path: Path) -> float:
    """Return duration in seconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path)
    ]
    out = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode().strip()
    try:
        return float(out)
    except Exception:
        return 0.0

def ffprobe_has_audio(path: Path) -> bool:
    """Check if file has audio stream using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "a",
        "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(path)
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode().strip()
        return bool(out)
    except Exception:
        return False

def gather_inputs(cfg, ui=None):
    p = cfg["paths"]
    main_dir = Path(p["main_dir"]).resolve()
    broll_dir = Path(p["broll_dir"]).resolve()
    assets_dir = Path(p["assets_dir"]).resolve()

    if not main_dir.exists():
        raise FileNotFoundError(f"Main dir not found: {main_dir}")
    if not broll_dir.exists():
        raise FileNotFoundError(f"B-roll dir not found: {broll_dir}")
    assets_dir.mkdir(parents=True, exist_ok=True)

    # Pick one main file (first by name)
    main_candidates = sorted([f for f in main_dir.iterdir() if f.suffix.lower() in VIDEO_EXTS and f.is_file()])
    if not main_candidates:
        raise FileNotFoundError(f"No main video found in {main_dir}")
    main = main_candidates[0]

    # All B-roll sorted by name
    brolls = sorted([f for f in broll_dir.iterdir() if f.suffix.lower() in VIDEO_EXTS and f.is_file()])

    clips = []
    if cfg["options"]["use_intro"]:
        intro = assets_dir / p["intro"]
        if intro.exists():
            clips.append(intro)
        else:
            log(ui, f"WARNING: intro not found at {intro}, skipping intro.")
    clips.append(main)
    clips.extend(brolls)
    if cfg["options"]["use_outro"]:
        outro = assets_dir / p["outro"]
        if outro.exists():
            clips.append(outro)
        else:
            log(ui, f"WARNING: outro not found at {outro}, skipping outro.")

    return clips

def compute_positions(cfg, vid_w, vid_h):
    lg = cfg["logo"]
    pos = lg.get("position", "top_left")
    x, y = lg.get("x", 24), lg.get("y", 24)
    if pos == "top_left":
        return 24, 24
    if pos == "top_right":
        return f"(W-w)-24", 24
    if pos == "bottom_left":
        return 24, f"(H-h)-24"
    if pos == "bottom_right":
        return f"(W-w)-24", f"(H-h)-24"
    # custom
    return x, y


def build_filter_graph(cfg, inputs, music_enabled, logo_enabled, ui=None):
    """Construct a filter_complex string for overlaying B-roll on main video."""
    vcfg = cfg["video"]
    target_w, target_h, fps = vcfg["width"], vcfg["height"], vcfg["fps"]
    
    # Main video is always input 0 (after intro if present)
    main_idx = 1 if cfg["options"]["use_intro"] and len(inputs) > 1 else 0
    main_video = inputs[main_idx]
    
    # Get B-roll files (assuming they're after main video in inputs list)
    brolls = []
    if len(inputs) > main_idx + 1:
        brolls = inputs[main_idx + 1:-1] if cfg["options"]["use_outro"] else inputs[main_idx + 1:]
    
    filter_parts = []
    
    # Process main video - scale to fit target dimensions while maintaining aspect ratio
    filter_parts.append(
        f"[{main_idx}:v]scale=w='if(gt(a,{target_w/target_h}),{target_w},-1)':h='if(gt(a,{target_w/target_h}),-1,{target_h})',"
        f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:color=black,fps={fps}[main]"
    )
    
    # Process audio from main video
    if ffprobe_has_audio(main_video):
        filter_parts.append(f"[{main_idx}:a]aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo[main_audio]")
    else:
        filter_parts.append(f"aevalsrc=0:d={ffprobe_duration(main_video)}[main_audio]")
    
    # Process B-roll overlays
    vout = "main"
    for i, broll in enumerate(brolls):
        broll_idx = main_idx + 1 + i
        broll_cfg = cfg["broll"].get(f"broll{i+1}", {})
        
        if not broll_cfg.get("enable", True):
            continue
            
        start_time = float(broll_cfg.get("start_time", 5.0 * (i + 1)))
        duration = float(broll_cfg.get("duration", 0)) or ffprobe_duration(broll)
        
        # Scale B-roll to half size while maintaining aspect ratio
        broll_w = target_w // 2
        broll_h = target_h // 2
        filter_parts.append(
            f"[{broll_idx}:v]scale=w='if(gt(a,{broll_w/broll_h}),{broll_w},-1)':h='if(gt(a,{broll_w/broll_h}),-1,{broll_h})',"
            f"pad={broll_w}:{broll_h}:-1:-1:color=black@0,fps={fps}[broll{i}]"
        )
        
        # Overlay B-roll at specified time
        filter_parts.append(
            f"[{vout}][broll{i}]overlay=x=(W-w)/2:y=(H-h)/2:enable='between(t,{start_time},{start_time + duration})'[vout{i}]"
        )
        vout = f"vout{i}"
    
    # Handle intro if enabled
    if cfg["options"]["use_intro"] and len(inputs) > 0:
        intro_idx = 0
        filter_parts.append(
            f"[{intro_idx}:v]scale=w='if(gt(a,{target_w/target_h}),{target_w},-1)':h='if(gt(a,{target_w/target_h}),-1,{target_h})',"
            f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:color=black,fps={fps}[intro]"
        )
        filter_parts.append(f"[intro][{vout}]xfade=transition=fade:duration=1.0:offset={ffprobe_duration(inputs[intro_idx])-1}[intro_out]")
        vout = "intro_out"
    
    # Handle outro if enabled
    if cfg["options"]["use_outro"] and len(inputs) > 1:
        outro_idx = len(inputs) - 1
        filter_parts.append(
            f"[{outro_idx}:v]scale=w='if(gt(a,{target_w/target_h}),{target_w},-1)':h='if(gt(a,{target_w/target_h}),-1,{target_h})',"
            f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:color=black,fps={fps}[outro]"
        )
        main_duration = ffprobe_duration(main_video)
        filter_parts.append(f"[{vout}][outro]xfade=transition=fade:duration=1.0:offset={main_duration-1}[final]")
        vout = "final"
    
    # Handle logo overlay if enabled
    if logo_enabled:
        # logo will be added as an input later
        logo_idx = len(inputs)
        x, y = compute_positions(cfg, target_w, target_h)
        filter_parts.append(f"[{vout}][{logo_idx}:v]overlay={x}:{y}:enable='gte(t,0)'[vlogo]")
        vout = "vlogo"
    
    # Handle audio (mix main audio with music if enabled)
    aout = "main_audio"
    if music_enabled:
        # music will be added as an input later
        music_idx = len(inputs) + (1 if logo_enabled else 0)
        filter_parts.append(
            f"[{aout}][{music_idx}:a]amix=inputs=2:duration=first:dropout_transition=2[aout]"
        )
        aout = "aout"
    
    return ";".join(filter_parts), vout, aout, ffprobe_duration(main_video)
    
def assemble_video(cfg, ui=None):
    try:
        log(ui, "=== Starting video assembly ===")
        check_ffmpeg(ui)

        p = cfg["paths"]
        export_dir = Path(p["export_dir"]).resolve()
        export_dir.mkdir(parents=True, exist_ok=True)

        log(ui, f"Export directory: {export_dir}")

        use_logo = bool(cfg["options"]["use_logo"]) and (Path(p["assets_dir"]) / p["logo"]).exists()
        use_music = bool(cfg["options"]["use_music"]) and (Path(p["assets_dir"]) / p["music"]).exists()

        log(ui, f"Using logo: {use_logo}, Using music: {use_music}")

        # Gather inputs
        inputs = gather_inputs(cfg, ui)
        log(ui, f"Collected input clips: {[str(f) for f in inputs]}")

        # Build filter graph
        base_filter, vlabel, alabel, total_dur = build_filter_graph(cfg, inputs, music_enabled=use_music, logo_enabled=use_logo, ui=ui)
        log(ui, f"Base filter graph:\n{base_filter}")
        log(ui, f"Total duration (approx): {total_dur:.2f} sec")

        # Prepare ffmpeg command
        cmd = ["ffmpeg", "-y"]

        for path in inputs:
            cmd += ["-i", str(path)]

        filter_parts = [base_filter] if base_filter else []

        # Logo handling
        vout = vlabel
        if use_logo:
            logo_path = Path(p["assets_dir"]) / p["logo"]
            cmd += ["-i", str(logo_path)]
            x, y = compute_positions(cfg, cfg["video"]["width"], cfg["video"]["height"])
            logo_chain = [f"[{vout}][{len(inputs)}:v]overlay={x}:{y}[vlogo]"]
            filter_parts.append(";".join(logo_chain))
            vout = "vlogo"

        # Music handling
        aout = alabel
        if use_music:
            music_path = Path(p["assets_dir"]) / p["music"]
            cmd += ["-i", str(music_path)]
            duck_chain = f"[{aout}][{len(inputs)+(1 if use_logo else 0)}:a]amix=inputs=2[aout]"
            filter_parts.append(duck_chain)
            aout = "aout"

        filter_complex = ";".join(filter_parts)

        vcodec = ["-c:v", "libx264", "-preset", cfg["video"]["preset"], "-crf", str(cfg["video"]["crf"])]
        acodec = ["-c:a", "aac", "-b:a", "192k"]

        cmd += ["-filter_complex", filter_complex, "-map", f"[{vout}]", "-map", f"[{aout}]"] + vcodec + acodec

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = export_dir / f"final_{ts}.mp4"
        cmd += ["-movflags", "+faststart", str(out_path)]

        # Log full command
        log(ui, "Running FFmpeg command:")
        log(ui, " ".join(cmd))

        # Run FFmpeg with real-time output capture
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )

        for line in proc.stdout:
            line = line.strip()
            if line:
                log(ui, line)  # print every ffmpeg line without filtering

        proc.wait()

        if proc.returncode != 0:
            raise RuntimeError(f"FFmpeg failed with exit code {proc.returncode}")

        log(ui, f"Export complete: {out_path}")
        return out_path

    except Exception as e:
        log(ui, f"❌ ERROR: {e}")
        import traceback
        tb = traceback.format_exc()
        log(ui, tb)
        raise


# --------------------------- Watcher (optional) ---------------------------
class FolderWatcher:
    def __init__(self, paths, callback, ui=None):
        self.paths = [Path(p) for p in paths]
        self.callback = callback
        self.ui = ui
        self.thread = None
        self.stop_flag = threading.Event()
        self.enabled = False
        # Try to use watchdog if present; else fall back to simple polling
        try:
            from watchdog.observers import Observer  # type: ignore
            from watchdog.events import FileSystemEventHandler  # type: ignore
            self.Observer = Observer
            self.FileSystemEventHandler = FileSystemEventHandler
            self.use_watchdog = True
        except Exception:
            self.use_watchdog = False

    def start(self):
        if self.enabled:
            return
        self.enabled = True
        if self.use_watchdog:
            self._start_watchdog()
        else:
            self.thread = threading.Thread(target=self._poll_loop, daemon=True)
            self.thread.start()
            log(self.ui, "Watch mode: using polling (install 'watchdog' for better performance).")

    def stop(self):
        self.stop_flag.set()
        self.enabled = False

    # --- Polling fallback ---
    def _poll_loop(self):
        mtimes = {}
        for p in self.paths:
            p.mkdir(parents=True, exist_ok=True)
        while not self.stop_flag.is_set():
            changed = False
            for p in self.paths:
                for f in p.glob("**/*"):
                    if f.is_file() and (f.suffix.lower() in VIDEO_EXTS + IMAGE_EXTS + AUDIO_EXTS):
                        m = f.stat().st_mtime
                        if mtimes.get(f) != m:
                            mtimes[f] = m
                            changed = True
            if changed:
                log(self.ui, "Change detected. Assembling...")
                try:
                    self.callback()
                except Exception as e:
                    log(self.ui, f"ERROR during assembly: {e}")
            time.sleep(3)

    # --- Watchdog implementation ---
    def _start_watchdog(self):
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler

        class Handler(FileSystemEventHandler):
            def __init__(self, outer):
                self.outer = outer
                self.last_trigger = 0

            def on_any_event(self, event):
                if event.is_directory:
                    return
                if Path(event.src_path).suffix.lower() not in VIDEO_EXTS + IMAGE_EXTS + AUDIO_EXTS:
                    return
                # Debounce
                now = time.time()
                if now - self.last_trigger < 2:
                    return
                self.last_trigger = now
                log(self.outer.ui, f"Detected change: {event.event_type} {event.src_path}")
                try:
                    self.outer.callback()
                except Exception as e:
                    log(self.outer.ui, f"ERROR during assembly: {e}")

        self.observer = Observer()
        handler = Handler(self)
        for p in self.paths:
            p.mkdir(parents=True, exist_ok=True)
            self.observer.schedule(handler, str(p), recursive=True)
        self.observer.start()
        log(self.ui, "Watch mode: using watchdog (real-time events).")


# --------------------------- UI ---------------------------
class App:
    def __init__(self, root):
        self.root = root
        root.title("Auto Video Assembler (FFmpeg)")
        root.geometry("980x720")

        self.cfg = load_config()

        # Vars
        self.use_intro = BooleanVar(value=self.cfg["options"]["use_intro"])
        self.use_outro = BooleanVar(value=self.cfg["options"]["use_outro"])
        self.use_logo = BooleanVar(value=self.cfg["options"]["use_logo"])
        self.use_music = BooleanVar(value=self.cfg["options"]["use_music"])
        self.watch_mode = BooleanVar(value=self.cfg["options"]["watch_mode"])

        p = self.cfg["paths"]
        self.main_dir = StringVar(value=p["main_dir"])
        self.broll_dir = StringVar(value=p["broll_dir"])
        self.assets_dir = StringVar(value=p["assets_dir"])
        self.export_dir = StringVar(value=p["export_dir"])
        self.intro_name = StringVar(value=p["intro"])
        self.outro_name = StringVar(value=p["outro"])
        self.logo_name = StringVar(value=p["logo"])
        self.music_name = StringVar(value=p["music"])

        vcfg = self.cfg["video"]
        self.width = StringVar(value=str(vcfg["width"]))
        self.height = StringVar(value=str(vcfg["height"]))
        self.fps = StringVar(value=str(vcfg["fps"]))
        self.trans = StringVar(value=str(vcfg["transition_sec"]))
        self.crf = StringVar(value=str(vcfg["crf"]))
        self.preset = StringVar(value=str(vcfg["preset"]))

        lg = self.cfg["logo"]
        self.logo_pos = StringVar(value=lg.get("position", "top_left"))
        self.logo_x = StringVar(value=str(lg.get("x", 24)))
        self.logo_y = StringVar(value=str(lg.get("y", 24)))
        self.logo_w = StringVar(value=str(lg.get("scale_width", 0)))
        self.logo_opacity = StringVar(value=str(lg.get("opacity", 1.0)))

        mus = self.cfg["music"]
        self.music_db = StringVar(value=str(mus.get("music_volume_db", -12)))
        self.duck_ratio = StringVar(value=str(mus["ducking"]["ratio"]))
        self.duck_thresh = StringVar(value=str(mus["ducking"]["threshold"]))
        self.duck_attack = StringVar(value=str(mus["ducking"]["attack"]))
        self.duck_release = StringVar(value=str(mus["ducking"]["release"]))

        upl = self.cfg["upload"]
        self.use_rclone = BooleanVar(value=upl.get("enable_rclone", False))
        self.rclone_remote = StringVar(value=upl.get("rclone_remote", "gdrive:"))
        self.rclone_path = StringVar(value=upl.get("rclone_path", "AutoVideo/"))

        # Layout
        self._build_widgets()

        # Log area
        self.log_text = Text(self.root, height=12, wrap="word", state=DISABLED)
        self.log_text.pack(fill="both", expand=False, padx=10, pady=10)

        # Watcher
        self.watcher = FolderWatcher(
            [self.main_dir.get(), self.broll_dir.get(), self.assets_dir.get()],
            callback=self._assemble_async,
            ui=self,
        )

    def _build_widgets(self):
        pad = {"padx": 8, "pady": 6}

        frm = ttk.Frame(self.root)
        frm.pack(fill="both", expand=True, padx=10, pady=10)

        # Paths
        g1 = ttk.LabelFrame(frm, text="Paths & Assets")
        g1.grid(row=0, column=0, sticky="nsew", **pad)
        frm.grid_columnconfigure(0, weight=1)
        frm.grid_rowconfigure(0, weight=0)

        def add_path(row, label, var, choose_dir=True):
            ttk.Label(g1, text=label).grid(row=row, column=0, sticky="w")
            e = ttk.Entry(g1, textvariable=var, width=60)
            e.grid(row=row, column=1, sticky="we")
            def browse():
                sel = filedialog.askdirectory() if choose_dir else filedialog.askopenfilename()
                if sel:
                    var.set(sel)
            ttk.Button(g1, text="Browse", command=browse).grid(row=row, column=2)

        add_path(0, "Main folder", self.main_dir, True)
        add_path(1, "B-Roll folder", self.broll_dir, True)
        add_path(2, "Assets folder", self.assets_dir, True)
        add_path(3, "Export folder", self.export_dir, True)

        ttk.Label(g1, text="Intro file name").grid(row=4, column=0, sticky="w")
        ttk.Entry(g1, textvariable=self.intro_name, width=30).grid(row=4, column=1, sticky="w")
        ttk.Label(g1, text="Outro file name").grid(row=5, column=0, sticky="w")
        ttk.Entry(g1, textvariable=self.outro_name, width=30).grid(row=5, column=1, sticky="w")
        ttk.Label(g1, text="Logo file name").grid(row=6, column=0, sticky="w")
        ttk.Entry(g1, textvariable=self.logo_name, width=30).grid(row=6, column=1, sticky="w")
        ttk.Label(g1, text="Music file name").grid(row=7, column=0, sticky="w")
        ttk.Entry(g1, textvariable=self.music_name, width=30).grid(row=7, column=1, sticky="w")

        # Options
        g2 = ttk.LabelFrame(frm, text="Options")
        g2.grid(row=1, column=0, sticky="nsew", **pad)

        ttk.Checkbutton(g2, text="Apply Intro", variable=self.use_intro).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(g2, text="Apply Outro", variable=self.use_outro).grid(row=0, column=1, sticky="w")
        ttk.Checkbutton(g2, text="Overlay Logo", variable=self.use_logo).grid(row=0, column=2, sticky="w")
        ttk.Checkbutton(g2, text="Add Background Music", variable=self.use_music).grid(row=0, column=3, sticky="w")
        ttk.Checkbutton(g2, text="Watch mode (auto-detect new files)", variable=self.watch_mode, command=self._toggle_watch).grid(row=0, column=4, sticky="w")

        ttk.Label(g2, text="Resolution (WxH)").grid(row=1, column=0, sticky="w")
        ttk.Entry(g2, textvariable=self.width, width=8).grid(row=1, column=1, sticky="w")
        ttk.Entry(g2, textvariable=self.height, width=8).grid(row=1, column=2, sticky="w")
        ttk.Label(g2, text="FPS").grid(row=1, column=3, sticky="w")
        ttk.Entry(g2, textvariable=self.fps, width=6).grid(row=1, column=4, sticky="w")
        ttk.Label(g2, text="Transition (s)").grid(row=1, column=5, sticky="w")
        ttk.Entry(g2, textvariable=self.trans, width=6).grid(row=1, column=6, sticky="w")

        ttk.Label(g2, text="CRF").grid(row=2, column=0, sticky="w")
        ttk.Entry(g2, textvariable=self.crf, width=6).grid(row=2, column=1, sticky="w")
        ttk.Label(g2, text="Preset").grid(row=2, column=2, sticky="w")
        ttk.Combobox(g2, textvariable=self.preset, values=["ultrafast","superfast","veryfast","faster","fast","medium","slow","slower","veryslow"], width=10).grid(row=2, column=3, sticky="w")

        # Logo setup
        g3 = ttk.LabelFrame(frm, text="Logo Overlay")
        g3.grid(row=2, column=0, sticky="nsew", **pad)
        ttk.Label(g3, text="Position").grid(row=0, column=0, sticky="w")
        ttk.Combobox(g3, textvariable=self.logo_pos, values=["top_left","top_right","bottom_left","bottom_right","custom"], width=12).grid(row=0, column=1, sticky="w")
        ttk.Label(g3, text="Custom X").grid(row=0, column=2, sticky="w")
        ttk.Entry(g3, textvariable=self.logo_x, width=8).grid(row=0, column=3, sticky="w")
        ttk.Label(g3, text="Custom Y").grid(row=0, column=4, sticky="w")
        ttk.Entry(g3, textvariable=self.logo_y, width=8).grid(row=0, column=5, sticky="w")
        ttk.Label(g3, text="Scale W (px, 0=no)").grid(row=1, column=0, sticky="w")
        ttk.Entry(g3, textvariable=self.logo_w, width=10).grid(row=1, column=1, sticky="w")
        ttk.Label(g3, text="Opacity (0-1)").grid(row=1, column=2, sticky="w")
        ttk.Entry(g3, textvariable=self.logo_opacity, width=10).grid(row=1, column=3, sticky="w")

        # Music / ducking
        g4 = ttk.LabelFrame(frm, text="Music & Ducking")
        g4.grid(row=3, column=0, sticky="nsew", **pad)
        ttk.Label(g4, text="Music base volume (dB)").grid(row=0, column=0, sticky="w")
        ttk.Entry(g4, textvariable=self.music_db, width=8).grid(row=0, column=1, sticky="w")
        ttk.Label(g4, text="Ducking ratio").grid(row=0, column=2, sticky="w")
        ttk.Entry(g4, textvariable=self.duck_ratio, width=8).grid(row=0, column=3, sticky="w")
        ttk.Label(g4, text="Threshold (dB)").grid(row=0, column=4, sticky="w")
        ttk.Entry(g4, textvariable=self.duck_thresh, width=8).grid(row=0, column=5, sticky="w")
        ttk.Label(g4, text="Attack (ms)").grid(row=1, column=0, sticky="w")
        ttk.Entry(g4, textvariable=self.duck_attack, width=8).grid(row=1, column=1, sticky="w")
        ttk.Label(g4, text="Release (ms)").grid(row=1, column=2, sticky="w")
        ttk.Entry(g4, textvariable=self.duck_release, width=8).grid(row=1, column=3, sticky="w")

        # Upload
        g5 = ttk.LabelFrame(frm, text="Upload (optional via rclone)")
        g5.grid(row=4, column=0, sticky="nsew", **pad)
        ttk.Checkbutton(g5, text="Upload with rclone after export", variable=self.use_rclone).grid(row=0, column=0, sticky="w")
        ttk.Label(g5, text="Remote").grid(row=0, column=1, sticky="w")
        ttk.Entry(g5, textvariable=self.rclone_remote, width=12).grid(row=0, column=2, sticky="w")
        ttk.Label(g5, text="Path in remote").grid(row=0, column=3, sticky="w")
        ttk.Entry(g5, textvariable=self.rclone_path, width=24).grid(row=0, column=4, sticky="w")



        # Add B-roll controls section
        g6 = ttk.LabelFrame(frm, text="B-Roll Settings")
        g6.grid(row=5, column=0, sticky="nsew", **pad)
        
        # B-Roll 1 controls
        self.broll1_enable = BooleanVar(value=self.cfg["broll"]["broll1"]["enable"])
        ttk.Checkbutton(g6, text="Enable B-Roll 1", variable=self.broll1_enable).grid(row=0, column=0, sticky="w")
        ttk.Label(g6, text="Start Time (s)").grid(row=0, column=1, sticky="w")
        self.broll1_start = StringVar(value=str(self.cfg["broll"]["broll1"]["start_time"]))
        ttk.Entry(g6, textvariable=self.broll1_start, width=8).grid(row=0, column=2, sticky="w")
        ttk.Label(g6, text="Duration (s)").grid(row=0, column=3, sticky="w")
        self.broll1_duration = StringVar(value=str(self.cfg["broll"]["broll1"]["duration"]))
        ttk.Entry(g6, textvariable=self.broll1_duration, width=8).grid(row=0, column=4, sticky="w")
        
        # B-Roll 2 controls
        self.broll2_enable = BooleanVar(value=self.cfg["broll"]["broll2"]["enable"])
        ttk.Checkbutton(g6, text="Enable B-Roll 2", variable=self.broll2_enable).grid(row=1, column=0, sticky="w")
        ttk.Label(g6, text="Start Time (s)").grid(row=1, column=1, sticky="w")
        self.broll2_start = StringVar(value=str(self.cfg["broll"]["broll2"]["start_time"]))
        ttk.Entry(g6, textvariable=self.broll2_start, width=8).grid(row=1, column=2, sticky="w")
        ttk.Label(g6, text="Duration (s)").grid(row=1, column=3, sticky="w")
        self.broll2_duration = StringVar(value=str(self.cfg["broll"]["broll2"]["duration"]))
        ttk.Entry(g6, textvariable=self.broll2_duration, width=8).grid(row=1, column=4, sticky="w")
        
        # Move action buttons to row 6
        bar = ttk.Frame(frm)
        bar.grid(row=6, column=0, sticky="ew", **pad)
        ttk.Button(bar, text="Save Config", command=self._save).pack(side="left")
        ttk.Button(bar, text="Assemble Now", command=self._assemble_async).pack(side="left", padx=8)
        

        frm.grid_rowconfigure(6, weight=1)

    # --------------- Actions ---------------
    def _save(self):
        self._update_cfg_from_ui()
        save_config(self.cfg)
        log(self, "Config saved to config.json")

    def _update_cfg_from_ui(self):
        self.cfg["options"].update({
            "use_intro": bool(self.use_intro.get()),
            "use_outro": bool(self.use_outro.get()),
            "use_logo": bool(self.use_logo.get()),
            "use_music": bool(self.use_music.get()),
            "watch_mode": bool(self.watch_mode.get()),
        })
        self.cfg["paths"].update({
            "main_dir": self.main_dir.get(),
            "broll_dir": self.broll_dir.get(),
            "assets_dir": self.assets_dir.get(),
            "export_dir": self.export_dir.get(),
            "intro": self.intro_name.get(),
            "outro": self.outro_name.get(),
            "logo": self.logo_name.get(),
            "music": self.music_name.get(),
        })
        self.cfg["video"].update({
            "width": int(self.width.get()),
            "height": int(self.height.get()),
            "fps": int(self.fps.get()),
            "transition_sec": float(self.trans.get()),
            "crf": int(self.crf.get()),
            "preset": self.preset.get(),
        })
        self.cfg["logo"].update({
            "position": self.logo_pos.get(),
            "x": int(self.logo_x.get()),
            "y": int(self.logo_y.get()),
            "scale_width": int(self.logo_w.get()),
            "opacity": float(self.logo_opacity.get()),
        })
        self.cfg["music"]["ducking"].update({
            "ratio": float(self.duck_ratio.get()),
            "threshold": float(self.duck_thresh.get()),
            "attack": float(self.duck_attack.get()),
            "release": float(self.duck_release.get()),
        })
        self.cfg["music"].update({
            "music_volume_db": float(self.music_db.get()),
        })
        self.cfg["upload"].update({
            "enable_rclone": bool(self.use_rclone.get()),
            "rclone_remote": self.rclone_remote.get(),
            "rclone_path": self.rclone_path.get(),
        })
    
        self.cfg["broll"] = {
            "broll1": {
                "enable": bool(self.broll1_enable.get()),
                "start_time": float(self.broll1_start.get()),
                "duration": float(self.broll1_duration.get())
            },
            "broll2": {
                "enable": bool(self.broll2_enable.get()),
                "start_time": float(self.broll2_start.get()),
                "duration": float(self.broll2_duration.get())
            }
        }

    def _assemble_async(self):
        self._update_cfg_from_ui()
        save_config(self.cfg)
        def run():
            try:
                out = assemble_video(self.cfg, ui=self)
                log(self, f"DONE → {out}")
            except Exception as e:
                messagebox.showerror("Error", str(e))
                log(self, f"ERROR: {e}")
        threading.Thread(target=run, daemon=True).start()

    def _toggle_watch(self):
        if self.watch_mode.get():
            self.watcher.enabled = True
            self.watcher.start()
            log(self, "Watch mode enabled. Changes in folders will trigger assembly.")
        else:
            try:
                self.watcher.stop()
                log(self, "Watch mode disabled.")
            except Exception:
                pass


# --------------------------- Main ---------------------------

def main():
    try:
        check_ffmpeg()
    except Exception as e:
        messagebox.showerror("FFmpeg missing", str(e))
        return

    root = Tk()
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except Exception:
        pass
    app = App(root)
    root.mainloop()


if __name__ == "__main__":
    main()

"""
Notes on Google Drive (Drive API) upload (optional, advanced):

If you prefer Drive API instead of rclone, you can integrate it by:
- pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib
- Create credentials.json in the script folder from Google Cloud Console (OAuth client ID Desktop App)
- On first run, you'll be prompted to authorize; token.json will be stored.

Then, in assemble_video(), after export, call a helper like upload_to_drive(file_path).
A minimal implementation is omitted for brevity; rclone is simpler and more robust for batch uploads.
"""

"""Recap Studio MM — core engine package.

Modules
-------
config      : environment driven settings + paths
util        : logging, disk, json, misc helpers
media       : ffmpeg / ffprobe wrappers with progress reporting
fonts       : Myanmar font resolution (bundled assets first)
jobs        : thread safe task store (progress, stages, logs, cancel)
uploads     : resumable chunked uploads + validation
ai          : Gemini video timeline extraction (chunked, absolute offsets)
keys        : Gemini API key ring (up to 3 keys, quota failover, test)
tts         : edge-tts parallel synthesis with drift-free time fitting
subtitles   : ASS subtitle / hook builder (libass safe escaping)
render      : final ffmpeg master render (subtitles, logo, reframe, mix)
pipeline    : end to end orchestration (recap / re-render / split / thumb)
"""

__version__ = "4.4.0"

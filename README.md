# 🎬 Ricordi Cinematic Director (Ultimate Pro)

**Ricordi Cinematic Director** is a high-performance automated video editing engine designed to transform your photo collections into cinematic documentaries. Unlike basic slideshow generators, this script performs deep metadata and visual analysis to apply intelligent camera movements, audio synchronization, and spatial geocoding.

## ✨ Key Features

* **🔍 Smart Face-Focus & Ken Burns**: Automatically detects faces within images to center the Ken Burns zoom effect on people, ensuring professional framing every time.
* **🌍 Spatial Geocoding with Caching**: Retrieves location names via GPS. Includes a "Spatial Caching" logic to minimize API requests (it skips server pings if you've moved less than a set threshold).
* **🧠 Best-Shot Selection**: Analyzes bursts or clusters of similar photos and automatically selects the sharpest frame, eliminating duplicates and blurry shots.
* **🛡️ Anti-OOM (Memory Safe)**: Advanced RAM management designed for systems with limited resources. It uses thumbnails for analysis and dynamic resizing for the final rendering process.
* **📅 Daily Split & No-Split Options**: Choose between generating individual videos for each day of your trip or one single, seamless feature-length movie.
* **🎵 Audio-Sync**: Automatically synchronizes clip duration to the length of your background music tracks.
* **🌙 Night Mode Detection**: Detects low-light conditions and applies soft color correction filters to night shots for a better visual mood.
* **🇪🇺 European Date Formatting**: Overlays timestamps in the `DD/MM/YYYY` format for natural readability.

## 🛠️ Requirements

Ensure you have `ffmpeg` installed on your system:
```bash
sudo apt update && sudo apt install ffmpeg

Install the Python dependencies:
'''bash
pip install -r requirements.txt

###🚀 Usage

Basic Example

*Generate daily videos with automatic titles and geocoding:
'''bash
python ricordi_ultimate_pro.py /path/to/photos -ad /path/to/music -od ./Output_Videos

Feature Film Mode (No Split)
*Create one single video of the entire trip:
'''bash
python ricordi_ultimate_pro.py /path/to/photos -ad /path/to/music --no-split -t "My Great Adventure"

##Advanced Parameters
Flag                Description
--no-split          Disables daily video splitting; creates one single file.
--workers X	        Set the number of parallel processes (use 1 or 2 if you encounter RAM/OOM issues).
--geo-threshold X	Distance in meters for GPS location caching (default: 300m).
-r	                Resolution (e.g., 1080p, 4k, 720p).
-z	                Ken Burns zoom intensity (e.g., 1.2 for 20% zoom).
--limit X	        Process only the first X photos (great for quick testing).

##📁 Project Configuration (Optional)

You can create a project.json file to customize titles and music for specific dates:
''JSON
{
  "daily_configs": {
    "2024-08-15": {
      "title": "Arrival in Tokyo",
      "audio_theme": "/music/japan_zen.mp3"
    }
  }
}
##📄 License
Distributed under the GNU GPL v3 License. Built with passion for preserving memories in cinematic style.

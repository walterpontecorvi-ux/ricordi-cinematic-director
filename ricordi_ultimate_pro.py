#!/usr/bin/env python3
"""
Ricordi Cinematic Director - Ultimate Pro Edition (Safe Memory & Anti-OOM)
Features: GPS Spatial Caching, Face-Focus, Best-Shot, Audio-Sync, 
No-Split option, European Date Format, and Memory Protection.
License: GNU GPL v3
"""

import os
import json
import argparse
import logging
import time
import multiprocessing
import warnings
from math import radians, cos, sin, asin, sqrt
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import cv2
import exifread
from PIL import Image, ImageOps
from geopy.geocoders import Nominatim

# --- GLOBAL CONFIG & SILENCING ---
warnings.filterwarnings("ignore")
logging.getLogger('exifread').setLevel(logging.ERROR)

face_cascade = None
def init_worker():
    global face_cascade
    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(face_cascade_path)

# --- UTILITY & MATH FUNCTIONS ---

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculates the distance in meters between two GPS points."""
    if None in [lat1, lon1, lat2, lon2]: return float('inf')
    R = 6371 # Earth radius in km
    dLat, dLon = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(dLat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dLon/2)**2
    c = 2 * asin(sqrt(a))
    return R * c * 1000 

def get_sharpness(img_np):
    """Calculates image sharpness using Laplacian variance."""
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def get_phash(path):
    """Generates a perceptual hash for visual similarity clustering."""
    try:
        with Image.open(path) as img:
            # Memory protection: resizing for hash
            img.thumbnail((256, 256))
            img = ImageOps.exif_transpose(img).convert('L').resize((8, 8), Image.Resampling.LANCZOS)
            pixels = np.array(img)
            avg = pixels.mean()
            return (pixels > avg).flatten()
    except: return None

# --- TECHNICAL ANALYSIS ENGINE (MEMORY SAFE) ---

def analyze_photo(path):
    """Deep metadata and visual analysis with RAM protection."""
    info = {'path': str(path), 'date': None, 'place': None, 'face_y': None, 'lat': None, 'lon': None, 'is_night': False}
    try:
        with Image.open(path) as img:
            # 1. EXIF Analysis
            exif = img._getexif()
            if exif and 36867 in exif:
                dt = datetime.strptime(exif[36867], '%Y:%m:%d %H:%M:%S')
                info['date'] = dt.strftime('%Y-%m-%d %H:%M:%S')
                if dt.hour >= 20 or dt.hour <= 6: info['is_night'] = True
            
            # 2. Memory Safe Visual Analysis
            img.thumbnail((1200, 1200)) # Drastically reduce RAM usage
            img = ImageOps.exif_transpose(img)
            img_rgb = np.array(img.convert('RGB'))
            
            gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)
            if len(faces) > 0:
                best_face = max(faces, key=lambda f: f[2] * f[3])
                info['face_y'] = float((best_face[1] + best_face[3]/2) / img_rgb.shape[0])
            
            del img_rgb, gray # Explicit cleanup
    except: pass

    try:
        with open(path, 'rb') as f:
            tags = exifread.process_file(f, details=False)
            def to_dec(v, r):
                d = float(v.values[0].num)/float(v.values[0].den)
                m = float(v.values[1].num)/float(v.values[1].den)
                s = float(v.values[2].num)/float(v.values[2].den)
                return -(d + m/60 + s/3600) if str(r) in ['S', 'W'] else (d + m/60 + s/3600)
            lat, lat_r = tags.get('GPS GPSLatitude'), tags.get('GPS GPSLatitudeRef')
            lon, lon_r = tags.get('GPS GPSLongitude'), tags.get('GPS GPSLongitudeRef')
            if lat and lat_r and lon and lon_r:
                info['lat'], info['lon'] = to_dec(lat, lat_r), to_dec(lon, lon_r)
    except: pass

    if not info['date']:
        info['date'] = datetime.fromtimestamp(os.path.getmtime(path)).strftime('%Y-%m-%d %H:%M:%S')
    return info

# --- MAIN DIRECTOR CLASS ---

class RicordiDirector:
    def __init__(self, args):
        self.args = args
        self.config = self._load_json(args.config) if args.config else {}
        self.res = self._parse_res(args.resolution or self.config.get('settings', {}).get('resolution', '1080p'))
        self.target_ratio = self.res[0] / self.res[1]
        self.cache_path = Path(args.folder).expanduser() / "analysis_cache.json"
        self.logger = logging.getLogger("Director")

    def _load_json(self, path):
        if path and os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f: return json.load(f)
        return {}

    def _parse_res(self, res):
        presets = {'720p': (1280, 720), '1080p': (1920, 1080), '4k': (3840, 2160)}
        return presets.get(res.lower(), (1920, 1080))

    def _cluster_bursts(self, media_list):
        clusters, current, last_h = [], [], None
        for m in media_list:
            h = get_phash(m['path'])
            if last_h is not None and np.count_nonzero(h != last_h) < 7:
                current.append(m)
            else:
                if current: clusters.append(current)
                current = [m]
            last_h = h
        if current: clusters.append(current)
        return clusters

    def create_clip(self, cluster, duration):
        import moviepy.editor as mpy
        loaded = []
        for item in cluster:
            try:
                with Image.open(item['path']) as img:
                    # RAM Optimization: Don't load 40MP if target is 1080p
                    max_w = self.res[0] * 1.5
                    if img.width > max_w:
                        img = img.resize((int(max_w), int(img.height * (max_w/img.width))), Image.Resampling.LANCZOS)
                    img = ImageOps.exif_transpose(img)
                    img_np = np.array(img.convert('RGB'))
                    loaded.append({'img': img_np, 'item': item, 'sharpness': get_sharpness(img_np)})
            except: continue
        
        if not loaded: return None
        best = max(loaded, key=lambda x: x['sharpness'])

        def make_frame(t):
            prog = t / duration
            img_np = best['img']; h, w = img_np.shape[:2]
            
            if (w/h) > self.target_ratio:
                nw = int(h * self.target_ratio); ox = (w - nw) // 2
                f_c = img_np[:, ox:ox+nw]
            else:
                nh = int(w / self.target_ratio)
                fy = best['item'].get('face_y')
                oy = max(0, min(h - nh, int(fy * h) - (nh // 3))) if fy else int((h - nh) * 0.70)
                f_c = img_np[oy:oy+nh, :]
            
            res = cv2.resize(f_c, self.res, interpolation=cv2.INTER_AREA)
            z = self.args.zoom or 1.2
            cur_z = 1.0 + (z - 1.0) * prog
            nw_z, nh_z = int(self.res[0]/cur_z), int(self.res[1]/cur_z)
            cx, cy = self.res[0]//2, self.res[1]//2
            if best['item'].get('face_y'): cy = int(best['item']['face_y'] * self.res[1])
            
            x1, y1 = max(0, min(self.res[0]-nw_z, cx - nw_z//2)), max(0, min(self.res[1]-nh_z, cy - nh_z//2))
            final = cv2.resize(res[y1:y1+nh_z, x1:x1+nw_z], self.res)

            if best['item'].get('is_night'):
                final = cv2.addWeighted(final, 1.0, np.full(final.shape, (20, 40, 60), dtype=np.uint8), 0.15, 0)
            
            # EUROPEAN DATE FORMAT: DD/MM/YYYY
            dt_raw = best['item']['date'].split(' ')[0]
            dt_euro = datetime.strptime(dt_raw, '%Y-%m-%d').strftime('%d/%m/%Y')
            lbl = dt_euro
            if best['item'].get('place'): lbl += f" - {best['item']['place']}"
            cv2.putText(final, lbl, (62, self.res[1]-58), cv2.FONT_HERSHEY_DUPLEX, 1.1, (0,0,0), 4, cv2.LINE_AA)
            cv2.putText(final, lbl, (60, self.res[1]-60), cv2.FONT_HERSHEY_DUPLEX, 1.1, (255,255,255), 2, cv2.LINE_AA)
            return final

        return mpy.VideoClip(make_frame, duration=duration).set_fps(self.args.fps)

    def run(self):
        import moviepy.editor as mpy
        self.logger.info("🎬 Initializing Safe Engine...")
        
        # 1. Multi-core Analysis with RAM protection
        cache = {}
        if self.cache_path.exists():
            with open(self.cache_path, 'r') as f: cache = json.load(f)

        exts = {'.jpg', '.jpeg', '.png', '.heic', '.webp'}
        files = [f for f in Path(self.args.folder).expanduser().rglob('*') if f.suffix.lower() in exts]
        new_files = [f for f in files if str(f) not in cache]
        
        if new_files:
            workers = self.args.workers
            self.logger.info(f"🔍 Analyzing {len(new_files)} new images using {workers} workers...")
            with multiprocessing.Pool(processes=workers, initializer=init_worker) as pool:
                results = pool.map(analyze_photo, new_files)
                for r in results: cache[r['path']] = r
            with open(self.cache_path, 'w') as f: json.dump(cache, f, indent=2)

        # 2. GPS Processing (Spatial Caching)
        geo_needed = [m for m in cache.values() if m['lat'] and not m['place']]
        if geo_needed:
            self.logger.info("🌍 Geocoding with Spatial Caching...")
            geolocator = Nominatim(user_agent="ricordi_director")
            last_lat, last_lon, last_place = None, None, None
            for i, it in enumerate(geo_needed):
                dist = haversine_distance(it['lat'], it['lon'], last_lat, last_lon)
                if dist < self.args.geo_threshold:
                    it['place'] = last_place
                else:
                    try:
                        time.sleep(1.1) 
                        loc = geolocator.reverse((it['lat'], it['lon']), language='en', timeout=5)
                        if loc:
                            addr = loc.raw.get('address', {})
                            p = addr.get('city') or addr.get('town') or addr.get('village') or addr.get('suburb')
                            it['place'] = p
                            last_lat, last_lon, last_place = it['lat'], it['lon'], p
                    except: continue
                if i % 10 == 0: 
                    with open(self.cache_path, 'w') as f: json.dump(cache, f, indent=2)

        media = sorted(cache.values(), key=lambda x: x['date'])
        if self.args.limit: media = media[:self.args.limit]

        # 3. Decision: Split or Single Movie
        groups = defaultdict(list)
        if self.args.no_split:
            groups["Full_Movie"] = media
        else:
            for m in media:
                day_key = m['date'].split(' ')[0]
                groups[day_key].append(m)

        for key, photos in sorted(groups.items()):
            day_cfg = self.config.get("daily_configs", {}).get(key, {})
            out_dir = Path(self.args.output_dir or ".").expanduser()
            out_dir.mkdir(parents=True, exist_ok=True)
            
            title_tag = day_cfg.get('title', self.args.title or 'Journal').replace(' ', '_')
            output_file = out_dir / f"{key.replace('-', '')}_{title_tag}.mp4"

            if output_file.exists():
                self.logger.info(f"⏩ {key} exists. Skipping.")
                continue

            self.logger.info(f"🎞 Rendering: {key}")
            clusters = self._cluster_bursts(photos)
            
            # Audio Sync Logic
            audio_path = day_cfg.get("audio_theme") or self.args.audio_dir
            audios_files = list(Path(audio_path).expanduser().rglob('*.mp3'))
            total_audio_dur = 0
            if audios_files:
                for a in audios_files:
                    with mpy.AudioFileClip(str(a)) as af: total_audio_dur += af.duration

            fade = 0.8
            if (self.args.sync_audio or self.config.get('settings', {}).get('sync_audio')) and total_audio_dur > 0:
                dur_std = (total_audio_dur / len(clusters)) + fade
                dur_std = max(2.5, min(8.5, dur_std))
            else:
                dur_std = 4.5

            clips, t_curr = [], 0
            intro_t = day_cfg.get("title") or self.args.title or f"Date: {key}"
            clips.append(mpy.TextClip(intro_t, fontsize=70, color='white', size=self.res, bg_color='black').set_duration(3.5).fadeout(fade))
            t_curr = 2.8

            for clus in clusters:
                c = self.create_clip(clus, dur_std)
                if c:
                    clips.append(c.set_start(t_curr).crossfadein(fade))
                    t_curr += (dur_std - fade)

            final_v = mpy.CompositeVideoClip(clips, size=self.res)
            if audios_files:
                full_a = mpy.concatenate_audioclips([mpy.AudioFileClip(str(a)) for a in audios_files])
                final_v = final_v.set_audio(full_a.subclip(0, min(full_a.duration, final_v.duration)).audio_fadeout(3))

            final_v.write_videofile(str(output_file), fps=self.args.fps, threads=self.args.workers, preset="medium")
            
            # Explicit Memory Cleanup
            final_v.close()
            if audios_files: full_a.close()

# --- CLI SETUP ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ricordi Cinematic Director Ultimate Pro")
    parser.add_argument('folder', help="Photos folder")
    parser.add_argument('-ad', '--audio-dir', required=True, help="Music folder")
    parser.add_argument('-c', '--config', help="Project JSON")
    parser.add_argument('-od', '--output-dir', help="Output folder")
    parser.add_argument('-r', '--resolution', default='1080p')
    parser.add_argument('-f', '--fps', type=int, default=24)
    parser.add_argument('-z', '--zoom', type=float, default=1.2)
    parser.add_argument('-t', '--title', help="General title")
    parser.add_argument('--sync-audio', action='store_true')
    parser.add_argument('--no-split', action='store_true', help="Single movie mode")
    parser.add_argument('--workers', type=int, default=2, help="Lower if you have RAM issues")
    parser.add_argument('--limit', type=int)
    parser.add_argument('--geo-threshold', type=int, default=300)
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()
    log_level = logging.DEBUG if args.debug else (logging.INFO if args.verbose else logging.WARNING)
    logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')
    
    RicordiDirector(args).run()

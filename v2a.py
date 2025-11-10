import cv2
import numpy as np
import os
import sys
from pathlib import Path
from datetime import datetime
import argparse
import tkinter as tk
from tkinter import filedialog
from multiprocessing import Pool, cpu_count
from functools import partial
import struct
import gzip
import time

# Try to import pygame
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("⚠️  Warning: pygame not installed. Install with: pip install pygame")

# Try to import rich
try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
    from rich.prompt import Prompt, Confirm
    from rich.panel import Panel
    from rich.table import Table
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

console = Console() if RICH_AVAILABLE else None


class ASCIIVideoFormat:
    """Custom compressed ASCII video format (.ascvid)"""
    MAGIC = b'ASCV'
    VERSION = 3
    
    @staticmethod
    def save_frame(file, chars, colors):
        """Save a single frame"""
        chars_bytes = ''.join(chars).encode('utf-8')
        
        if colors:
            colors_bytes = b''.join(bytes(c) for c in colors)
        else:
            colors_bytes = b''
        
        data = struct.pack('<I', len(chars_bytes)) + chars_bytes + colors_bytes
        compressed = gzip.compress(data, compresslevel=6)
        
        file.write(struct.pack('<I', len(compressed)))
        file.write(compressed)
    
    @staticmethod
    def write_header(file, fps, total_frames, width, height, colored):
        """Write file header"""
        file.write(ASCIIVideoFormat.MAGIC)
        file.write(struct.pack('<B', ASCIIVideoFormat.VERSION))
        file.write(struct.pack('<f', fps))
        file.write(struct.pack('<I', total_frames))
        file.write(struct.pack('<H', width))
        file.write(struct.pack('<H', height))
        file.write(struct.pack('<B', 1 if colored else 0))
    
    @staticmethod
    def read_header(file):
        """Read file header"""
        magic = file.read(4)
        if magic != ASCIIVideoFormat.MAGIC:
            raise ValueError("Invalid ASCII video file")
        
        version = struct.unpack('<B', file.read(1))[0]
        if version != ASCIIVideoFormat.VERSION:
            raise ValueError(f"Unsupported version {version}. Please reconvert the video.")
        
        fps = struct.unpack('<f', file.read(4))[0]
        total_frames = struct.unpack('<I', file.read(4))[0]
        width = struct.unpack('<H', file.read(2))[0]
        height = struct.unpack('<H', file.read(2))[0]
        colored = struct.unpack('<B', file.read(1))[0] == 1
        
        return {
            'version': version,
            'fps': fps,
            'total_frames': total_frames,
            'width': width,
            'height': height,
            'colored': colored
        }
    
    @staticmethod
    def read_frame(file, width, height, colored):
        """Read a single frame"""
        size_data = file.read(4)
        if not size_data:
            return None, None
        
        size = struct.unpack('<I', size_data)[0]
        compressed = file.read(size)
        data = gzip.decompress(compressed)
        
        chars_len = struct.unpack('<I', data[:4])[0]
        chars_bytes = data[4:4+chars_len]
        chars = chars_bytes.decode('utf-8')
        
        colors = None
        if colored:
            colors_bytes = data[4+chars_len:]
            colors = []
            for i in range(0, len(colors_bytes), 3):
                colors.append((colors_bytes[i], colors_bytes[i+1], colors_bytes[i+2]))
        
        return chars, colors


def render_frame_simple(chars, colors, ascii_width, ascii_height, colored, char_width=6, char_height=10):
    """Simple but reliable frame rendering using OpenCV"""
    # Create image
    img_width = ascii_width * char_width
    img_height = ascii_height * char_height
    img = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    
    # Use OpenCV to draw text
    font = cv2.FONT_HERSHEY_PLAIN
    font_scale = 0.8
    thickness = 1
    
    for y in range(ascii_height):
        for x in range(ascii_width):
            idx = y * ascii_width + x
            if idx < len(chars):
                char = chars[idx]
                
                # Get color
                if colored and colors and idx < len(colors):
                    color_bgr = (colors[idx][2], colors[idx][1], colors[idx][0])  # RGB to BGR
                else:
                    color_bgr = (200, 200, 200)
                
                # Position
                text_x = x * char_width
                text_y = y * char_height + char_height - 2
                
                # Draw character
                cv2.putText(img, char, (text_x, text_y), font, font_scale, color_bgr, thickness, cv2.LINE_AA)
    
    return img


def render_frame_pil(chars, colors, ascii_width, ascii_height, colored, font_size=10):
    """High-quality rendering using PIL"""
    try:
        from PIL import Image, ImageDraw, ImageFont
        
        char_width = int(font_size * 0.6)
        char_height = font_size
        
        img_width = ascii_width * char_width
        img_height = ascii_height * char_height
        
        # Create PIL image
        pil_img = Image.new('RGB', (img_width, img_height), color=(0, 0, 0))
        draw = ImageDraw.Draw(pil_img)
        
        # Load font
        font = None
        font_paths = [
            "cour.ttf",
            "Courier New.ttf", 
            "C:\\Windows\\Fonts\\cour.ttf",
            "/System/Library/Fonts/Courier.dfont",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf"
        ]
        
        for font_path in font_paths:
            try:
                font = ImageFont.truetype(font_path, font_size)
                break
            except:
                continue
        
        if not font:
            font = ImageFont.load_default()
        
        # Draw characters
        for y in range(ascii_height):
            for x in range(ascii_width):
                idx = y * ascii_width + x
                if idx < len(chars):
                    char = chars[idx]
                    
                    # Get color
                    if colored and colors and idx < len(colors):
                        color = colors[idx]
                    else:
                        color = (200, 200, 200)
                    
                    # Draw
                    draw.text((x * char_width, y * char_height), char, fill=color, font=font)
        
        # Convert to numpy array (RGB)
        return np.array(pil_img)
        
    except ImportError:
        return None


def render_single_frame_for_export(args):
    """Worker function for parallel frame rendering - FIXED VERSION"""
    frame_num, chars, colors, ascii_width, ascii_height, colored, font_size, use_pil = args
    
    if use_pil:
        # Try PIL first (better quality)
        img = render_frame_pil(chars, colors, ascii_width, ascii_height, colored, font_size)
        if img is not None:
            return frame_num, img
    
    # Fallback to OpenCV (more compatible)
    char_width = int(font_size * 0.6)
    char_height = font_size
    img = render_frame_simple(chars, colors, ascii_width, ascii_height, colored, char_width, char_height)
    
    return frame_num, img


class VideoToASCII:
    CHARSETS = {
        "1": (" .:-=+*#%@", "Simple"),
        "2": (" .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$", "Standard"),
        "3": (" ░▒▓█", "Blocks"),
        "4": (" .,:;i1tfLCG08@", "Detailed"),
        "5": (" .:;+=xX$&#", "Classic")
    }
    
    def __init__(self, video_path, width=100, height=None, colored=True, charset="2", workers=None):
        self.video_path = video_path
        self.width = width
        self.height = height
        self.colored = colored
        self.chars = self.CHARSETS.get(charset, self.CHARSETS["2"])[0]
        self.workers = workers or min(cpu_count(), 8)
    
    def calculate_height(self, video_width, video_height, char_aspect=0.5):
        """Calculate ASCII height to maintain video aspect ratio"""
        if self.height:
            return self.height
        
        video_aspect = video_width / video_height
        ascii_height = int(self.width * char_aspect / video_aspect)
        
        return ascii_height
    
    def save_as_ascvid(self, output_file, max_frames=None):
        """Save as custom .ascvid format"""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return False
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if max_frames:
            total_frames = min(total_frames, max_frames)
        
        ascii_height = self.calculate_height(video_width, video_height, char_aspect=0.5)
        
        if RICH_AVAILABLE:
            console.print(f"[cyan]📐 Video: {video_width}x{video_height} ({video_width/video_height:.2f}:1)[/cyan]")
            console.print(f"[cyan]📐 ASCII: {self.width}x{ascii_height} chars[/cyan]")
        else:
            print(f"📐 Video: {video_width}x{video_height} ({video_width/video_height:.2f}:1)")
            print(f"📐 ASCII: {self.width}x{ascii_height} chars")
        
        with open(output_file, 'wb') as f:
            ASCIIVideoFormat.write_header(f, fps, total_frames, self.width, ascii_height, self.colored)
        
        batch_size = self.workers * 4
        
        process_func = partial(process_frame_to_data, 
                              width=self.width,
                              height=ascii_height,
                              chars=self.chars, 
                              colored=self.colored)
        
        if RICH_AVAILABLE:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
                console=console
            ) as progress:
                task = progress.add_task(f"[cyan]Converting ({self.workers} workers)...", total=total_frames)
                
                frame_count = 0
                with Pool(self.workers) as pool:
                    while True:
                        if max_frames and frame_count >= max_frames:
                            break
                        
                        batch = []
                        for _ in range(batch_size):
                            if max_frames and frame_count + len(batch) >= max_frames:
                                break
                            ret, frame = cap.read()
                            if not ret:
                                break
                            batch.append(frame)
                        
                        if not batch:
                            break
                        
                        with open(output_file, 'ab') as f:
                            for chars, colors in pool.imap(process_func, batch):
                                ASCIIVideoFormat.save_frame(f, chars, colors)
                                frame_count += 1
                                progress.update(task, advance=1)
        else:
            frame_count = 0
            with Pool(self.workers) as pool:
                while True:
                    if max_frames and frame_count >= max_frames:
                        break
                    
                    batch = []
                    for _ in range(batch_size):
                        if max_frames and frame_count + len(batch) >= max_frames:
                            break
                        ret, frame = cap.read()
                        if not ret:
                            break
                        batch.append(frame)
                    
                    if not batch:
                        break
                    
                    with open(output_file, 'ab') as f:
                        for chars, colors in pool.imap(process_func, batch):
                            ASCIIVideoFormat.save_frame(f, chars, colors)
                            frame_count += 1
                            print(f"Processing: {frame_count}/{total_frames} ({frame_count/total_frames*100:.1f}%)", end='\r')
        
        cap.release()
        print()
        return True


class ASCIIVideoPlayer:
    """Player with GUI, Terminal, and FIXED Video Export"""
    
    def __init__(self, video_file):
        self.video_file = video_file
        self.metadata = None
        self.frame_positions = []
        self.current_frame = 0
        self.playing = False
        self.surface_cache = {}
        
    def load_metadata(self):
        """Load video metadata and build frame index"""
        with open(self.video_file, 'rb') as f:
            self.metadata = ASCIIVideoFormat.read_header(f)
            
            current_pos = f.tell()
            for i in range(self.metadata['total_frames']):
                self.frame_positions.append(current_pos)
                size_data = f.read(4)
                if not size_data:
                    break
                size = struct.unpack('<I', size_data)[0]
                f.seek(size, 1)
                current_pos = f.tell()
        
        return self.metadata
    
    def get_frame(self, frame_num):
        """Get a specific frame"""
        if frame_num >= len(self.frame_positions):
            return None, None
        
        with open(self.video_file, 'rb') as f:
            f.seek(self.frame_positions[frame_num])
            return ASCIIVideoFormat.read_frame(f, 
                                              self.metadata['width'], 
                                              self.metadata['height'],
                                              self.metadata['colored'])
    
    def export_to_video_fast(self, output_file, scale=1.0, font_size=10, workers=None):
        """FIXED: Fast parallel video export"""
        if not self.metadata:
            self.load_metadata()
        
        if workers is None:
            workers = min(cpu_count(), 8)
        
        # Check if PIL is available
        try:
            from PIL import Image
            use_pil = True
            if RICH_AVAILABLE:
                console.print("[green]✓[/green] Using PIL for high-quality rendering")
            else:
                print("✓ Using PIL for high-quality rendering")
        except ImportError:
            use_pil = False
            if RICH_AVAILABLE:
                console.print("[yellow]⚠️[/yellow] PIL not available, using OpenCV (lower quality)")
            else:
                print("⚠️  Using OpenCV rendering (install Pillow for better quality)")
        
        # Calculate dimensions
        char_width = int(font_size * 0.6)
        char_height = font_size
        
        base_width = self.metadata['width'] * char_width
        base_height = self.metadata['height'] * char_height
        
        video_width = int(base_width * scale)
        video_height = int(base_height * scale)
        
        # Make even
        video_width = video_width + (video_width % 2)
        video_height = video_height + (video_height % 2)
        
        if RICH_AVAILABLE:
            console.print(f"\n[cyan]🎬 Export settings:[/cyan]")
            console.print(f"[cyan]📐 Output: {video_width}x{video_height}px @ {self.metadata['fps']:.1f} FPS[/cyan]")
            console.print(f"[cyan]🎞️  Frames: {self.metadata['total_frames']}[/cyan]")
            console.print(f"[cyan]⚡ Workers: {workers}[/cyan]")
        else:
            print(f"\n🎬 Output: {video_width}x{video_height}px @ {self.metadata['fps']:.1f} FPS")
            print(f"🎞️  {self.metadata['total_frames']} frames")
            print(f"⚡ {workers} workers")
        
        # Create video writer - try different codecs
        fourcc_options = [
            ('mp4v', cv2.VideoWriter_fourcc(*'mp4v')),
            ('avc1', cv2.VideoWriter_fourcc(*'avc1')),
            ('XVID', cv2.VideoWriter_fourcc(*'XVID')),
        ]
        
        out = None
        for codec_name, fourcc in fourcc_options:
            out = cv2.VideoWriter(output_file, fourcc, self.metadata['fps'], (video_width, video_height))
            if out.isOpened():
                if RICH_AVAILABLE:
                    console.print(f"[green]✓[/green] Using codec: {codec_name}")
                else:
                    print(f"✓ Using codec: {codec_name}")
                break
            out.release()
        
        if not out or not out.isOpened():
            print("❌ Failed to create video writer!")
            return False
        
        try:
            # Load all frames first
            if RICH_AVAILABLE:
                console.print("\n[cyan]📖 Loading frames...[/cyan]")
            else:
                print("\n📖 Loading frames...")
            
            frames_data = []
            for frame_num in range(self.metadata['total_frames']):
                chars, colors = self.get_frame(frame_num)
                if chars:
                    frames_data.append((frame_num, chars, colors, 
                                      self.metadata['width'], self.metadata['height'], 
                                      self.metadata['colored'], font_size, use_pil))
            
            # Render frames in parallel
            if RICH_AVAILABLE:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                    TimeRemainingColumn(),
                    console=console
                ) as progress:
                    task = progress.add_task(f"[cyan]Rendering video ({workers} workers)...", 
                                           total=len(frames_data))
                    
                    # Process in batches
                    batch_size = workers * 4
                    rendered_frames = {}
                    
                    with Pool(workers) as pool:
                        for i in range(0, len(frames_data), batch_size):
                            batch = frames_data[i:i + batch_size]
                            
                            for frame_num, img in pool.imap_unordered(render_single_frame_for_export, batch):
                                rendered_frames[frame_num] = img
                                progress.update(task, advance=1)
                            
                            # Write frames in order
                            while len(rendered_frames) > batch_size // 2:
                                frame_nums = sorted(rendered_frames.keys())
                                for fn in frame_nums[:batch_size // 4]:
                                    img = rendered_frames[fn]
                                    
                                    # Scale if needed
                                    if scale != 1.0:
                                        img = cv2.resize(img, (video_width, video_height), 
                                                       interpolation=cv2.INTER_LINEAR)
                                    
                                    # Convert RGB to BGR for OpenCV
                                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                                    
                                    # Write frame
                                    out.write(img_bgr)
                                    
                                    del rendered_frames[fn]
                    
                    # Write remaining frames
                    for frame_num in sorted(rendered_frames.keys()):
                        img = rendered_frames[frame_num]
                        
                        if scale != 1.0:
                            img = cv2.resize(img, (video_width, video_height), 
                                           interpolation=cv2.INTER_LINEAR)
                        
                        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        out.write(img_bgr)
            else:
                # Non-rich progress
                rendered_frames = {}
                batch_size = workers * 4
                processed = 0
                
                with Pool(workers) as pool:
                    for i in range(0, len(frames_data), batch_size):
                        batch = frames_data[i:i + batch_size]
                        
                        for frame_num, img in pool.imap_unordered(render_single_frame_for_export, batch):
                            rendered_frames[frame_num] = img
                            processed += 1
                            print(f"Rendering: {processed}/{len(frames_data)} ({processed/len(frames_data)*100:.1f}%)", end='\r')
                        
                        while len(rendered_frames) > batch_size // 2:
                            frame_nums = sorted(rendered_frames.keys())
                            for fn in frame_nums[:batch_size // 4]:
                                img = rendered_frames[fn]
                                
                                if scale != 1.0:
                                    img = cv2.resize(img, (video_width, video_height))
                                
                                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                                out.write(img_bgr)
                                
                                del rendered_frames[fn]
                
                for frame_num in sorted(rendered_frames.keys()):
                    img = rendered_frames[frame_num]
                    
                    if scale != 1.0:
                        img = cv2.resize(img, (video_width, video_height))
                    
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    out.write(img_bgr)
                
                print()
            
            out.release()
            return True
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            out.release()
            return False
    
    def render_frame_to_surface(self, chars, colors, font, char_width, char_height):
        """Pre-render a frame to pygame surface"""
        width = self.metadata['width'] * char_width
        height = self.metadata['height'] * char_height
        surface = pygame.Surface((width, height))
        surface.fill((0, 0, 0))
        
        if self.metadata['colored'] and colors:
            for y in range(self.metadata['height']):
                for x in range(self.metadata['width']):
                    idx = y * self.metadata['width'] + x
                    if idx < len(chars):
                        char = chars[idx]
                        color = colors[idx] if idx < len(colors) else (200, 200, 200)
                        
                        char_surface = font.render(char, True, color)
                        surface.blit(char_surface, (x * char_width, y * char_height))
        else:
            for y in range(self.metadata['height']):
                for x in range(self.metadata['width']):
                    idx = y * self.metadata['width'] + x
                    if idx < len(chars):
                        char = chars[idx]
                        char_surface = font.render(char, True, (200, 200, 200))
                        surface.blit(char_surface, (x * char_width, y * char_height))
        
        return surface
    
    def play_gui(self, font_size=None):
        """Play video in GUI with auto-scaling"""
        if not PYGAME_AVAILABLE:
            print("❌ Pygame not installed.")
            return
        
        if not self.metadata:
            self.load_metadata()
        
        pygame.init()
        
        display_info = pygame.display.Info()
        screen_width = display_info.current_w
        screen_height = display_info.current_h
        
        if font_size is None:
            font_size = 10
        
        font = None
        font_names = ['Courier New', 'Consolas', 'Monaco', 'Menlo', 'DejaVu Sans Mono', 'courier']
        
        for font_name in font_names:
            try:
                font = pygame.font.SysFont(font_name, font_size)
                if font:
                    break
            except:
                continue
        
        if not font:
            font = pygame.font.Font(None, font_size)
        
        test_surface = font.render('W', True, (255, 255, 255))
        char_width = test_surface.get_width()
        char_height = test_surface.get_height()
        
        native_width = self.metadata['width'] * char_width
        native_height = self.metadata['height'] * char_height
        
        controls_height = 90
        max_window_width = int(screen_width * 0.85)
        max_window_height = int(screen_height * 0.85)
        max_video_height = max_window_height - controls_height - 60
        
        if native_width > max_window_width or native_height > max_video_height:
            scale_w = max_window_width / native_width
            scale_h = max_video_height / native_height
            scale = min(scale_w, scale_h)
            
            window_width = int(native_width * scale) + 40
            window_height = int(native_height * scale) + controls_height + 60
        else:
            window_width = native_width + 40
            window_height = native_height + controls_height + 60
        
        screen = pygame.display.set_mode((window_width, window_height), pygame.RESIZABLE)
        pygame.display.set_caption(f"ASCII Video Player - {Path(self.video_file).name}")
        clock = pygame.time.Clock()
        
        BG_COLOR = (10, 10, 10)
        UI_BG_COLOR = (25, 25, 25)
        UI_TEXT_COLOR = (180, 180, 180)
        BUTTON_COLOR = (50, 50, 50)
        BUTTON_HOVER_COLOR = (70, 70, 70)
        BUTTON_ACTIVE_COLOR = (0, 150, 100)
        ACCENT_COLOR = (0, 200, 255)
        PROGRESS_BG = (40, 40, 40)
        
        ui_font = pygame.font.Font(None, 22)
        small_font = pygame.font.Font(None, 16)
        
        max_cache_size = 30
        
        fps = self.metadata['fps']
        dragging_progress = False
        fullscreen = False
        auto_scale = True
        
        print(f"\n🎮 GUI Player Started!")
        print(f"📺 Window: {window_width}x{window_height}")
        print(f"🎨 ASCII: {self.metadata['width']}x{self.metadata['height']} chars")
        print(f"\n⌨️  CONTROLS:")
        print(f"   Space - Play/Pause  |  ← → - Skip  |  F - Fullscreen  |  Q - Quit\n")
        
        running = True
        last_window_size = (window_width, window_height)
        
        while running:
            current_width, current_height = screen.get_size()
            
            if (current_width, current_height) != last_window_size:
                last_window_size = (current_width, current_height)
                self.surface_cache.clear()
            
            controls_height = 90
            available_height = current_height - controls_height - 40
            available_width = current_width - 40
            
            if auto_scale:
                scale_w = available_width / native_width
                scale_h = available_height / native_height
                scale = min(scale_w, scale_h, 1.0)
                
                scaled_width = int(native_width * scale)
                scaled_height = int(native_height * scale)
            else:
                scaled_width = native_width
                scaled_height = native_height
            
            video_x = (current_width - scaled_width) // 2
            video_y = 20
            video_rect = pygame.Rect(video_x, video_y, scaled_width, scaled_height)
            
            controls_y = video_y + scaled_height + 10
            controls_rect = pygame.Rect(20, controls_y, current_width - 40, 80)
            progress_rect = pygame.Rect(30, controls_y + 10, current_width - 60, 15)
            
            button_y = controls_y + 35
            button_height = 28
            
            play_button = pygame.Rect(30, button_y, 90, button_height)
            restart_button = pygame.Rect(130, button_y, 90, button_height)
            skip_back_button = pygame.Rect(230, button_y, 70, button_height)
            skip_forward_button = pygame.Rect(310, button_y, 70, button_height)
            scale_button = pygame.Rect(390, button_y, 90, button_height)
            fullscreen_button = pygame.Rect(490, button_y, 90, button_height)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.playing = not self.playing
                    elif event.key == pygame.K_LEFT:
                        self.current_frame = max(0, self.current_frame - int(fps * 5))
                    elif event.key == pygame.K_RIGHT:
                        self.current_frame = min(self.metadata['total_frames'] - 1, 
                                                self.current_frame + int(fps * 5))
                    elif event.key == pygame.K_r:
                        self.current_frame = 0
                        self.playing = False
                    elif event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_UP:
                        fps = min(120, fps + 5)
                    elif event.key == pygame.K_DOWN:
                        fps = max(1, fps - 5)
                    elif event.key == pygame.K_f:
                        fullscreen = not fullscreen
                        if fullscreen:
                            screen = pygame.display.set_mode((screen_width, screen_height), pygame.FULLSCREEN)
                        else:
                            screen = pygame.display.set_mode((window_width, window_height), pygame.RESIZABLE)
                        self.surface_cache.clear()
                    elif event.key == pygame.K_a:
                        auto_scale = not auto_scale
                        self.surface_cache.clear()
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = event.pos
                    
                    if play_button.collidepoint(mouse_pos):
                        self.playing = not self.playing
                    elif restart_button.collidepoint(mouse_pos):
                        self.current_frame = 0
                        self.playing = False
                    elif skip_back_button.collidepoint(mouse_pos):
                        self.current_frame = max(0, self.current_frame - int(fps * 5))
                    elif skip_forward_button.collidepoint(mouse_pos):
                        self.current_frame = min(self.metadata['total_frames'] - 1, 
                                                self.current_frame + int(fps * 5))
                    elif scale_button.collidepoint(mouse_pos):
                        auto_scale = not auto_scale
                        self.surface_cache.clear()
                    elif fullscreen_button.collidepoint(mouse_pos):
                        fullscreen = not fullscreen
                        if fullscreen:
                            screen = pygame.display.set_mode((screen_width, screen_height), pygame.FULLSCREEN)
                        else:
                            screen = pygame.display.set_mode((window_width, window_height), pygame.RESIZABLE)
                        self.surface_cache.clear()
                    elif progress_rect.collidepoint(mouse_pos):
                        dragging_progress = True
                        progress = (mouse_pos[0] - progress_rect.left) / progress_rect.width
                        self.current_frame = int(progress * self.metadata['total_frames'])
                        self.current_frame = max(0, min(self.current_frame, self.metadata['total_frames'] - 1))
                
                elif event.type == pygame.MOUSEBUTTONUP:
                    dragging_progress = False
                
                elif event.type == pygame.MOUSEMOTION:
                    if dragging_progress:
                        progress = (event.pos[0] - progress_rect.left) / progress_rect.width
                        self.current_frame = int(progress * self.metadata['total_frames'])
                        self.current_frame = max(0, min(self.current_frame, self.metadata['total_frames'] - 1))
            
            if self.playing:
                self.current_frame += 1
                if self.current_frame >= self.metadata['total_frames']:
                    self.current_frame = 0
                    self.playing = False
            
            cache_key = f"{self.current_frame}"
            
            if cache_key not in self.surface_cache:
                chars, colors = self.get_frame(self.current_frame)
                if chars:
                    frame_surface = self.render_frame_to_surface(chars, colors, font, char_width, char_height)
                    self.surface_cache[cache_key] = frame_surface
                    
                    if len(self.surface_cache) > max_cache_size:
                        oldest_key = next(iter(self.surface_cache))
                        del self.surface_cache[oldest_key]
            
            frame_surface = self.surface_cache.get(cache_key)
            
            screen.fill(BG_COLOR)
            
            if frame_surface:
                if auto_scale and (scaled_width != native_width or scaled_height != native_height):
                    scaled_surface = pygame.transform.smoothscale(frame_surface, (scaled_width, scaled_height))
                    screen.blit(scaled_surface, video_rect)
                else:
                    screen.blit(frame_surface, video_rect)
            
            pygame.draw.rect(screen, (50, 50, 50), video_rect, 1)
            pygame.draw.rect(screen, UI_BG_COLOR, controls_rect, border_radius=8)
            
            pygame.draw.rect(screen, PROGRESS_BG, progress_rect, border_radius=4)
            if self.metadata['total_frames'] > 0:
                progress = self.current_frame / max(1, self.metadata['total_frames'] - 1)
                progress_width = int(progress_rect.width * progress)
                if progress_width > 0:
                    fill_rect = pygame.Rect(progress_rect.left, progress_rect.top, 
                                           progress_width, progress_rect.height)
                    pygame.draw.rect(screen, ACCENT_COLOR, fill_rect, border_radius=4)
            
            mouse_pos = pygame.mouse.get_pos()
            
            def draw_button(rect, text, is_active=False):
                if is_active:
                    color = BUTTON_ACTIVE_COLOR
                elif rect.collidepoint(mouse_pos):
                    color = BUTTON_HOVER_COLOR
                else:
                    color = BUTTON_COLOR
                
                pygame.draw.rect(screen, color, rect, border_radius=5)
                pygame.draw.rect(screen, (100, 100, 100), rect, 1, border_radius=5)
                
                text_surface = small_font.render(text, True, (255, 255, 255))
                text_rect = text_surface.get_rect(center=rect.center)
                screen.blit(text_surface, text_rect)
            
            draw_button(play_button, "⏸ Pause" if self.playing else "▶ Play", self.playing)
            draw_button(restart_button, "⏮ Restart")
            draw_button(skip_back_button, "⏪ -5s")
            draw_button(skip_forward_button, "⏩ +5s")
            draw_button(scale_button, "🔲 Fit" if auto_scale else "📏 1:1", auto_scale)
            draw_button(fullscreen_button, "🗗 Exit" if fullscreen else "⛶ Full")
            
            current_time = self.current_frame / fps
            total_time = self.metadata['total_frames'] / fps
            
            if auto_scale:
                scale_percent = int((scaled_width / native_width) * 100)
            else:
                scale_percent = 100
            
            info_text = f"{self.current_frame + 1}/{self.metadata['total_frames']} | {format_time(current_time)}/{format_time(total_time)} | {fps:.0f} FPS | {scale_percent}%"
            
            info_surface = small_font.render(info_text, True, UI_TEXT_COLOR)
            screen.blit(info_surface, (fullscreen_button.right + 15, button_y + 6))
            
            pygame.display.flip()
            clock.tick(fps)
        
        pygame.quit()
        print("\n👋 Player closed")
    
    def play_terminal(self):
        """Play in terminal with colors"""
        if not self.metadata:
            self.load_metadata()
        
        fps = self.metadata['fps']
        frame_delay = 1.0 / fps
        
        os.system('cls' if os.name == 'nt' else 'clear')
        
        if RICH_AVAILABLE:
            console.print(Panel(f"""
[bold cyan]🎬 ASCII Video Player - Terminal[/bold cyan]

[yellow]📹 File:[/yellow] {Path(self.video_file).name}
[yellow]📊 Resolution:[/yellow] {self.metadata['width']}x{self.metadata['height']} chars
[yellow]🎞️  Frames:[/yellow] {self.metadata['total_frames']} @ {fps:.1f} FPS
[yellow]🎨 Colors:[/yellow] {'Enabled ✨' if self.metadata['colored'] else 'Disabled'}

[dim]Press Ctrl+C to stop[/dim]
            """, border_style="cyan"))
            input("\nPress ENTER to start...")
        else:
            print("🎬 ASCII Video Player")
            print(f"📹 {Path(self.video_file).name}")
            input("\nPress ENTER to start...")
        
        try:
            self.current_frame = 0
            
            while self.current_frame < self.metadata['total_frames']:
                start_time = time.time()
                
                chars, colors = self.get_frame(self.current_frame)
                if not chars:
                    break
                
                os.system('cls' if os.name == 'nt' else 'clear')
                
                if self.metadata['colored'] and colors:
                    output_lines = []
                    for y in range(self.metadata['height']):
                        line = ""
                        for x in range(self.metadata['width']):
                            idx = y * self.metadata['width'] + x
                            char = chars[idx]
                            r, g, b = colors[idx]
                            line += f"\033[38;2;{r};{g};{b}m{char}"
                        line += "\033[0m"
                        output_lines.append(line)
                    
                    print('\n'.join(output_lines))
                else:
                    for y in range(self.metadata['height']):
                        start_idx = y * self.metadata['width']
                        end_idx = start_idx + self.metadata['width']
                        print(chars[start_idx:end_idx])
                
                current_time = self.current_frame / fps
                total_time = self.metadata['total_frames'] / fps
                progress_bar_width = 40
                progress = self.current_frame / max(1, self.metadata['total_frames'] - 1)
                filled = int(progress_bar_width * progress)
                bar = "█" * filled + "░" * (progress_bar_width - filled)
                
                print(f"\n\033[36m{bar}\033[0m")
                print(f"📹 \033[33m{self.current_frame + 1}\033[0m/\033[33m{self.metadata['total_frames']}\033[0m | "
                      f"⏱️  \033[32m{format_time(current_time)}\033[0m/\033[32m{format_time(total_time)}\033[0m | "
                      f"🎬 \033[35m{fps:.1f} FPS\033[0m")
                
                self.current_frame += 1
                
                elapsed = time.time() - start_time
                sleep_time = max(0, frame_delay - elapsed)
                time.sleep(sleep_time)
            
            print("\n\n✅ \033[32mDone!\033[0m")
            
        except KeyboardInterrupt:
            print("\n\n⏹️  \033[33mStopped\033[0m")
    
    def get_info(self):
        """Get info"""
        if not self.metadata:
            self.load_metadata()
        return self.metadata


def process_frame_to_data(frame, width, height, chars, colored):
    """Process frame to ASCII data"""
    frame_height, frame_width = frame.shape[:2]
    
    if height is None:
        aspect_ratio = frame_height / frame_width
        height = int(width * aspect_ratio * 0.55)
    
    frame = cv2.resize(frame, (width, height))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    char_list = []
    color_list = []
    
    for y in range(frame.shape[0]):
        for x in range(frame.shape[1]):
            brightness = gray[y, x]
            char_idx = int((brightness / 255) * (len(chars) - 1))
            char_list.append(chars[char_idx])
            
            if colored:
                b, g, r = frame[y, x]
                color_list.append((int(r), int(g), int(b)))
    
    return char_list, color_list if colored else None


def format_time(seconds):
    """Format time"""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins}:{secs:02d}"


def select_video_file():
    """File picker"""
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        file_path = filedialog.askopenfilename(
            title="Select a video file",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv *.webm *.m4v *.ascvid"),
                ("ASCII Video", "*.ascvid"),
                ("All files", "*.*")
            ]
        )
        
        root.destroy()
        return file_path if file_path else None
    except Exception as e:
        print(f"Error: {e}")
        return None


def select_output_location(default_name="ascii_video.ascvid"):
    """Save dialog"""
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        if default_name.endswith('.mp4'):
            filetypes = [("MP4 Video", "*.mp4"), ("All files", "*.*")]
        else:
            filetypes = [("ASCII Video", "*.ascvid"), ("All files", "*.*")]
        
        file_path = filedialog.asksaveasfilename(
            title="Save as",
            defaultextension=os.path.splitext(default_name)[1],
            initialfile=default_name,
            filetypes=filetypes
        )
        
        root.destroy()
        return file_path if file_path else None
    except Exception as e:
        print(f"Error: {e}")
        return None


def show_banner():
    """Banner"""
    if RICH_AVAILABLE:
        banner = f"""
╔═══════════════════════════════════════╗
║   🎬 VIDEO TO ASCII CONVERTER 🎨      ║
║   📐 Fixed Export • ⚡ FAST           ║
║   💾 Optimized • 60 FPS ({cpu_count()} cores)     ║
╚═══════════════════════════════════════╝
        """
        console.print(Panel(banner, style="bold cyan"))
    else:
        print("="*40)
        print("   VIDEO TO ASCII CONVERTER")
        print("="*40)


def get_video_info(video_path):
    """Get video info"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    info = {
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 0
    }
    cap.release()
    return info


def export_mode():
    """Export mode - FIXED"""
    if RICH_AVAILABLE:
        console.print("\n[bold cyan]🎥 Export to MP4 (FIXED!)[/bold cyan]\n")
    else:
        print("\n🎥 Export to MP4\n")
    
    video_file = select_video_file()
    if not video_file or not os.path.exists(video_file):
        return
    
    if not video_file.endswith('.ascvid'):
        print("❌ Select a .ascvid file!")
        return
    
    player = ASCIIVideoPlayer(video_file)
    info = player.get_info()
    
    if RICH_AVAILABLE:
        table = Table(title="📹 ASCII Video")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Resolution", f"{info['width']}x{info['height']} chars")
        table.add_row("FPS", f"{info['fps']:.1f}")
        table.add_row("Frames", str(info['total_frames']))
        console.print(table)
        
        console.print("\n[bold yellow]⚙️  Export Settings[/bold yellow]\n")
        
        font_size = int(Prompt.ask("🔤 Font size", default="12"))
        scale = float(Prompt.ask("📏 Scale", default="1.0"))
        workers = int(Prompt.ask(f"⚡ Workers", default=str(min(cpu_count(), 8))))
        
    else:
        print(f"\n📹 {info['width']}x{info['height']} chars")
        
        font_size = int(input("Font size [12]: ") or "12")
        scale = float(input("Scale [1.0]: ") or "1.0")
        workers = int(input(f"Workers [8]: ") or "8")
    
    output_path = select_output_location(f"ascii_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
    if not output_path:
        return
    
    start_time = time.time()
    success = player.export_to_video_fast(output_path, scale=scale, font_size=font_size, workers=workers)
    elapsed = time.time() - start_time
    
    if success:
        size = os.path.getsize(output_path) / (1024 * 1024)
        if RICH_AVAILABLE:
            console.print(f"\n[bold green]✅ Exported in {elapsed:.1f}s![/bold green]")
            console.print(f"[cyan]📁 {output_path}[/cyan]")
            console.print(f"[cyan]💾 {size:.2f} MB[/cyan]")
        else:
            print(f"\n✅ Done in {elapsed:.1f}s! ({size:.2f} MB)")
            print(f"📁 {output_path}")


def play_mode():
    """Play mode"""
    if RICH_AVAILABLE:
        console.print("\n[bold cyan]🎮 ASCII Video Player[/bold cyan]\n")
    
    video_file = select_video_file()
    if not video_file or not os.path.exists(video_file):
        return
    
    player = ASCIIVideoPlayer(video_file)
    
    try:
        info = player.get_info()
        
        if RICH_AVAILABLE:
            table = Table(title="📹 Video")
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="green")
            table.add_row("Resolution", f"{info['width']}x{info['height']} chars")
            table.add_row("FPS", f"{info['fps']:.1f}")
            table.add_row("Frames", str(info['total_frames']))
            table.add_row("Colors", "Yes ✨" if info['colored'] else "No")
            console.print(table)
        
        if PYGAME_AVAILABLE:
            if RICH_AVAILABLE:
                choice = Prompt.ask("\n[bold]Player[/bold]", choices=["1", "2"], default="1")
            else:
                choice = input("\n1. GUI  2. Terminal [1]: ") or "1"
            
            if choice == "1":
                player.play_gui()
            else:
                player.play_terminal()
        else:
            player.play_terminal()
            
    except Exception as e:
        print(f"\n❌ Error: {e}")


def convert_mode():
    """Convert mode"""
    show_banner()
    
    video_path = select_video_file()
    if not video_path or not os.path.exists(video_path):
        return
    
    if video_path.endswith('.ascvid'):
        print("❌ Already ASCII!")
        return
    
    info = get_video_info(video_path)
    if not info:
        return
    
    if RICH_AVAILABLE:
        table = Table(title="📹 Source")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Resolution", f"{info['width']}x{info['height']}")
        table.add_row("FPS", f"{info['fps']:.1f}")
        console.print(table)
        
        width = int(Prompt.ask("🔢 ASCII width", default="80"))
        colored = Confirm.ask("🎨 Colors?", default=True)
        charset = Prompt.ask("Choose charset (1-5)", default="2")
        workers = int(Prompt.ask(f"⚡ Workers", default="8"))
        
    else:
        print(f"\n📹 {info['width']}x{info['height']}")
        width = int(input("Width [80]: ") or "80")
        colored = input("Colors? (y/n) [y]: ").lower() != 'n'
        charset = "2"
        workers = 8
    
    converter = VideoToASCII(video_path, width=width, colored=colored, charset=charset, workers=workers)
    
    output_path = select_output_location(f"ascii_{datetime.now().strftime('%Y%m%d_%H%M%S')}.ascvid")
    if not output_path:
        return
    
    success = converter.save_as_ascvid(output_path)
    
    if success:
        size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"\n✅ Done! ({size:.2f} MB)")


def interactive_mode():
    """Interactive mode"""
    show_banner()
    
    if RICH_AVAILABLE:
        console.print("\n[bold]Choose:[/bold]")
        console.print("  1. 🎬 Convert")
        console.print("  2. ▶️  Play")
        console.print("  3. 🎥 Export to MP4")
        mode = Prompt.ask("Select", choices=["1", "2", "3"], default="1")
    else:
        print("\n1. Convert  2. Play  3. Export\n")
        mode = input("Choose [1]: ") or "1"
    
    if mode == "1":
        convert_mode()
    elif mode == "2":
        play_mode()
    else:
        export_mode()


def main():
    """Main"""
    parser = argparse.ArgumentParser(description='ASCII Video - FIXED Export')
    parser.add_argument('video', nargs='?')
    parser.add_argument('-w', '--width', type=int, default=80)
    parser.add_argument('--height', type=int)
    parser.add_argument('-c', '--color', action='store_true', default=True)
    parser.add_argument('--no-color', action='store_true')
    parser.add_argument('-s', '--charset', choices=['1', '2', '3', '4', '5'], default='2')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--play', action='store_true')
    parser.add_argument('--terminal', action='store_true')
    parser.add_argument('--export', action='store_true')
    parser.add_argument('--font-size', type=int, default=12)
    parser.add_argument('--scale', type=float, default=1.0)
    parser.add_argument('-o', '--output')
    
    args = parser.parse_args()
    
    if not args.video:
        interactive_mode()
        return
    
    # Export
    if args.export and args.video.endswith('.ascvid'):
        player = ASCIIVideoPlayer(args.video)
        output = args.output or f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        
        print(f"🎥 Exporting...")
        start = time.time()
        if player.export_to_video_fast(output, scale=args.scale, font_size=args.font_size, workers=args.workers):
            print(f"✅ Done in {time.time()-start:.1f}s → {output}")
        return
    
    # Play
    if args.play or args.video.endswith('.ascvid'):
        player = ASCIIVideoPlayer(args.video)
        if args.terminal:
            player.play_terminal()
        else:
            player.play_gui()
        return
    
    # Convert
    if not os.path.exists(args.video):
        print(f"❌ Not found!")
        return
    
    colored = not args.no_color
    converter = VideoToASCII(args.video, width=args.width, height=args.height,
                            colored=colored, charset=args.charset, workers=args.workers)
    
    output = args.output or f"ascii_{datetime.now().strftime('%Y%m%d_%H%M%S')}.ascvid"
    
    if converter.save_as_ascvid(output):
        print(f"✅ {output}")


if __name__ == "__main__":
    main()
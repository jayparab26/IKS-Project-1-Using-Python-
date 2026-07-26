import sys
import numpy as np
import pygame
import librosa

# --- Initialization ---
pygame.init()
pygame.mixer.init()

WIDTH, HEIGHT = 950, 680
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Acoustic Signal Analysis & Harmonic Ratio Study")

# Color Palette
BG_COLOR = (15, 17, 26)
CARD_BG = (24, 28, 43)
CARD_BORDER = (45, 52, 80)
ACCENT_BLUE = (99, 102, 241)
ACCENT_GREEN = (16, 185, 129)
ACCENT_GOLD = (245, 158, 11)
TEXT_MAIN = (240, 243, 246)
TEXT_MUTED = (148, 163, 184)

FONT_SMALL = pygame.font.SysFont("Arial", 14)
FONT_MED = pygame.font.SysFont("Arial", 17)
FONT_BOLD = pygame.font.SysFont("Arial", 17, bold=True)
TITLE_FONT = pygame.font.SysFont("Arial", 22, bold=True)

# Audio & Instrument Configuration
INSTRUMENTS = {
    'bell': {'name': 'Temple Bell', 'file': 'Templebell.mp3', 'icon': '🔔', 'key': '1'},
    'conch': {'name': 'Shankh (Conch)', 'file': 'shell.mp3', 'icon': '🐚', 'key': '2'},
    'sa': {'name': 'Tanpura (Sa)', 'file': 'tanpura.mp3', 'icon': '🪕', 'key': '3'}
}

# --- Load Audio Data & Sample Rates ---
audio_data = {}
sample_rates = {}

print("Loading audio files...")
for key, info in INSTRUMENTS.items():
    try:
        data, sr = librosa.load(info['file'], sr=None)
        audio_data[key] = data
        sample_rates[key] = sr
    except Exception as e:
        print(f"Error loading {info['file']}: {e}")
        sr = 22050
        audio_data[key] = np.sin(2 * np.pi * 440 * np.linspace(0, 1, sr))
        sample_rates[key] = sr

selected_instrument = 'bell'
is_playing = False

# --- Live Audio Processing ---
def get_live_audio_chunk(instrument, current_ms, chunk_samples=2048):
    data = audio_data[instrument]
    sr = sample_rates[instrument]
    
    start_sample = int((current_ms / 1000.0) * sr) % max(1, len(data) - chunk_samples)
    end_sample = min(start_sample + chunk_samples, len(data))
    
    chunk = data[start_sample:end_sample]
    if len(chunk) < chunk_samples:
        chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
        
    fft_spectrum = np.abs(np.fft.rfft(chunk))
    freqs = np.fft.rfftfreq(chunk_samples, 1.0 / sr)
    
    peak_idx = np.argmax(fft_spectrum)
    observed_f0 = freqs[peak_idx]
    
    return chunk, fft_spectrum, observed_f0

# --- UI Rendering Functions ---
def draw_card(surface, rect, is_active):
    color = ACCENT_BLUE if is_active else CARD_BORDER
    pygame.draw.rect(surface, CARD_BG, rect, border_radius=8)
    pygame.draw.rect(surface, color, rect, width=2 if is_active else 1, border_radius=8)

def draw_waveform(surface, raw_data, rect, title_text):
    draw_card(surface, rect, False)
    
    # Render Graph Label / Title Header inside panel
    lbl = FONT_SMALL.render(title_text, True, TEXT_MUTED)
    surface.blit(lbl, (rect.x + 12, rect.y + 8))
    
    # Grid lines
    for y in range(rect.y + 35, rect.y + rect.height, 25):
        pygame.draw.line(surface, (30, 36, 56), (rect.x + 10, y), (rect.x + rect.width - 10, y), 1)
        
    chunk = np.nan_to_num(raw_data[:1000])
    if len(chunk) == 0:
        return
    
    max_val = float(np.max(np.abs(chunk))) or 1.0
    step = (rect.width - 20) / len(chunk)
    points = []
    
    for i, val in enumerate(chunk):
        x = int(rect.x + 10 + (i * step))
        y = int(rect.y + 15 + (rect.height / 2) - (val / max_val * (rect.height / 2.6)))
        points.append((x, y))
        
    if len(points) > 1:
        pygame.draw.lines(surface, ACCENT_BLUE, False, points, 2)

def draw_refined_spectrum(surface, fft_spectrum, rect, title_text):
    draw_card(surface, rect, False)
    
    # Render Graph Label / Title Header inside panel
    lbl = FONT_SMALL.render(title_text, True, TEXT_MUTED)
    surface.blit(lbl, (rect.x + 12, rect.y + 8))
    
    fft_spectrum = np.nan_to_num(fft_spectrum)
    
    # Background Grid Lines
    for y in range(rect.y + 35, rect.y + rect.height, 25):
        pygame.draw.line(surface, (30, 36, 56), (rect.x + 10, y), (rect.x + rect.width - 10, y), 1)
        
    num_bars = 70
    if len(fft_spectrum) == 0:
        return
    
    chunk_size = len(fft_spectrum) // num_bars
    padding = 10
    usable_width = rect.width - (padding * 2)
    bar_width = usable_width / num_bars
    max_val = float(np.max(fft_spectrum)) or 1.0

    peak_points = []
    
    for i in range(num_bars):
        segment = fft_spectrum[i * chunk_size : (i + 1) * chunk_size]
        avg_val = float(np.mean(segment)) if len(segment) > 0 else 0.0
        norm_val = (avg_val / max_val)
        bar_height = norm_val * (rect.height - 35)
        
        x = int(rect.x + padding + (i * bar_width))
        y = int(rect.y + rect.height - 10 - bar_height)
        w = max(1, int(bar_width - 2))
        h = int(bar_height)
        
        color = (
            int(16 + norm_val * (99 - 16)),
            int(185 + norm_val * (102 - 185)),
            int(129 + norm_val * (241 - 129))
        )
        
        pygame.draw.rect(surface, color, (x, y, w, h), border_top_left_radius=2, border_top_right_radius=2)
        peak_points.append((x + w // 2, y))

    if len(peak_points) > 1:
        pygame.draw.lines(surface, ACCENT_GOLD, False, peak_points, 2)

# --- Main Event Loop ---
clock = pygame.time.Clock()
running = True

button_rects = {
    'bell': pygame.Rect(20, 560, 280, 85),
    'conch': pygame.Rect(335, 560, 280, 85),
    'sa': pygame.Rect(650, 560, 280, 85)
}

while running:
    screen.fill(BG_COLOR)
    mouse_pos = pygame.mouse.get_pos()
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for inst_key, rect in button_rects.items():
                if rect.collidepoint(mouse_pos):
                    selected_instrument = inst_key
                    if is_playing:
                        pygame.mixer.music.load(INSTRUMENTS[selected_instrument]['file'])
                        pygame.mixer.music.play(-1)
                        
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                is_playing = not is_playing
                if is_playing:
                    pygame.mixer.music.load(INSTRUMENTS[selected_instrument]['file'])
                    pygame.mixer.music.play(-1)
                else:
                    pygame.mixer.music.stop()
            elif event.key in [pygame.K_1, pygame.K_2, pygame.K_3]:
                key_map = {pygame.K_1: 'bell', pygame.K_2: 'conch', pygame.K_3: 'sa'}
                selected_instrument = key_map[event.key]
                if is_playing:
                    pygame.mixer.music.load(INSTRUMENTS[selected_instrument]['file'])
                    pygame.mixer.music.play(-1)

    current_time_ms = pygame.mixer.music.get_pos() if is_playing else 0
    if current_time_ms < 0:
        current_time_ms = 0

    audio_chunk, fft_data, observed_f0 = get_live_audio_chunk(selected_instrument, current_time_ms)

    # Render Title Header
    title = TITLE_FONT.render("Acoustic Signal Analysis & Harmonic Ratio Study", True, TEXT_MAIN)
    screen.blit(title, (20, 15))
    
    status_color = ACCENT_GREEN if is_playing else ACCENT_BLUE
    status_str = "● PLAYING [Press SPACE to Stop]" if is_playing else "○ STOPPED [Press SPACE to Play]"
    status_lbl = FONT_BOLD.render(status_str, True, status_color)
    screen.blit(status_lbl, (630, 18))

    # Render Labeled Graphs
    draw_waveform(screen, audio_chunk, pygame.Rect(20, 55, 910, 130), "WAVEFORM ANALYSIS: Time-Domain Signal x(t)")
    draw_refined_spectrum(screen, fft_data, pygame.Rect(20, 200, 910, 170), "FREQUENCY SPECTRUM: Fast Fourier Transform X(f)")

    # Render Harmonic Ratio Values Card
    ratio_card = pygame.Rect(20, 385, 910, 125)
    draw_card(screen, ratio_card, False)
    
    lbl_f0 = FONT_BOLD.render(f"Observed Base Frequency (Sa): {observed_f0:.1f} Hz", True, TEXT_MAIN)
    lbl_f1 = FONT_MED.render(f"Calculated Pa Ratio (1.5x):  {(observed_f0 * 1.5):.1f} Hz", True, ACCENT_BLUE)
    lbl_f2 = FONT_MED.render(f"Calculated Taar Sa Ratio (2.0x): {(observed_f0 * 2.0):.1f} Hz", True, ACCENT_GREEN)
    
    screen.blit(lbl_f0, (40, 400))
    screen.blit(lbl_f1, (40, 435))
    screen.blit(lbl_f2, (40, 465))

    # Render Instrument Selector Cards
    lbl_select = FONT_BOLD.render("SELECT INSTRUMENT / AUDIO SOURCE:", True, TEXT_MUTED)
    screen.blit(lbl_select, (20, 530))

    for inst_key, rect in button_rects.items():
        is_selected = (inst_key == selected_instrument)
        draw_card(screen, rect, is_selected)
        
        img_rect = pygame.Rect(rect.x + 12, rect.y + 12, 60, 60)
        pygame.draw.rect(screen, (35, 42, 66), img_rect, border_radius=6)
        
        icon = pygame.font.SysFont("Segoe UI Emoji", 28).render(INSTRUMENTS[inst_key]['icon'], True, TEXT_MAIN)
        screen.blit(icon, (img_rect.x + 12, img_rect.y + 10))

        name_txt = FONT_BOLD.render(INSTRUMENTS[inst_key]['name'], True, TEXT_MAIN if is_selected else TEXT_MUTED)
        key_txt = FONT_SMALL.render(f"Press [{INSTRUMENTS[inst_key]['key']}] or Click", True, ACCENT_GOLD if is_selected else TEXT_MUTED)
        
        screen.blit(name_txt, (rect.x + 85, rect.y + 20))
        screen.blit(key_txt, (rect.x + 85, rect.y + 48))

    pygame.display.flip()
    clock.tick(30)

pygame.quit()
sys.exit()
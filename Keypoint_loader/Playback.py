from PySide6.QtCore import QObject, QTimer, Signal


class PlaybackController(QObject):
    frame_changed = Signal(int)

    def __init__(self, max_frames):
        super().__init__()
        self.max_frames = max_frames
        self.current_frame = 0
        self.speed = 1.0

        self.timer = QTimer()
        self.timer.timeout.connect(self._tick)
        self.base_interval = 33  # ~30 fps base
        self._update_timer_interval()
    
    def _update_timer_interval(self):
        """Update timer interval based on speed (inverse relationship)"""
        if self.speed > 0:
            interval = int(self.base_interval / self.speed)
            self.timer.setInterval(max(1, interval))
    
    def set_speed(self, speed):
        """Set playback speed. 1.0 = normal, 0.5 = half speed, 2.0 = double speed"""
        if speed > 0:
            self.speed = speed
            self._update_timer_interval()

    def play(self):
        self.timer.start()

    def pause(self):
        self.timer.stop()

    def _tick(self):
        if self.max_frames > 0:
            self.current_frame = (self.current_frame + 1) % self.max_frames
            self.frame_changed.emit(self.current_frame)
    
    def next_frame(self):
        """Go to the next frame"""
        if self.max_frames > 0:
            self.current_frame = (self.current_frame + 1) % self.max_frames
            self.frame_changed.emit(self.current_frame)
    
    def previous_frame(self):
        """Go to the previous frame"""
        if self.max_frames > 0:
            self.current_frame = (self.current_frame - 1) % self.max_frames
            self.frame_changed.emit(self.current_frame)
    
    def set_frame(self, frame_index):
        """Jump to a specific frame"""
        if 0 <= frame_index < self.max_frames:
            self.current_frame = frame_index
            self.frame_changed.emit(self.current_frame)
    
    def get_current_frame(self):
        """Get the current frame index"""
        return self.current_frame
    
    #Update max frames when new data is loaded
    def update_max_frames(self, max_frames):
        self.max_frames = max(max_frames, 0)
        self.current_frame = 0

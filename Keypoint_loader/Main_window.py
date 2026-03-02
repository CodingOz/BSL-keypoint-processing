from PySide6.QtWidgets import (
    QMainWindow, QWidget, QPushButton,
    QLabel, QVBoxLayout, QHBoxLayout, QListWidget,
    QListWidgetItem, QFileDialog, QCheckBox, QSlider, QSpinBox, QDoubleSpinBox, QComboBox
)
from PySide6.QtCore import Qt, Signal
import os


class MainWindow(QMainWindow):
    cluster_filters = Signal(dict)
    palm_mode_changed = Signal(str)

    def __init__(self, graph_view, data_model=None):
        super().__init__()
        self.setWindowTitle("Keypoint Viewer")
        self.data_model = data_model
        self.current_directory = None

        # control buttons
        self.play_button = QPushButton("Play")
        self.pause_button = QPushButton("Pause")
        self.frame_label = QLabel("Frame: 0")
        self.browse_button = QPushButton("Browse Directory")
        
        # frame navigation buttons
        self.prev_button = QPushButton("< Previous")
        self.next_button = QPushButton("Next >")
        
        # frame selection spinbox
        self.frame_spinbox = QSpinBox()
        self.frame_spinbox.setMinimum(0)
        self.frame_spinbox.setMaximum(0)
        self.frame_spinbox.setValue(0)

        # cluster visibility checkboxes
        self.show_left_cb = QCheckBox("Show Left")
        self.show_right_cb = QCheckBox("Show Right")
        self.show_pose_cb = QCheckBox("Show Pose")
        self.show_left_cb.setChecked(True)
        self.show_right_cb.setChecked(True)
        self.show_pose_cb.setChecked(True)
        
        # palm visibility dropdown
        self.palm_mode_combo = QComboBox()
        self.palm_mode_combo.addItems([
            "None",
            "Real Palms",
            "Kalman Estimated",
            "Cubic Spline",
            "PCHIP"
        ])
        self.palm_mode_combo.setCurrentText("None")

        # speed control
        self.speed_label = QLabel("Speed: 1.0x")
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setMinimum(10)  # 0.1x speed
        self.speed_slider.setMaximum(400)  # 4.0x speed
        self.speed_slider.setValue(100)  # 1.0x default
        self.speed_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.speed_slider.setTickInterval(50)

        # create multiple rows of controls
        controls_row1 = QHBoxLayout()
        controls_row1.addWidget(self.play_button)
        controls_row1.addWidget(self.pause_button)
        controls_row1.addWidget(self.prev_button)
        controls_row1.addWidget(self.next_button)
        controls_row1.addWidget(self.frame_label)
        controls_row1.addWidget(QLabel("Go to frame:"))
        controls_row1.addWidget(self.frame_spinbox)
        controls_row1.addWidget(self.browse_button)
        controls_row1.addStretch()

        controls_row2 = QHBoxLayout()
        controls_row2.addWidget(QLabel("Clusters:"))
        controls_row2.addWidget(self.show_left_cb)
        controls_row2.addWidget(self.show_right_cb)
        controls_row2.addWidget(self.show_pose_cb)
        controls_row2.addStretch()

        controls_row3 = QHBoxLayout()
        controls_row3.addWidget(QLabel("Palm Centers:"))
        controls_row3.addWidget(self.palm_mode_combo)
        controls_row3.addStretch()

        controls_row4 = QHBoxLayout()
        controls_row4.addWidget(self.speed_label)
        controls_row4.addWidget(self.speed_slider)
        controls_row4.addStretch()

        # stack all control rows vertically
        controls = QVBoxLayout()
        controls.addLayout(controls_row1)
        controls.addLayout(controls_row2)
        controls.addLayout(controls_row3)
        controls.addLayout(controls_row4)

        # file list panel on the right
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        
        file_panel = QVBoxLayout()
        file_panel.addWidget(QLabel("JSON Files:"))
        file_panel.addWidget(self.file_list)

        # graph on left, controls below, file list on right
        main_layout = QHBoxLayout()
        
        left_layout = QVBoxLayout()
        left_layout.addWidget(graph_view)
        left_layout.addLayout(controls)
        
        main_layout.addLayout(left_layout, 3)  # 3 for graph
        main_layout.addLayout(file_panel, 1)   # 1 for file list

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)
        
        self.file_list.itemChanged.connect(self._on_file_selection_changed)
        self.browse_button.clicked.connect(self._on_browse_directory)
        
        self.show_left_cb.stateChanged.connect(self._on_cluster_filters)
        self.show_right_cb.stateChanged.connect(self._on_cluster_filters)
        self.show_pose_cb.stateChanged.connect(self._on_cluster_filters)
        
        self.palm_mode_combo.currentTextChanged.connect(self._on_palm_mode_changed)

    def _on_browse_directory(self):
        """open directory dialog and populate file list"""
        directory = QFileDialog.getExistingDirectory(self, "Select Directory with JSON Files")
        if directory:
            self.set_directory(directory)

    def set_directory(self, directory):
        """populate file list with files from directory"""
        import os
        self.current_directory = directory
        self.file_list.clear()
        
        json_files = [f for f in os.listdir(directory) if f.endswith('.json')]
        json_files.sort()
        
        for file in json_files:
            item = QListWidgetItem(file)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
            self.file_list.addItem(item)

    def _on_file_selection_changed(self):
        """handle file selection/deselection changes"""
        if not self.data_model or not self.current_directory:
            return
        
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            filepath = os.path.join(self.current_directory, item.text())
            
            is_checked = item.checkState() == Qt.CheckState.Checked
            currently_loaded = item.text() in self.data_model.get_loaded_files()
            
            if is_checked and not currently_loaded:
                self.data_model.load_from_json(filepath, add=True)
            elif not is_checked and currently_loaded:
                self.data_model.load_from_json(filepath, add=False)

    def _on_cluster_filters(self):
        """emit signal when cluster visibility changes"""
        cluster_filter = {
            "left": self.show_left_cb.isChecked(),
            "right": self.show_right_cb.isChecked(),
            "pose": self.show_pose_cb.isChecked()
        }
        self.cluster_filters.emit(cluster_filter)

    def set_playback_controller(self, playback):
        """provide a reference to the PlaybackController so the UI can control speed."""
        self.playback = playback
        # connect UI buttons
        self.play_button.clicked.connect(playback.play)
        self.pause_button.clicked.connect(playback.pause)
        self.prev_button.clicked.connect(playback.previous_frame)
        self.next_button.clicked.connect(playback.next_frame)
        # connect speed slider
        self.speed_slider.valueChanged.connect(self._on_speed_slider_changed)
        # connect frame spinbox
        self.frame_spinbox.valueChanged.connect(self._on_frame_spinbox_changed)
        # spinbox max value
        self.frame_spinbox.setMaximum(max(playback.max_frames - 1, 0))

    def _on_speed_slider_changed(self, value):
        # slider ranges from 10..400 representing 0.1x..4.0x
        speed = value / 100.0
        self.speed_label.setText(f"Speed: {speed:.2f}x")
        if hasattr(self, 'playback') and self.playback is not None:
            self.playback.set_speed(speed)
    
    def _on_frame_spinbox_changed(self, value):
        """handle frame spinbox value changes"""
        if hasattr(self, 'playback') and self.playback is not None:
            self.playback.set_frame(value)
    
    def update_frame_display(self, frame_index):
        """update frame display label and spinbox without triggering spinbox signal"""
        self.frame_label.setText(f"Frame: {frame_index}")
        # block signals to prevent re-triggering frame change
        self.frame_spinbox.blockSignals(True)
        self.frame_spinbox.setValue(frame_index)
        self.frame_spinbox.blockSignals(False)
    
    def _on_palm_mode_changed(self, mode_text):
        """handle palm mode dropdown selection change"""
        # map dropdown text to internal mode names
        mode_map = {
            "None": None,
            "Real Palms": "real",
            "Kalman Estimated": "kalman",
            "Cubic Spline": "cubic_spline",
            "PCHIP": "pchip"
        }
        
        palm_mode = mode_map.get(mode_text, None)
        
        # load the specified palm type for all loaded files if not 'None'
        if palm_mode and self.data_model and self.current_directory:
            for i in range(self.file_list.count()):
                item = self.file_list.item(i)
                if item.checkState() == Qt.CheckState.Checked:
                    filepath = os.path.join(self.current_directory, item.text())
                    try:
                        self.data_model.load_palm_centers_by_type(
                            filepath, 
                            item.text(), 
                            palm_mode,
                            self.data_model.loaded_files.get(item.text(), 'gray')
                        )
                    except Exception as e:
                        print(f"Error loading {mode_text} palms for {item.text()}: {e}")
        
        self.palm_mode_changed.emit(palm_mode)
        if hasattr(self, 'data_model'):
            self.data_model.set_palm_mode(palm_mode)
    
    def get_directory_from_user(self):
        """show directory picker on startup"""
        directory = QFileDialog.getExistingDirectory(self, "Select Directory with JSON Files")
        if directory:
            self.set_directory(directory)
        return directory

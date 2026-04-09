import json
import os
from copy import deepcopy
import sys
 
 
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QLabel,
    QVBoxLayout, QHBoxLayout, QFileDialog, QMessageBox, QProgressBar,
    QSpinBox, QSlider, QCheckBox
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QKeySequence, QShortcut
 
from Keypoint_loader.Data_model import DataModel
from Keypoint_loader.Playback import PlaybackController
from Keypoint_loader.Graph_view import GraphView
 
 
def discover_files(root_dir):
    """Return a list of (letter, filepath) tuples sorted by letter then name."""
    files = []
    if not os.path.isdir(root_dir):
        return files
    for letter in sorted(os.listdir(root_dir)):
        letter_dir = os.path.join(root_dir, letter)
        if not os.path.isdir(letter_dir):
            continue
        for fname in sorted(os.listdir(letter_dir)):
            if fname.endswith(".json"):
                files.append((letter, os.path.join(letter_dir, fname)))
    return files


class HandDominanceNormaliser:
    def swapToRightDominant(self, filepath, show_logs=False):
        with open(filepath, "r") as f:
            data = json.load(f)

        for frame in data["frames"]:
            hands = frame["hands"]
            hands["left"], hands["right"] = hands["right"], hands["left"]
            for side in ("left", "right"):
                for lm in hands[side]:
                    lm["cluster_id"] = 1 if lm["cluster_id"] == 0 else 0
                    lm["x"] = 1.0 - lm["x"]

        data["metadata"]["swapped_dominance"] = True

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        if show_logs:
            print(f"[swapToRightDominant] {filepath} — swapped in place, swapped_dominance=True")
    
    def matchSwappedCorpus(self, preswapped_corpus, target_corpus, show_logs=False):
        '''the the source vidoes of V1 and V2 file are identical we can assume the hand forminance if too
        thus if one copus have been manualy labled the the other one can mimic its operations without
        the need for more manual labling
        
        takes
            preswapped_corpus: already labled and swapped corpus
            target_corpus: corpus to be labled and swapped based on preswapped_corpus
        '''
        
        preswapped_items = discover_files(preswapped_corpus)
        target_items = discover_files(target_corpus)
        
        if len(preswapped_items) != len(target_items):
            print("error: two copuses must have equivilent structure")
        
        for index, preswapped_item, in enumerate(preswapped_items):
            target_item = target_items[index]
            with open(preswapped_item[1], "r") as f:
                preswapped_data = json.load(f)
            if preswapped_data['metadata'].get("swapped_dominance") == True:
                self.swapToRightDominant(target_item[1], show_logs=show_logs)
                

def file_already_labelled(filepath):
    """Check whether a file already has a dominance label in its metadata."""
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
        meta = data.get("metadata", {})
        return ("dominance_label" in meta) or ("swapped_dominance" in meta)
    except Exception:
        return False
    

class SessionLog:
    """Persists labelling decisions so the session can be resumed."""
 
    def __init__(self, root_dir):
        self.log_path = os.path.join(root_dir, "dominance_labels.json")
        self.data = {}
        if os.path.exists(self.log_path):
            with open(self.log_path, "r") as f:
                self.data = json.load(f)
 
    def is_labelled(self, filepath):
        return os.path.basename(filepath) in self.data
 
    def record(self, filepath, label):
        """label is 'right' or 'left'"""
        self.data[os.path.basename(filepath)] = {
            "path": filepath,
            "label": label
        }
        self._save()
 
    def _save(self):
        with open(self.log_path, "w") as f:
            json.dump(self.data, f, indent=2)
 
 
 
class DominanceLabellingWindow(QMainWindow):
    """Single-file viewer with dominance labelling controls."""
 
    def __init__(self, root_dir=None):
        super().__init__()
        self.setWindowTitle("Hand Dominance Labeller")
        self.resize(1100, 750)
 
        # state
        self.root_dir = root_dir
        self.file_queue = []
        self.current_idx = -1
        self.normaliser = HandDominanceNormaliser()
        self.session_log = None
 
        self.data_model = DataModel()
        self.graph_view = GraphView()
        self.playback = PlaybackController(0)
 
        self.playback.frame_changed.connect(
            lambda i: self.graph_view.update_frame(self.data_model.get_frame(i))
        )
        self.playback.frame_changed.connect(self._update_frame_label)
 
        self._build_ui()
        self._bind_shortcuts()
 
        if self.root_dir:
            self._initialise_queue()
        else:
            self._pick_directory()
            
    def _build_ui(self):
        self.progress_label = QLabel("No directory loaded")
        self.progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = QFont()
        font.setPointSize(11)
        self.progress_label.setFont(font)
 
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
 
        info_row = QVBoxLayout()
        info_row.addWidget(self.progress_label)
        info_row.addWidget(self.progress_bar)
 
        self.play_btn = QPushButton("Play")
        self.pause_btn = QPushButton("Pause")
        self.prev_btn = QPushButton("Prev Frame")
        self.next_btn = QPushButton("Next Frame")
        self.frame_label = QLabel("Frame: 0")
 
        self.frame_spinbox = QSpinBox()
        self.frame_spinbox.setMinimum(0)
        self.frame_spinbox.setMaximum(0)
 
        self.speed_label = QLabel("Speed: 1.0x")
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setMinimum(10)
        self.speed_slider.setMaximum(400)
        self.speed_slider.setValue(100)
        self.speed_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.speed_slider.setTickInterval(50)
 
        # cluster checkboxes
        self.show_left_cb = QCheckBox("Left Hand")
        self.show_right_cb = QCheckBox("Right Hand")
        self.show_pose_cb = QCheckBox("Pose")
        self.show_left_cb.setChecked(True)
        self.show_right_cb.setChecked(True)
        self.show_pose_cb.setChecked(True)
 
        playback_row = QHBoxLayout()
        playback_row.addWidget(self.play_btn)
        playback_row.addWidget(self.pause_btn)
        playback_row.addWidget(self.prev_btn)
        playback_row.addWidget(self.next_btn)
        playback_row.addWidget(self.frame_label)
        playback_row.addWidget(QLabel("Go to:"))
        playback_row.addWidget(self.frame_spinbox)
        playback_row.addStretch()
 
        options_row = QHBoxLayout()
        options_row.addWidget(QLabel("Show:"))
        options_row.addWidget(self.show_left_cb)
        options_row.addWidget(self.show_right_cb)
        options_row.addWidget(self.show_pose_cb)
        options_row.addWidget(self.speed_label)
        options_row.addWidget(self.speed_slider)
        options_row.addStretch()
 
        self.right_btn = QPushButton(" Right Dominant  (R)")
        self.left_btn = QPushButton(" Left Dominant — swap  (L)")
        self.skip_btn = QPushButton(" Skip  (S)")
 
        for btn in (self.right_btn, self.left_btn, self.skip_btn):
            btn.setMinimumHeight(48)
            btn.setFont(QFont("", 12))
 
        self.right_btn.setStyleSheet("background-color: #2e7d32; color: white; border-radius: 6px;")
        self.left_btn.setStyleSheet("background-color: #c62828; color: white; border-radius: 6px;")
        self.skip_btn.setStyleSheet("background-color: #555; color: white; border-radius: 6px;")
 
        dominance_row = QHBoxLayout()
        dominance_row.addWidget(self.right_btn)
        dominance_row.addWidget(self.left_btn)
        dominance_row.addWidget(self.skip_btn)
 
        layout = QVBoxLayout()
        layout.addLayout(info_row)
        layout.addWidget(self.graph_view, stretch=1)
        layout.addLayout(playback_row)
        layout.addLayout(options_row)
        layout.addLayout(dominance_row)
 
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
 
        self.play_btn.clicked.connect(self.playback.play)
        self.pause_btn.clicked.connect(self.playback.pause)
        self.prev_btn.clicked.connect(self.playback.previous_frame)
        self.next_btn.clicked.connect(self.playback.next_frame)
        self.frame_spinbox.valueChanged.connect(self._on_spinbox)
        self.speed_slider.valueChanged.connect(self._on_speed)
 
        self.show_left_cb.stateChanged.connect(self._on_cluster_filter)
        self.show_right_cb.stateChanged.connect(self._on_cluster_filter)
        self.show_pose_cb.stateChanged.connect(self._on_cluster_filter)
 
        self.right_btn.clicked.connect(self._label_right)
        self.left_btn.clicked.connect(self._label_left)
        self.skip_btn.clicked.connect(self._skip)
 
    def _bind_shortcuts(self):
        QShortcut(QKeySequence("R"), self).activated.connect(self._label_right)
        QShortcut(QKeySequence("L"), self).activated.connect(self._label_left)
        QShortcut(QKeySequence("S"), self).activated.connect(self._skip)
        QShortcut(QKeySequence("Space"), self).activated.connect(self._toggle_play)
 
 
    def _pick_directory(self):
        d = QFileDialog.getExistingDirectory(
            self, "Select the Unprocessed_keypoints root directory"
        )
        if d:
            self.root_dir = d
            self._initialise_queue()
        else:
            QMessageBox.warning(self, "No directory", "No directory selected. Exiting.")
            QApplication.quit()
 
    def _initialise_queue(self):
        self.session_log = SessionLog(self.root_dir)
        all_files = discover_files(self.root_dir)
 
        # filter out already-labelled files (from session log or file metadata)
        self.file_queue = [
            (letter, fp) for letter, fp in all_files
            if not self.session_log.is_labelled(fp) and not file_already_labelled(fp)
        ]
        self.total_files = len(all_files)
        self.labelled_count = self.total_files - len(self.file_queue)
 
        if not self.file_queue:
            QMessageBox.information(self, "Done", "All files have already been labelled!")
            return
 
        self.progress_bar.setMaximum(self.total_files)
        self._load_file(0)
 
    def _load_file(self, idx):
        """Load a single file into the viewer."""
        if idx < 0 or idx >= len(self.file_queue):
            self._finish()
            return
 
        self.current_idx = idx
        letter, filepath = self.file_queue[idx]
 
        # stop any running playback
        self.playback.pause()
 
        # clear previous data
        self.data_model = DataModel()
        self.data_model.load_from_json(filepath, add=True, load_palms=False)
 
        # reset playback
        frame_count = self.data_model.frame_count()
        self.playback.update_max_frames(frame_count)
        self.playback.set_frame(0)
 
        self.frame_spinbox.setMaximum(max(frame_count - 1, 0))
        self.frame_spinbox.setValue(0)
 
        # render first frame
        self.graph_view.update_frame(self.data_model.get_frame(0))
        self._update_frame_label(0)
 
        # update progress
        done = self.labelled_count + idx
        remaining = len(self.file_queue) - idx
        filename = os.path.basename(filepath)
        self.progress_label.setText(
            f"Letter: {letter}   |   File: {filename}   |   "
            f"{done} done · {remaining} remaining"
        )
        self.progress_bar.setValue(done)
 
    def _reload_current_file(self):
        """Reload the current file (used after swap to show the mirrored result)."""
        if self.current_idx < 0:
            return
        letter, filepath = self.file_queue[self.current_idx]
 
        self.data_model = DataModel()
        self.data_model.load_from_json(filepath, add=True, load_palms=False)
 
        frame_count = self.data_model.frame_count()
        self.playback.update_max_frames(frame_count)
        self.playback.set_frame(0)
 
        self.frame_spinbox.setMaximum(max(frame_count - 1, 0))
        self.frame_spinbox.setValue(0)
        self.graph_view.update_frame(self.data_model.get_frame(0))

    def _label_right(self):
        if self.current_idx < 0 or self.current_idx >= len(self.file_queue):
            return
        letter, filepath = self.file_queue[self.current_idx]
 
        self._stamp_metadata(filepath, "right", swapped=False)
        self.session_log.record(filepath, "right")
 
        self.labelled_count += 1
        self._load_file(self.current_idx + 1)
 
    def _label_left(self):
        if self.current_idx < 0 or self.current_idx >= len(self.file_queue):
            return
        letter, filepath = self.file_queue[self.current_idx]
 
        self.normaliser.swapToRightDominant(filepath, show_logs=True)
 
        self._stamp_metadata(filepath, "left", swapped=True)
        self.session_log.record(filepath, "left")
 
        self._reload_current_file()
 
        self.labelled_count += 1
        self._load_file(self.current_idx + 1)
 
    def _skip(self):
        if self.current_idx < 0 or self.current_idx >= len(self.file_queue):
            return
        self._load_file(self.current_idx + 1)
 
    def _stamp_metadata(self, filepath, label, swapped):
        """Write dominance_label into the JSON metadata for future reference."""
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
            if "metadata" not in data:
                data["metadata"] = {}
            data["metadata"]["dominance_label"] = label
            if not swapped:
                data["metadata"]["swapped_dominance"] = False
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: could not stamp metadata on {filepath}: {e}")
 
    def _finish(self):
        self.playback.pause()
        self.progress_label.setText("All files labelled!")
        self.progress_bar.setValue(self.total_files)
        QMessageBox.information(
            self, "Complete",
            f"All {self.total_files} files have been labelled.\n"
            f"Session log saved to: {self.session_log.log_path}"
        )
        
    def _update_frame_label(self, idx):
        self.frame_label.setText(f"Frame: {idx}")
        self.frame_spinbox.blockSignals(True)
        self.frame_spinbox.setValue(idx)
        self.frame_spinbox.blockSignals(False)
 
    def _on_spinbox(self, value):
        self.playback.set_frame(value)
 
    def _on_speed(self, value):
        speed = value / 100.0
        self.speed_label.setText(f"Speed: {speed:.2f}x")
        self.playback.set_speed(speed)
 
    def _toggle_play(self):
        if self.playback.timer.isActive():
            self.playback.pause()
        else:
            self.playback.play()
 
    def _on_cluster_filter(self):
        cluster_filter = {
            "left": self.show_left_cb.isChecked(),
            "right": self.show_right_cb.isChecked(),
            "pose": self.show_pose_cb.isChecked()
        }
        self.graph_view.set_cluster_filter(cluster_filter)
        if self.current_idx >= 0:
            self.graph_view.update_frame(
                self.data_model.get_frame(self.playback.current_frame))
                
                
if __name__ == "__main__":
    root_dir = r"C:\Users\Oscar Strong\Documents\GitHub\BSL-keypoint-processing\Unproccessed_keypoints_V1"
    
    target = r"C:\Users\Oscar Strong\Documents\GitHub\BSL-keypoint-processing\Unproccessed_keypoints_V2"
    
    dom_hander = HandDominanceNormaliser()
    dom_hander.matchSwappedCorpus(root_dir, target, show_logs=True)
    '''
    app = QApplication(sys.argv)
    window = DominanceLabellingWindow(root_dir=root_dir)
    window.show()
    sys.exit(app.exec())'''

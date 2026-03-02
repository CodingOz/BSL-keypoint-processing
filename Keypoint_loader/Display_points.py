import sys
from PySide6.QtWidgets import QApplication

from Data_model import DataModel
from Playback import PlaybackController
from Graph_view import GraphView
from Main_window import MainWindow


def main():
    app = QApplication(sys.argv)

    data = DataModel()
    graph = GraphView()
    window = MainWindow(graph, data_model=data)

    # directory passed as command-line argument
    if len(sys.argv) > 1:
        directory = sys.argv[1]
        window.set_directory(directory)
    else:
        window.get_directory_from_user()

    # update playback controller based on loaded files
    playback = PlaybackController(data.frame_count())

    playback.frame_changed.connect(
        lambda i: graph.update_frame(data.get_frame(i))
    )
    playback.frame_changed.connect(
        lambda i: window.update_frame_display(i)
    )

    # connect playback to UI via MainWindow helper
    window.set_playback_controller(playback)
    
    # update playback and display when files are loaded
    def update_playback_on_file_change():
        playback.update_max_frames(data.frame_count())
        window.frame_spinbox.setMaximum(max(data.frame_count() - 1, 0))
        
        # update graph with current frame (frame 0)
        graph.update_frame(data.get_frame(0))
        window.update_frame_display(0)
    
    window.file_list.itemChanged.connect(update_playback_on_file_change)
    
    # connect cluster filter changes
    window.cluster_filters.connect(graph.set_cluster_filter)
    window.cluster_filters.connect(
        lambda _: graph.update_frame(data.get_frame(playback.current_frame))
    )
    
    # connect palm mode changes
    window.palm_mode_changed.connect(graph.set_palm_mode)
    window.palm_mode_changed.connect(
        lambda _: graph.update_frame(data.get_frame(playback.current_frame))
    )

    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
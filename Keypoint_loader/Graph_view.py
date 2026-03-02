import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer, QPointF
import time
from PySide6.QtWidgets import QToolTip
from PySide6.QtGui import QCursor


class GraphView(pg.GraphicsLayoutWidget):

    def __init__(self):
        super().__init__()
        self.plot = self.addPlot()
        self.plot.setAspectLocked(True)
        # normalized MediaPipe coordinate ranges by default
        self.plot.setXRange(0, 1)
        self.plot.setYRange(0, 1)

        # invert the view's y axis for image coordinates
        self.plot.getViewBox().invertY(True)

        # label axes so the coordinate system is clear to users
        self.plot.setLabel('bottom', 'X (0 left → 1 right)', units='')
        self.plot.setLabel('left', 'Y (0 top → 1 bottom)', units='')
        # show tick marks with 0..1 range
        self.plot.getAxis('bottom').setTicks([[(0, '0'), (0.5, '0.5'), (1, '1')]])
        self.plot.getAxis('left').setTicks([[(0, '0'), (0.5, '0.5'), (1, '1')]])
        
        self.plot.showGrid(x=True, y=True, alpha=0.3)

        self.scatter_items = {}  # {(filename, cluster): ScatterPlotItem}
        self.point_metadata = {}  # {(filename, cluster, point_idx): metadata_dict}
        self.scatter_points = {}  # {(filename, cluster): numpy array of points}
        self.palm_scatter_items = {}  # {(filename, palm_type): ScatterPlotItem}
        self.palm_metadata = {}  # {(filename, palm_type, point_idx): metadata_dict}
        self.palm_points = {}  # {(filename, palm_type): numpy array of points}
        self.cluster_filter = {"left": True, "right": True, "pose": True}
        
        # track mouse position for continuous hover checking
        self.last_mouse_pos = None
        self.current_hover_key = None  # track which point is being hovered
        self.hover_timer = QTimer()
        self.hover_timer.timeout.connect(self._check_hover)
        self.hover_timer.start(100)  # check every 100ms
        self.last_tooltip_refresh = 0.0
        self.tooltip_refresh_interval = 1.0  # seconds
        self.palm_mode = None  # None, 'real', or 'estimated'
        
        # enable mouse tracking for hover
        self.plot.scene().sigMouseMoved.connect(self._on_mouse_move)
        self.setMouseTracking(True)

    def set_cluster_filter(self, cluster_filter):
        """update which clusters to display"""
        self.cluster_filter = cluster_filter
    
    def set_palm_mode(self, palm_mode):
        """set palm display mode: None, 'real', or 'estimated'"""
        self.palm_mode = palm_mode

    def _on_mouse_move(self, pos):
        """handle mouse movement - store position for hover checking"""
        self.last_mouse_pos = pos
    
    def _check_hover(self):
        """periodically check if mouse is hovering over a point"""
        if self.last_mouse_pos is None:
            # mouse not over widget
            if self.current_hover_key is not None:
                QToolTip.hideText()
                self.current_hover_key = None
            return
        
        try:
            view_pos = self.plot.getViewBox().mapSceneToView(self.last_mouse_pos)
        except:
            return
        
        # iterate through scatter items to find nearby points
        hover_distance = 0.02  # Normalized distance threshold for hover
        
        # check regular landmark points first
        for key, points_array in self.scatter_points.items():
            if points_array is None or len(points_array) == 0:
                continue
            
            for idx, point in enumerate(points_array):
                dx = abs(point[0] - view_pos.x())
                dy = abs(point[1] - view_pos.y())
                distance = (dx**2 + dy**2)**0.5
                
                if distance < hover_distance:
                    # found a nearby point
                    new_hover_key = (key[0], key[1], idx)
                    meta_key = new_hover_key
                    
                    if meta_key in self.point_metadata:
                        meta = self.point_metadata[meta_key]
                        tooltip_text = (
                            f"File: {key[0]}\n"
                            f"Cluster: {key[1]}\n"
                            f"Cluster ID: {meta.get('cluster_id')}\n"
                            f"Landmark ID: {meta.get('landmark_id')}\n"
                            f"X: {meta.get('x'):.4f}\n"
                            f"Y: {meta.get('y'):.4f}\n"
                            f"Z: {meta.get('z'):.4f}"
                        )
                        now = time.time()
                        # show tooltip immediately when changing point, otherwise
                        # refresh at a lower frequency to avoid flicker.
                        if new_hover_key != self.current_hover_key or (now - self.last_tooltip_refresh) > self.tooltip_refresh_interval:
                            QToolTip.showText(QCursor.pos(), tooltip_text, self)
                            self.last_tooltip_refresh = now
                            self.current_hover_key = new_hover_key
                    return
        
        # check palm points
        for key, points_array in self.palm_points.items():
            if points_array is None or len(points_array) == 0:
                continue
            
            for idx, point in enumerate(points_array):
                dx = abs(point[0] - view_pos.x())
                dy = abs(point[1] - view_pos.y())
                distance = (dx**2 + dy**2)**0.5
                
                if distance < hover_distance:
                    # found a nearby palm point
                    new_hover_key = (key[0], key[1], idx)
                    meta_key = new_hover_key
                    
                    if meta_key in self.palm_metadata:
                        meta = self.palm_metadata[meta_key]
                        tooltip_text = (
                            f"File: {key[0]}\n"
                            f"Palm Type: {key[1]}\n"
                            f"Hand Side: {meta.get('side')}\n"
                            f"X: {meta.get('x'):.4f}\n"
                            f"Y: {meta.get('y'):.4f}"
                        )
                        if 'estimated' in meta:
                            tooltip_text += f"\nEstimated: {meta.get('estimated')}"
                        
                        now = time.time()
                        if new_hover_key != self.current_hover_key or (now - self.last_tooltip_refresh) > self.tooltip_refresh_interval:
                            QToolTip.showText(QCursor.pos(), tooltip_text, self)
                            self.last_tooltip_refresh = now
                            self.current_hover_key = new_hover_key
                    return
        
        # no point nearby
        if self.current_hover_key is not None:
            QToolTip.hideText()
            self.current_hover_key = None

    def update_frame(self, frame):
        """Update display with points from all loaded files"""
        
        # clear old scatter items if frame structure changed
        if "points_by_file" not in frame:
            for item in self.scatter_items.values():
                self.plot.removeItem(item)
            self.scatter_items = {}
            self.point_metadata = {}
            self.scatter_points = {}
            # also clear palm items
            for item in self.palm_scatter_items.values():
                self.plot.removeItem(item)
            self.palm_scatter_items = {}
            self.palm_metadata = {}
            self.palm_points = {}
            return
        
        points_by_file = frame.get("points_by_file", {})
        
        # remove scatter items for files/clusters no longer in frame
        keys_to_remove = [k for k in self.scatter_items if k[0] not in points_by_file]
        for k in keys_to_remove:
            self.plot.removeItem(self.scatter_items[k])
            del self.scatter_items[k]
            if k in self.scatter_points:
                del self.scatter_points[k]
            # also remove metadata for removed items
            meta_keys_to_remove = [mk for mk in self.point_metadata if mk[0] == k[0] and mk[1] == k[1]]
            for mk in meta_keys_to_remove:
                del self.point_metadata[mk]
        
        # update or create scatter items for each file and cluster
        for filename, data in points_by_file.items():
            points_by_cluster = data["points_by_cluster"]
            metadata_by_cluster = data.get("metadata_by_cluster", {})
            color = data["color"]
            
            for cluster_name, points in points_by_cluster.items():
                key = (filename, cluster_name)
                
                # check if this cluster should be displayed
                if not self.cluster_filter.get(cluster_name, True):
                    if key in self.scatter_items:
                        self.plot.removeItem(self.scatter_items[key])
                        del self.scatter_items[key]
                    if key in self.scatter_points:
                        del self.scatter_points[key]
                    continue
                
                # create scatter item if needed
                if key not in self.scatter_items:
                    scatter = pg.ScatterPlotItem(size=12, brush=color)
                    self.plot.addItem(scatter)
                    self.scatter_items[key] = scatter
                else:
                    self.scatter_items[key].setBrush(color)
                
                # update data
                if points.size:
                    self.scatter_items[key].setData(
                        points[:, 0],
                        points[:, 1]
                    )
                    
                    # store points array for hover detection
                    self.scatter_points[key] = points
                    
                    # store metadata for each point
                    metadata_list = metadata_by_cluster.get(cluster_name, [])
                    for idx, meta in enumerate(metadata_list):
                        meta_key = (filename, cluster_name, idx)
                        self.point_metadata[meta_key] = meta
                else:
                    self.scatter_items[key].setData([], [])
        
        # handle palm rendering
        palms_by_file = frame.get("palms_by_file", {})
        
        # remove palm scatter items for files no longer in frame
        palm_keys_to_remove = [k for k in self.palm_scatter_items if k[0] not in palms_by_file]
        for k in palm_keys_to_remove:
            self.plot.removeItem(self.palm_scatter_items[k])
            del self.palm_scatter_items[k]
            if k in self.palm_points:
                del self.palm_points[k]
            # also remove metadata for removed items
            meta_keys_to_remove = [mk for mk in self.palm_metadata if mk[0] == k[0] and mk[1] == k[1]]
            for mk in meta_keys_to_remove:
                del self.palm_metadata[mk]
        
        # update or create palm scatter items
        if self.palm_mode in ['real', 'kalman', 'cubic_spline', 'pchip']:
            for filename, data in palms_by_file.items():
                color = data.get("color", 'gray')
                
                # determine which palm data to use based on mode
                palm_data_mapping = {
                    'real': ('real_palms', 'palm_metadata', 'o'),
                    'kalman': ('kalman_palms', 'kalman_metadata', 's'),
                    'cubic_spline': ('cubic_spline_palms', 'cubic_spline_metadata', '^'),
                    'pchip': ('pchip_spline_palms', 'pchip_spline_metadata', 'd')
                }
                
                if self.palm_mode not in palm_data_mapping:
                    continue
                
                palms_key, meta_key, marker_symbol = palm_data_mapping[self.palm_mode]
                
                if palms_key not in data:
                    continue
                
                palm_points = data[palms_key]
                key = (filename, self.palm_mode)
                
                # create scatter item if needed
                if key not in self.palm_scatter_items:
                    scatter = pg.ScatterPlotItem(size=15, symbol=marker_symbol, pen=color, brush=None)
                    self.plot.addItem(scatter)
                    self.palm_scatter_items[key] = scatter
                else:
                    self.palm_scatter_items[key].setPen(color)
                
                # update data
                if palm_points.size:
                    self.palm_scatter_items[key].setData(
                        palm_points[:, 0],
                        palm_points[:, 1]
                    )
                    
                    # store points array for hover detection
                    self.palm_points[key] = palm_points
                    
                    # store metadata for each palm
                    if self.palm_mode == "real":
                        metadata_list = data.get("palm_metadata", [])
                    else:
                        metadata_list = data.get("estimated_palm_metadata", [])

                    for idx, meta in enumerate(metadata_list):
                        meta_key = (filename, self.palm_mode, idx)
                        self.palm_metadata[meta_key] = meta
                else:
                    self.palm_scatter_items[key].setData([], [])
        else:
            # hide all palm items if mode is None
            for item in self.palm_scatter_items.values():
                self.plot.removeItem(item)
            self.palm_scatter_items = {}
            self.palm_metadata = {}
            self.palm_points = {}

import argparse
import os
import sys

import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import map_coordinates
from PyQt6.QtCore import QLocale, QPointF, QRectF, Qt
from PyQt6.QtGui import (QBrush, QColor, QImage, QPainter, QPainterPath, QPen,
                         QTransform)
from PyQt6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox,
                             QFileDialog,
                             QGraphicsEllipseItem, QGraphicsItem,
                             QGraphicsPathItem, QGraphicsScene,
                             QGraphicsSimpleTextItem, QGraphicsView,
                             QGridLayout, QGroupBox, QHBoxLayout, QLabel,
                             QListWidget, QMainWindow, QMessageBox,
                             QPushButton, QSlider, QSpinBox, QVBoxLayout, QWidget)

import pyLensLib.lenstool as lst
from pyLensLib.compositeModel import compositeModel
from pyLensLib.genlen import genlen
from pyLensLib.piemd import piemd
from pyLensLib.pointsrc import pointsrc
from pyLensLib.sie import sie


CLUSTER_PATHS = {
    'S1063': '/Users/maxmen3/stiva/pietro_models/',
    'M0416_B22': '/Users/maxmen3/stiva/pietro_models/',
    'M0416_canucs': '/Users/maxmen3/stiva/canucs_models/',
    'M1206pl': '/Users/maxmen3/stiva/pietro_models/',
    'PSZ1G311_200': '/Users/maxmen3/stiva/clusters/PSZ1G311/delens/',
    'A370': '/Users/maxmen3/Library/CloudStorage/GoogleDrive-massimo.meneghetti@inaf.it/My Drive/Old_Projects/FF/lens_models/from_params/',
    'A2744': '/Users/maxmen3/stiva/pietro_models/',
    'M0717': '/Users/maxmen3/Library/CloudStorage/GoogleDrive-massimo.meneghetti@inaf.it/My Drive/Old_Projects/FF/lens_models/from_params/',
    'M1149': '/Users/maxmen3/Library/CloudStorage/GoogleDrive-massimo.meneghetti@inaf.it/My Drive/Old_Projects/FF/lens_models/from_params/',
    'M0329': '/Users/maxmen3/stiva/gabriel_models/best_fit/',
    'M1931': '/Users/maxmen3/stiva/gabriel_models/best_fit/',
    'M2129': '/Users/maxmen3/stiva/gabriel_models/best_fit/',
    'R2129': '/Users/maxmen3/stiva/gabriel_models/best_fit/',
    'PLCK-G287': '/Users/maxmen3/stiva/daddona_models/PLCK-G287/',
    'elgordo': '/Users/maxmen3/stiva/gabriel_models/J_A+A_678_A3/files/',
    'A2390': '/Users/maxmen3/stiva/abriola_models/Deflection_Maps_A2390_Gravity/',
}


def normalize_rgb(rgb):
    rgb = np.asarray(rgb, dtype=float)
    finite = np.isfinite(rgb)
    if not np.any(finite):
        return np.zeros((*rgb.shape[:2], 3), dtype=np.uint8)
    rgb = np.nan_to_num(rgb, nan=0.0, posinf=0.0, neginf=0.0)
    vmax = np.nanpercentile(rgb, 99.8)
    if vmax <= 0:
        vmax = np.nanmax(rgb)
    if vmax <= 0:
        vmax = 1.0
    rgb = np.clip(rgb / vmax, 0.0, 1.0)
    return np.ascontiguousarray((255.0 * rgb).astype(np.uint8))


def qimage_from_rgb(rgb_uint8):
    rgb_uint8 = np.ascontiguousarray(rgb_uint8)
    height, width, channels = rgb_uint8.shape
    if channels != 3:
        raise ValueError('RGB image must have shape (ny, nx, 3).')
    image = QImage(
        rgb_uint8.data,
        width,
        height,
        3 * width,
        QImage.Format.Format_RGB888,
    )
    return image.copy()


def load_rgb_fits(path):
    with fits.open(path) as hdul:
        data = hdul[0].data
        header = hdul[0].header.copy()
    if data.ndim == 3 and data.shape[0] == 3:
        data = np.transpose(data, (1, 2, 0))
    if data.ndim != 3 or data.shape[2] != 3:
        raise ValueError(f'{path} does not contain a 3-channel RGB FITS image.')
    return normalize_rgb(data), header


def sigma0_from_theta_e(theta_e, q, co, zl, zs):
    q = max(float(q), 1.0e-4)
    ds = co.angular_diameter_distance(zs).value
    dls = co.angular_diameter_distance_z1z2(zl, zs).value
    if dls <= 0:
        raise ValueError('Source redshift must be larger than lens redshift.')
    c_kms = 299792.458
    arcsec_per_rad = 180.0 / np.pi * 3600.0
    denom = arcsec_per_rad * 4.0 * np.pi * dls / ds
    return c_kms * np.sqrt(float(theta_e) * np.sqrt(q) / denom)


class LenstoolGridModel(genlen):
    def __init__(self, deflector_model):
        super().__init__()
        self.co = deflector_model.co
        self.zl = float(deflector_model.zl)
        self.zs = float(deflector_model.zs)
        self.ds = deflector_model.ds
        self.dls = deflector_model.dls
        self.dl = deflector_model.dl
        self.thetax = np.asarray(deflector_model.thetax, dtype=float)
        self.thetay = np.asarray(deflector_model.thetay, dtype=float)
        self.theta1 = np.asarray(deflector_model.theta1, dtype=float)
        self.theta2 = np.asarray(deflector_model.theta2, dtype=float)
        self.pixel_scale = float(deflector_model.pixel_scale)
        self.nray1 = int(deflector_model.nray1)
        self.nray2 = int(deflector_model.nray2)
        self.a1 = np.asarray(deflector_model.a1, dtype=float).copy()
        self.a2 = np.asarray(deflector_model.a2, dtype=float).copy()
        self.ka = np.asarray(deflector_model.ka, dtype=float).copy()
        self.g1 = np.asarray(deflector_model.g1, dtype=float).copy()
        self.g2 = np.asarray(deflector_model.g2, dtype=float).copy()
        self.computed_potential = False

    def angle(self, theta1, theta2):
        return self.sample(self.a1, theta1, theta2), self.sample(self.a2, theta1, theta2)

    def kappa(self, theta1, theta2):
        return self.sample(self.ka, theta1, theta2)

    def gamma(self, theta1, theta2):
        return self.sample(self.g1, theta1, theta2), self.sample(self.g2, theta1, theta2)

    def sample(self, values, theta1, theta2):
        theta1_arr = np.asarray(theta1, dtype=float)
        theta2_arr = np.asarray(theta2, dtype=float)
        if theta1_arr.shape == values.shape and theta2_arr.shape == values.shape:
            return values.copy()
        xpix = (theta1_arr - self.thetax[0]) / self.pixel_scale
        ypix = (theta2_arr - self.thetay[0]) / self.pixel_scale
        sampled = map_coordinates(
            values,
            [np.ravel(ypix), np.ravel(xpix)],
            order=1,
            mode='nearest',
        ).reshape(theta1_arr.shape)
        if sampled.shape == ():
            return float(sampled)
        return sampled


class PrecomputedGridModel(genlen):
    def __init__(self, co, zl, zs, thetax, thetay, a1, a2, ka, g1, g2):
        super().__init__()
        self.co = co
        self.zl = float(zl)
        self.zs = float(zs)
        self.ds = co.angular_diameter_distance(zs)
        self.dls = co.angular_diameter_distance_z1z2(zl, zs)
        self.dl = co.angular_diameter_distance(zl)
        self.thetax = np.asarray(thetax, dtype=float)
        self.thetay = np.asarray(thetay, dtype=float)
        self.theta1, self.theta2 = np.meshgrid(self.thetax, self.thetay)
        self.pixel_scale = float(self.thetax[1] - self.thetax[0])
        self.nray1 = int(len(self.thetax))
        self.nray2 = int(len(self.thetay))
        self.a1 = np.asarray(a1, dtype=float)
        self.a2 = np.asarray(a2, dtype=float)
        self.ka = np.asarray(ka, dtype=float)
        self.g1 = np.asarray(g1, dtype=float)
        self.g2 = np.asarray(g2, dtype=float)
        self.computed_potential = False

    def angle(self, theta1, theta2):
        return self.sample(self.a1, theta1, theta2), self.sample(self.a2, theta1, theta2)

    def kappa(self, theta1, theta2):
        return self.sample(self.ka, theta1, theta2)

    def gamma(self, theta1, theta2):
        return self.sample(self.g1, theta1, theta2), self.sample(self.g2, theta1, theta2)

    def sample(self, values, theta1, theta2):
        theta1_arr = np.asarray(theta1, dtype=float)
        theta2_arr = np.asarray(theta2, dtype=float)
        if theta1_arr.shape == values.shape and theta2_arr.shape == values.shape:
            return values.copy()
        xpix = (theta1_arr - self.thetax[0]) / self.pixel_scale
        ypix = (theta2_arr - self.thetay[0]) / self.pixel_scale
        sampled = map_coordinates(
            values,
            [np.ravel(ypix), np.ravel(xpix)],
            order=1,
            mode='nearest',
        ).reshape(theta1_arr.shape)
        if sampled.shape == ():
            return float(sampled)
        return sampled


class ArcsecImageItem(QGraphicsItem):
    def __init__(self, qimage, extent, parent=None):
        super().__init__(parent)
        self.qimage = qimage
        self.rect = QRectF(extent[0], extent[2], extent[1] - extent[0], extent[3] - extent[2])
        self.setZValue(-100)

    def boundingRect(self):
        return self.rect

    def paint(self, painter, option, widget=None):
        painter.drawImage(self.rect, self.qimage)


class PlaneView(QGraphicsView):
    def __init__(self, controller, plane_name):
        super().__init__()
        self.controller = controller
        self.plane_name = plane_name
        self._drawing = False
        self._ctrl_panning = False
        self._pan_last_pos = None
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing
            | QPainter.RenderHint.SmoothPixmapTransform
        )
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def reset_view(self):
        if self.scene() is None:
            return
        rect = self.scene().sceneRect()
        if rect.isNull():
            return
        self.setTransform(QTransform())
        self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)
        self.scale(1.0, -1.0)

    def update_drag_mode(self):
        if self.controller.mode() == 'Pan/zoom':
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        else:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)

    def wheelEvent(self, event):
        zoom_in = event.angleDelta().y() > 0
        factor = 1.20 if zoom_in else 1.0 / 1.20
        self.scale(factor, factor)
        event.accept()

    def has_navigation_modifier(self, event):
        modifiers = event.modifiers()
        return bool(
            modifiers & Qt.KeyboardModifier.ControlModifier
            or modifiers & Qt.KeyboardModifier.MetaModifier
        )

    def start_ctrl_pan(self, event):
        self._ctrl_panning = True
        self._pan_last_pos = event.position().toPoint()
        self.setCursor(Qt.CursorShape.ClosedHandCursor)
        event.accept()

    def mousePressEvent(self, event):
        self.update_drag_mode()
        if self.has_navigation_modifier(event) and event.button() != Qt.MouseButton.NoButton:
            self.start_ctrl_pan(event)
            return
        if event.button() == Qt.MouseButton.LeftButton and self.controller.mode() != 'Pan/zoom':
            point = self.mapToScene(event.position().toPoint())
            if self.plane_name == 'image' and self.controller.mode() == 'Draw lens':
                self._drawing = True
                self.controller.begin_sie_interaction(point)
                event.accept()
                return
            if self.controller.mode() == 'Click image' and self.plane_name == 'image':
                self.controller.predict_from_image(point)
                event.accept()
                return
            if self.controller.mode() == 'Mark image family' and self.plane_name == 'image':
                if self.controller.select_family_marker_at(point):
                    event.accept()
                    return
                self.controller.mark_family_point(point)
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        point = self.mapToScene(event.position().toPoint())
        self.controller.update_status_position(self.plane_name, point)
        if self._ctrl_panning and self._pan_last_pos is not None:
            pos = event.position().toPoint()
            delta = pos - self._pan_last_pos
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            self._pan_last_pos = pos
            event.accept()
            return
        if self.has_navigation_modifier(event) and event.buttons():
            self.start_ctrl_pan(event)
            return
        if self._drawing:
            self.controller.update_sie_interaction(point)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._ctrl_panning:
            self._ctrl_panning = False
            self._pan_last_pos = None
            self.unsetCursor()
            event.accept()
            return
        if self._drawing and event.button() == Qt.MouseButton.LeftButton:
            point = self.mapToScene(event.position().toPoint())
            self._drawing = False
            self.controller.finish_sie_interaction(point)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self.controller.delete_selected_object()
            event.accept()
            return
        super().keyPressEvent(event)


class ViewerWindow(QMainWindow):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self._closing_from_controller = False
        self.setWindowTitle('CL Image Finder - Image viewer')
        self.resize(1200, 900)

        central = QWidget(self)
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.addWidget(controller.image_wrapper, stretch=1)
        layout.addWidget(controller.source_wrapper, stretch=0)

    def close_from_controller(self):
        self._closing_from_controller = True
        self.close()

    def closeEvent(self, event):
        if self._closing_from_controller:
            super().closeEvent(event)
            return
        event.ignore()
        self.hide()
        self.controller.status_label.setText('Viewer window hidden. Press Show viewer to reopen it.')


class CLImageFinder(QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.co = FlatLambdaCDM(H0=70.0, Om0=0.3)
        self.lens = None
        self.sie_components = []
        self.selected_sie_index = None
        self.pending_new_sie = False
        self.loading_component_controls = False
        self.sie_center = QPointF(0.0, 0.0)
        self.sie_theta_e = 5.0
        self.source_marker_items = []
        self.image_marker_items = []
        self.last_prediction_anchor = None
        self.current_family_points = []
        self.current_family_items = []
        self.image_families = []
        self.family_items = []
        self.observed_family_catalog_path = None
        self.observed_families = []
        self.observed_family_items = []
        self.selected_family_marker = None
        self.critical_items = []
        self.caustic_items = []
        self.sie_item = None
        self.sie_items = []
        self.sie_preview_item = None
        self.sie_handle_items = []
        self.lenstool_model = None
        self.lenstool_model_prefix = None
        self.lenstool_model_zl = None
        self.lenstool_model_zs0 = None
        self.member_catalog_path = None
        self.member_catalog_ref = None
        self.member_catalog = []
        self.member_items = []
        self.member_catalog_revision = 0
        self.member_layer_cache_key = None
        self.member_layer_cache = None
        self.member_layer_cache_count = 0
        self.member_physical_cache_key = None
        self.member_physical_cache_zs = None
        self.member_physical_cache_maps = None
        self.model_update_counter = 0
        self.active_sie_drag = None
        self.drag_start_point = None
        self.drag_start_center = None
        self.rgb_item = None
        self.rgb_qimage = None
        self.rgb_wcs = None
        self.rgb_shape = None
        self.reference_radec = None
        self.current_prefix = None
        self.current_parfile = None
        self.extent = (-30.0, 30.0, -30.0, 30.0)

        self.setWindowTitle('CL Image Finder - Controls')
        self.resize(1050, 460)
        self.build_ui()
        self.load_cluster(args.cluster, args.prefix, args.rgb)
        self.reset_views()

    def build_ui(self):
        central = QWidget(self)
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        panel_toggle_layout = QHBoxLayout()
        main_layout.addLayout(panel_toggle_layout)
        panel_toggle_layout.addWidget(QLabel('UI panels'))
        self.panel_toggle_layout = panel_toggle_layout
        self.panel_visibility = {}
        self.panel_widgets = {}

        controls_layout = QHBoxLayout()
        main_layout.addLayout(controls_layout)

        main_controls_box = QGroupBox('Main controls')
        main_controls_layout = QGridLayout(main_controls_box)
        controls_layout.addWidget(main_controls_box, stretch=2)

        lens_components_box = QGroupBox('Lens components')
        lens_components_layout = QGridLayout(lens_components_box)
        controls_layout.addWidget(lens_components_box, stretch=3)

        lenstool_box = QGroupBox('Lenstool')
        lenstool_layout = QGridLayout(lenstool_box)
        controls_layout.addWidget(lenstool_box, stretch=1)

        families_box = QGroupBox('Families')
        families_layout = QGridLayout(families_box)
        controls_layout.addWidget(families_box, stretch=1)

        self.cluster_combo = QComboBox()
        self.cluster_combo.addItems(list(CLUSTER_PATHS.keys()))
        if self.args.cluster in CLUSTER_PATHS:
            self.cluster_combo.setCurrentText(self.args.cluster)
        self.cluster_combo.currentTextChanged.connect(self.change_cluster)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(['Draw lens', 'Click image', 'Mark image family', 'Pan/zoom'])
        self.mode_combo.currentTextChanged.connect(self.update_view_modes)
        self.profile_combo = QComboBox()
        self.profile_combo.addItems(['SIE', 'PIEMD'])
        self.profile_combo.currentTextChanged.connect(self.on_profile_changed)

        self.theta_e_spin = self.double_spin(1.0e-4, 300.0, 5.0, 4, 0.1)
        self.q_spin = self.double_spin(0.01, 1.0, 0.75, 4, 0.01)
        self.pa_spin = self.double_spin(-180.0, 180.0, 0.0, 2, 0.1)
        self.core_spin = self.double_spin(0.0, 50.0, 0.0, 4, 0.1)
        self.theta_t_spin = self.double_spin(1.0e-4, 1000.0, 100.0, 4, 1.0)
        self.zl_spin = self.double_spin(0.0, 10.0, 0.4, 4, 0.1)
        self.zs_spin = self.double_spin(0.0, 20.0, 2.0, 4, 0.1)
        self.nray_spin = QSpinBox()
        self.nray_spin.setRange(64, 4096)
        self.nray_spin.setValue(self.args.nray)
        self.nray_spin.setSingleStep(64)
        for spin in [
            self.theta_e_spin,
            self.q_spin,
            self.pa_spin,
            self.core_spin,
            self.theta_t_spin,
            self.zl_spin,
            self.zs_spin,
        ]:
            spin.editingFinished.connect(self.update_sie_from_controls)
        self.nray_spin.editingFinished.connect(self.update_sie_from_controls)
        for spin in [self.zl_spin, self.zs_spin, self.nray_spin]:
            spin.valueChanged.connect(lambda _value: self.update_sie_from_controls())
        self.show_radial_check = QCheckBox('Radial critical lines')
        self.show_radial_check.stateChanged.connect(lambda _state: self.draw_critical_curves())
        self.show_source_inset_check = QCheckBox('Show source inset')
        self.show_source_inset_check.stateChanged.connect(self.update_source_inset_visibility)
        self.use_lenstool_check = QCheckBox('Use Lenstool model')
        self.use_lenstool_check.stateChanged.connect(self.toggle_lenstool_model)

        self.update_sie_button = QPushButton('Update model')
        self.update_sie_button.setToolTip('Rebuild the full model using the current component, redshift, and grid settings.')
        self.update_sie_button.clicked.connect(self.force_update_model)
        self.add_sie_button = QPushButton('Add component')
        self.add_sie_button.clicked.connect(self.start_add_sie)
        self.load_lenstool_button = QPushButton('Load Lenstool...')
        self.load_lenstool_button.clicked.connect(self.choose_lenstool_model)
        self.load_members_button = QPushButton('Load members...')
        self.load_members_button.clicked.connect(self.choose_member_catalog)
        self.use_members_check = QCheckBox('Use members')
        self.use_members_check.setEnabled(False)
        self.use_members_check.stateChanged.connect(self.toggle_member_catalog)
        self.show_members_check = QCheckBox('Show members')
        self.show_members_check.setChecked(True)
        self.show_members_check.setEnabled(False)
        self.show_members_check.stateChanged.connect(lambda _state: self.draw_member_markers())
        self.update_members_button = QPushButton('Update members')
        self.update_members_button.setEnabled(False)
        self.update_members_button.clicked.connect(self.update_sie_from_controls)

        self.member_mag0_slider, self.member_mag0_spin = self.slider_spin_pair(
            10.0, 30.0, 17.0, 3, 0.1
        )
        self.member_sigma0_slider, self.member_sigma0_spin = self.slider_spin_pair(
            10.0, 800.0, 300.0, 3, 5.0
        )
        self.member_alpha_slider, self.member_alpha_spin = self.slider_spin_pair(
            0.0, 1.5, 0.28, 4, 0.01
        )
        self.member_cut0_slider, self.member_cut0_spin = self.slider_spin_pair(
            0.1, 1000.0, 10.0, 4, 0.1, log_scale=True
        )
        self.member_beta_slider, self.member_beta_spin = self.slider_spin_pair(
            0.0, 2.0, 0.64, 4, 0.01
        )
        self.member_core0_slider, self.member_core0_spin = self.slider_spin_pair(
            1.0e-5, 10.0, 1.0e-4, 6, 1.0e-4, log_scale=True
        )
        self.member_magmax_slider, self.member_magmax_spin = self.slider_spin_pair(
            10.0, 35.0, 25.0, 3, 0.1
        )
        self.member_widgets = [
            self.member_mag0_slider,
            self.member_mag0_spin,
            self.member_sigma0_slider,
            self.member_sigma0_spin,
            self.member_alpha_slider,
            self.member_alpha_spin,
            self.member_cut0_slider,
            self.member_cut0_spin,
            self.member_beta_slider,
            self.member_beta_spin,
            self.member_core0_slider,
            self.member_core0_spin,
            self.member_magmax_slider,
            self.member_magmax_spin,
        ]
        self.set_member_controls_enabled(False)

        self.add_family_button = QPushButton('Add family')
        self.add_family_button.clicked.connect(self.add_current_family)
        self.export_families_button = QPushButton('Export families')
        self.export_families_button.clicked.connect(self.export_families)
        self.load_observed_families_button = QPushButton('Load observed...')
        self.load_observed_families_button.clicked.connect(self.choose_observed_family_catalog)
        self.show_observed_families_check = QCheckBox('Show observed')
        self.show_observed_families_check.setChecked(True)
        self.show_observed_families_check.setEnabled(False)
        self.show_observed_families_check.stateChanged.connect(lambda _state: self.redraw_observed_family_markers())
        self.clear_button = QPushButton('Clear markers')
        self.clear_button.clicked.connect(
            lambda _checked=False: self.clear_prediction_markers(clear_table=True, clear_anchor=True)
        )
        self.reset_button = QPushButton('Reset view')
        self.reset_button.clicked.connect(self.reset_views)
        self.show_viewer_button = QPushButton('Show viewer')
        self.show_viewer_button.clicked.connect(self.show_viewer_window)

        cluster_label = QLabel('Cluster')
        mode_label = QLabel('Mode')
        zl_label = QLabel('zl')
        zs_label = QLabel('zs')
        nray_label = QLabel('Nray')
        main_controls_layout.addWidget(cluster_label, 0, 0)
        main_controls_layout.addWidget(self.cluster_combo, 0, 1)
        main_controls_layout.addWidget(mode_label, 1, 0)
        main_controls_layout.addWidget(self.mode_combo, 1, 1)
        main_controls_layout.addWidget(zl_label, 2, 0)
        main_controls_layout.addWidget(self.zl_spin, 2, 1)
        main_controls_layout.addWidget(zs_label, 3, 0)
        main_controls_layout.addWidget(self.zs_spin, 3, 1)
        main_controls_layout.addWidget(nray_label, 4, 0)
        main_controls_layout.addWidget(self.nray_spin, 4, 1)
        main_controls_layout.addWidget(self.show_radial_check, 5, 0, 1, 2)
        main_controls_layout.addWidget(self.show_source_inset_check, 6, 0, 1, 2)
        main_controls_layout.addWidget(self.update_sie_button, 7, 0, 1, 2)
        main_controls_layout.addWidget(self.clear_button, 8, 0)
        main_controls_layout.addWidget(self.reset_button, 8, 1)
        main_controls_layout.addWidget(self.show_viewer_button, 9, 0, 1, 2)

        profile_label = QLabel('Profile')
        theta_e_label = QLabel('theta_E')
        q_label = QLabel('q')
        pa_label = QLabel('PA [deg]')
        core_label = QLabel('core radius')
        self.theta_t_label = QLabel('theta_t')
        lens_components_layout.addWidget(profile_label, 0, 0)
        lens_components_layout.addWidget(self.profile_combo, 0, 1)
        lens_components_layout.addWidget(self.add_sie_button, 0, 2, 1, 2)
        lens_components_layout.addWidget(theta_e_label, 1, 0)
        lens_components_layout.addWidget(self.theta_e_spin, 1, 1)
        lens_components_layout.addWidget(q_label, 1, 2)
        lens_components_layout.addWidget(self.q_spin, 1, 3)
        lens_components_layout.addWidget(pa_label, 2, 0)
        lens_components_layout.addWidget(self.pa_spin, 2, 1)
        lens_components_layout.addWidget(core_label, 2, 2)
        lens_components_layout.addWidget(self.core_spin, 2, 3)
        lens_components_layout.addWidget(self.theta_t_label, 3, 0)
        lens_components_layout.addWidget(self.theta_t_spin, 3, 1)

        lenstool_layout.addWidget(self.use_lenstool_check, 0, 0, 1, 2)
        lenstool_layout.addWidget(self.load_lenstool_button, 1, 0, 1, 2)

        families_layout.addWidget(self.add_family_button, 0, 0)
        families_layout.addWidget(self.export_families_button, 1, 0)
        families_layout.addWidget(self.load_observed_families_button, 2, 0)
        families_layout.addWidget(self.show_observed_families_check, 3, 0)

        member_box = QGroupBox('Cluster member galaxies')
        member_layout = QGridLayout(member_box)
        main_layout.addWidget(member_box)
        member_layout.addWidget(self.load_members_button, 0, 0, 1, 2)
        member_layout.addWidget(self.use_members_check, 0, 2, 1, 2)
        member_layout.addWidget(self.show_members_check, 0, 4, 1, 2)
        member_layout.addWidget(self.update_members_button, 0, 6, 1, 2)
        self.add_member_control(member_layout, 1, 0, 'mag_0', self.member_mag0_slider, self.member_mag0_spin)
        self.add_member_control(member_layout, 1, 3, 'sigma0_0', self.member_sigma0_slider, self.member_sigma0_spin)
        self.add_member_control(member_layout, 1, 6, 'alpha', self.member_alpha_slider, self.member_alpha_spin)
        self.add_member_control(member_layout, 2, 0, 'cut_0', self.member_cut0_slider, self.member_cut0_spin)
        self.add_member_control(member_layout, 2, 3, 'beta', self.member_beta_slider, self.member_beta_spin)
        self.add_member_control(member_layout, 2, 6, 'core_0', self.member_core0_slider, self.member_core0_spin)
        self.add_member_control(member_layout, 3, 0, 'mag max', self.member_magmax_slider, self.member_magmax_spin)

        self.source_scene = QGraphicsScene(self)
        self.image_scene = QGraphicsScene(self)
        self.source_view = PlaneView(self, 'source')
        self.image_view = PlaneView(self, 'image')
        self.source_view.setScene(self.source_scene)
        self.image_view.setScene(self.image_scene)
        self.image_wrapper = self.with_title('Image plane', self.image_view)
        self.source_wrapper = self.with_title('Source plane inset', self.source_view)
        self.source_wrapper.setMaximumWidth(420)
        self.source_wrapper.hide()
        self.viewer_window = ViewerWindow(self)

        bottom_layout = QHBoxLayout()
        main_layout.addLayout(bottom_layout)
        self.status_label = QLabel('Ready')
        bottom_layout.addWidget(self.status_label, stretch=1)
        self.family_list_widget = QListWidget()
        self.family_list_widget.hide()
        bottom_layout.addWidget(self.family_list_widget, stretch=1)

        self.create_panel_toggle('Main controls', [main_controls_box])
        self.create_panel_toggle('Lens components', [lens_components_box])
        self.create_panel_toggle('Lenstool', [lenstool_box])
        self.create_panel_toggle('Families', [families_box, self.family_list_widget])
        self.create_panel_toggle('Cluster member galaxies', [member_box])
        self.panel_toggle_layout.addStretch(1)
        self.update_profile_specific_controls()

    def with_title(self, title, widget):
        wrapper = QWidget()
        layout = QVBoxLayout(wrapper)
        label = QLabel(title)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)
        layout.addWidget(widget)
        return wrapper

    def create_panel_toggle(self, label, widgets):
        button = QPushButton(label)
        button.setCheckable(True)
        button.setChecked(True)
        button.setToolTip(f'Show or hide the {label.lower()} UI panel')
        self.panel_widgets[label] = list(widgets)
        self.panel_visibility[label] = True
        button.toggled.connect(lambda checked, panel=label: self.set_panel_visible(panel, checked))
        self.panel_toggle_layout.addWidget(button)
        return button

    def set_panel_visible(self, panel, visible):
        self.panel_visibility[panel] = bool(visible)
        for widget in self.panel_widgets.get(panel, []):
            if widget is self.family_list_widget and visible:
                continue
            widget.setVisible(bool(visible))
        if panel == 'Families':
            if visible:
                self.update_family_list()
            else:
                self.family_list_widget.hide()
        if panel == 'Lens components':
            self.update_profile_specific_controls()

    def panel_is_visible(self, panel):
        return self.panel_visibility.get(panel, True)

    def on_profile_changed(self, _text):
        self.update_profile_specific_controls()
        self.update_sie_from_controls()

    def update_profile_specific_controls(self):
        if not hasattr(self, 'theta_t_label'):
            return
        show_theta_t = (
            self.profile_combo.currentText() == 'PIEMD'
            and self.panel_is_visible('Lens components')
        )
        self.theta_t_label.setVisible(show_theta_t)
        self.theta_t_spin.setVisible(show_theta_t)

    def double_spin(self, bottom, top, value, decimals, step):
        spin = QDoubleSpinBox()
        spin.setRange(bottom, top)
        spin.setDecimals(decimals)
        spin.setSingleStep(step)
        spin.setValue(value)
        spin.setLocale(QLocale.c())
        spin.setKeyboardTracking(False)
        return spin

    def slider_spin_pair(self, bottom, top, value, decimals, step, log_scale=False):
        spin = self.double_spin(bottom, top, value, decimals, step)
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, 1000)
        if log_scale:
            log_bottom = np.log10(bottom)
            log_top = np.log10(top)

            def value_from_slider(slider_value):
                frac = slider_value / 1000.0
                return 10.0 ** (log_bottom + frac * (log_top - log_bottom))

            def slider_from_value(spin_value):
                spin_value = np.clip(float(spin_value), bottom, top)
                frac = (np.log10(spin_value) - log_bottom) / (log_top - log_bottom)
                return int(np.clip(round(frac * 1000.0), 0, 1000))
        else:
            def value_from_slider(slider_value):
                frac = slider_value / 1000.0
                return bottom + frac * (top - bottom)

            def slider_from_value(spin_value):
                frac = (float(spin_value) - bottom) / (top - bottom)
                return int(np.clip(round(frac * 1000.0), 0, 1000))

        def update_spin(slider_value):
            spin.blockSignals(True)
            spin.setValue(value_from_slider(slider_value))
            spin.blockSignals(False)

        def update_slider():
            slider.blockSignals(True)
            slider.setValue(slider_from_value(spin.value()))
            slider.blockSignals(False)

        slider.valueChanged.connect(update_spin)
        spin.editingFinished.connect(update_slider)
        update_slider()
        return slider, spin

    def add_member_control(self, layout, row, col, label, slider, spin):
        layout.addWidget(QLabel(label), row, col)
        layout.addWidget(slider, row, col + 1)
        layout.addWidget(spin, row, col + 2)

    def set_member_controls_enabled(self, enabled):
        for widget in getattr(self, 'member_widgets', []):
            widget.setEnabled(enabled)
        if hasattr(self, 'update_members_button'):
            self.update_members_button.setEnabled(enabled)

    def mode(self):
        return self.mode_combo.currentText()

    def update_view_modes(self):
        self.source_view.update_drag_mode()
        self.image_view.update_drag_mode()
        self.ensure_sie_handles_visible()
        if self.mode() == 'Mark image family':
            self.status_label.setText('Click image positions for one family, then press Add family.')

    def update_source_inset_visibility(self):
        show_inset = self.show_source_inset_check.isChecked()
        self.source_wrapper.setVisible(show_inset)
        if show_inset:
            self.show_viewer_window()
            self.source_view.reset_view()

    def show_viewer_window(self, _checked=False):
        if not hasattr(self, 'viewer_window'):
            return
        self.viewer_window.show()
        self.viewer_window.raise_()
        self.viewer_window.activateWindow()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self.delete_selected_object()
            event.accept()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event):
        if hasattr(self, 'viewer_window'):
            self.viewer_window.close_from_controller()
        super().closeEvent(event)

    def cluster_prefix(self, cluster, prefix_override=None):
        if prefix_override:
            return prefix_override
        return os.path.join(CLUSTER_PATHS[cluster], cluster)

    def load_cluster(self, cluster, prefix_override=None, rgb_override=None):
        prefix = self.cluster_prefix(cluster, prefix_override)
        parfile = prefix + '.par'
        rgbfile = rgb_override or prefix + '_rgb.fits'
        self.current_prefix = prefix
        self.current_parfile = parfile
        self.reference_radec = None
        self.lenstool_model = None
        self.lenstool_model_prefix = None
        self.lenstool_model_zl = None
        self.lenstool_model_zs0 = None
        if hasattr(self, 'use_lenstool_check'):
            self.use_lenstool_check.blockSignals(True)
            self.use_lenstool_check.setChecked(False)
            self.use_lenstool_check.blockSignals(False)
        self.reset_member_catalog()
        self.update_member_catalog_availability()

        zl = self.zl_spin.value()
        if os.path.exists(parfile):
            try:
                zl = float(lst.getLensRedshift(parfile))
            except Exception:
                zl = self.zl_spin.value()
            self.reference_radec = self.reference_radec_from_parfile(parfile)
        self.zl_spin.setValue(zl)

        fov = float(self.args.fov)
        if os.path.exists(parfile):
            try:
                lims = lst.getFoV(parfile)
                fov = float(lims[1] - lims[0])
            except Exception:
                fov = float(self.args.fov)
        half = 0.5 * fov
        self.extent = (-half, half, -half, half)

        if os.path.exists(rgbfile):
            try:
                rgb_data, rgb_header = load_rgb_fits(rgbfile)
                self.rgb_qimage = qimage_from_rgb(rgb_data)
                self.rgb_shape = rgb_data.shape[:2]
                self.rgb_wcs = self.wcs_from_header(rgb_header)
                self.status_label.setText(f'Loaded RGB: {rgbfile}')
            except Exception as exc:
                self.rgb_qimage = self.blank_image()
                self.rgb_shape = (self.rgb_qimage.height(), self.rgb_qimage.width())
                self.rgb_wcs = None
                self.status_label.setText(f'Could not load RGB ({exc}); using blank image.')
        else:
            self.rgb_qimage = self.blank_image()
            self.rgb_shape = (self.rgb_qimage.height(), self.rgb_qimage.width())
            self.rgb_wcs = None
            self.status_label.setText(f'RGB not found: {rgbfile}; using blank image.')

        self.rebuild_scenes()
        self.lens = None
        self.status_label.setText('Draw the first component, or press Add component.')

    def blank_image(self):
        data = np.full((512, 512, 3), 18, dtype=np.uint8)
        data[:, :, 1] = 22
        data[:, :, 2] = 28
        return qimage_from_rgb(data)

    def wcs_from_header(self, header):
        try:
            wcs = WCS(header).celestial
        except Exception:
            return None
        if not wcs.has_celestial:
            return None
        return wcs

    def reference_radec_from_parfile(self, parfile):
        try:
            with open(parfile, 'r', encoding='utf-8') as handle:
                has_reference = any(line.lstrip().startswith('reference ') for line in handle)
            if not has_reference:
                return None
            return tuple(float(v) for v in lst.getRef_RA_DEC(parfile))
        except Exception:
            return None

    def change_cluster(self, cluster):
        if getattr(self, 'args', None) is None:
            return
        self.load_cluster(cluster)
        self.reset_views()

    def rebuild_scenes(self):
        self.source_scene.clear()
        self.image_scene.clear()
        rect = QRectF(
            self.extent[0],
            self.extent[2],
            self.extent[1] - self.extent[0],
            self.extent[3] - self.extent[2],
        )
        self.source_scene.setSceneRect(rect)
        self.image_scene.setSceneRect(rect)
        self.image_scene.addItem(ArcsecImageItem(self.rgb_qimage, self.extent))
        self.add_plane_frame(self.source_scene)
        self.add_plane_frame(self.image_scene)
        self.sie_components = []
        self.selected_sie_index = None
        self.pending_new_sie = False
        self.sie_item = None
        self.sie_items = []
        self.sie_preview_item = None
        self.source_marker_items = []
        self.image_marker_items = []
        self.current_family_points = []
        self.current_family_items = []
        self.image_families = []
        self.family_items = []
        self.reset_observed_family_catalog()
        self.selected_family_marker = None
        self.family_list_widget.clear()
        self.family_list_widget.hide()
        self.critical_items = []
        self.caustic_items = []
        self.sie_handle_items = []
        self.active_sie_drag = None
        self.drag_start_point = None
        self.drag_start_center = None
        self.last_prediction_anchor = None
        self.member_items = []
        self.draw_member_markers()

    def add_plane_frame(self, scene):
        xmin, xmax, ymin, ymax = self.extent
        pen = QPen(QColor(180, 180, 180, 110))
        pen.setCosmetic(True)
        scene.addRect(QRectF(xmin, ymin, xmax - xmin, ymax - ymin), pen)
        axis_pen = QPen(QColor(120, 170, 220, 90))
        axis_pen.setCosmetic(True)
        scene.addLine(xmin, 0.0, xmax, 0.0, axis_pen)
        scene.addLine(0.0, ymin, 0.0, ymax, axis_pen)

    def reset_views(self):
        self.source_view.reset_view()
        self.image_view.reset_view()

    def start_add_sie(self):
        self.pending_new_sie = True
        self.selected_sie_index = None
        self.selected_family_marker = None
        self.redraw_current_family_markers()
        self.redraw_family_markers()
        self.remove_sie_handles()
        self.draw_all_sie_ellipses()
        self.mode_combo.setCurrentText('Draw lens')
        self.status_label.setText('Draw the new SIE ellipse on the image plane.')

    def toggle_lenstool_model(self, _state=None):
        if self.use_lenstool_check.isChecked() and self.lenstool_model is None:
            prefix = self.current_prefix
            if not prefix or not self.load_lenstool_model(prefix):
                self.use_lenstool_check.blockSignals(True)
                self.use_lenstool_check.setChecked(False)
                self.use_lenstool_check.blockSignals(False)
                return
        if self.use_lenstool_check.isChecked():
            self.clear_member_catalog_for_lenstool()
        self.update_member_catalog_availability()
        self.update_sie_from_controls()

    def choose_lenstool_model(self, _checked=False):
        start_dir = os.path.dirname(self.current_parfile) if self.current_parfile else os.getcwd()
        parfile, _ = QFileDialog.getOpenFileName(
            self,
            'Load Lenstool parameter file',
            start_dir,
            'Lenstool parameter files (*.par);;All files (*)',
        )
        if not parfile:
            return
        prefix = parfile[:-4] if parfile.endswith('.par') else os.path.splitext(parfile)[0]
        if self.load_lenstool_model(prefix):
            self.use_lenstool_check.blockSignals(True)
            self.use_lenstool_check.setChecked(True)
            self.use_lenstool_check.blockSignals(False)
            self.update_sie_from_controls()

    def reset_member_catalog(self):
        self.member_catalog_path = None
        self.member_catalog_ref = None
        self.member_catalog = []
        self.invalidate_member_layer_cache()
        self.member_catalog_revision += 1
        if hasattr(self, 'member_items'):
            self.remove_items(self.member_items)
            self.member_items = []
        if hasattr(self, 'use_members_check'):
            self.use_members_check.blockSignals(True)
            self.use_members_check.setChecked(False)
            self.use_members_check.setEnabled(False)
            self.use_members_check.blockSignals(False)
        if hasattr(self, 'show_members_check'):
            self.show_members_check.setEnabled(False)
        if hasattr(self, 'update_members_button'):
            self.set_member_controls_enabled(False)
        self.update_member_catalog_availability()

    def clear_member_catalog_for_lenstool(self):
        if not getattr(self, 'member_catalog', None):
            self.update_member_catalog_availability()
            return False
        self.reset_member_catalog()
        self.draw_member_markers()
        return True

    def member_catalog_allowed(self):
        return getattr(self, 'lenstool_model', None) is None

    def update_member_catalog_availability(self):
        if not hasattr(self, 'load_members_button'):
            return
        allowed = self.member_catalog_allowed()
        self.load_members_button.setEnabled(allowed)
        if allowed:
            self.load_members_button.setToolTip('Load a cluster member catalog.')
        else:
            self.load_members_button.setToolTip(
                'Cluster member catalogs are disabled while a Lenstool model is loaded.'
            )

    def choose_member_catalog(self, _checked=False):
        if not self.member_catalog_allowed():
            QMessageBox.information(
                self,
                'Cluster members disabled',
                'Cluster member catalogs are disabled while a Lenstool model is loaded. '
                'You can still add SIE/PIEMD lens components manually.',
            )
            return
        start_dir = os.path.dirname(self.current_parfile) if self.current_parfile else os.getcwd()
        local_catalog = os.path.join(os.path.dirname(__file__), 'members_S1063.cat')
        if os.path.exists(local_catalog):
            start_dir = os.path.dirname(local_catalog)
        path, _ = QFileDialog.getOpenFileName(
            self,
            'Load cluster member catalog',
            start_dir,
            'Lenstool/member catalogs (*.cat *.txt *.dat);;All files (*)',
        )
        if not path:
            return
        try:
            catalog_ref, catalog = self.parse_member_catalog(path)
        except Exception as exc:
            QMessageBox.warning(self, 'Could not load member catalog', str(exc))
            return
        self.member_catalog_path = path
        self.member_catalog_ref = catalog_ref
        self.member_catalog = catalog
        self.invalidate_member_layer_cache()
        self.member_catalog_revision += 1
        if self.reference_radec is None:
            self.reference_radec = catalog_ref
        self.use_members_check.blockSignals(True)
        self.use_members_check.setEnabled(True)
        self.use_members_check.setChecked(True)
        self.use_members_check.blockSignals(False)
        self.show_members_check.setEnabled(True)
        self.set_member_controls_enabled(True)
        self.draw_member_markers()
        self.update_sie_from_controls()
        self.status_label.setText(f'Loaded {len(catalog)} cluster member galaxies from {path}.')

    def toggle_member_catalog(self, _state=None):
        enabled = self.members_enabled()
        self.set_member_controls_enabled(enabled)
        self.draw_member_markers()
        self.update_sie_from_controls()

    def members_enabled(self):
        return (
            hasattr(self, 'use_members_check')
            and self.use_members_check.isChecked()
            and bool(self.member_catalog)
            and self.member_catalog_allowed()
        )

    def invalidate_member_layer_cache(self):
        self.member_layer_cache_key = None
        self.member_layer_cache = None
        self.member_layer_cache_count = 0
        self.member_physical_cache_key = None
        self.member_physical_cache_zs = None
        self.member_physical_cache_maps = None

    def parse_member_catalog(self, path):
        with open(path, 'r', encoding='utf-8') as handle:
            lines = handle.readlines()
        first = next((line.strip() for line in lines if line.strip()), '')
        if not first.startswith('#REFERENCE'):
            raise ValueError('The first non-empty line must be "#REFERENCE type RA DEC".')
        parts = first.split()
        if len(parts) < 4:
            raise ValueError('Invalid #REFERENCE line. Expected "#REFERENCE type RA DEC".')
        catalog_ref = (float(parts[2]), float(parts[3]))
        catalog = []
        for line_number, line in enumerate(lines, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue
            values = stripped.split()
            if len(values) < 7:
                raise ValueError(
                    f'Catalog line {line_number} has {len(values)} columns; expected at least 7.'
                )
            galaxy_id = values[0]
            ra = float(values[1])
            dec = float(values[2])
            mag = float(values[6])
            q = self.member_catalog_q(values)
            pa_deg = float(values[5]) if len(values) > 5 else 0.0
            x, y = self.radec_to_scene(ra, dec, fallback_ref=catalog_ref)
            catalog.append({
                'id': galaxy_id,
                'ra': ra,
                'dec': dec,
                'x': x,
                'y': y,
                'mag': mag,
                'q': q,
                'pa_deg': pa_deg,
            })
        if not catalog:
            raise ValueError('The member catalog does not contain any galaxy rows.')
        return catalog_ref, catalog

    def member_catalog_q(self, values):
        if len(values) <= 4:
            return 1.0
        try:
            major = abs(float(values[3]))
            minor = abs(float(values[4]))
        except ValueError:
            return 1.0
        if major <= 0.0 or minor <= 0.0:
            return 1.0
        return float(np.clip(min(major, minor) / max(major, minor), 0.01, 1.0))

    def radec_to_scene(self, ra, dec, fallback_ref=None):
        reference = self.reference_radec or fallback_ref
        if reference is None:
            if self.rgb_wcs is not None and self.rgb_shape is not None:
                xpix, ypix = self.rgb_wcs.world_to_pixel_values(float(ra), float(dec))
                ny, nx = self.rgb_shape
                xmin, xmax, ymin, ymax = self.extent
                x = xmin + float(xpix) / (nx - 1) * (xmax - xmin)
                y = ymin + float(ypix) / (ny - 1) * (ymax - ymin)
                return float(x), float(y)
            return float(ra), float(dec)
        ra_ref, dec_ref = reference
        x = -(float(ra) - float(ra_ref)) * np.cos(np.radians(float(dec_ref))) * 3600.0
        y = (float(dec) - float(dec_ref)) * 3600.0
        return float(x), float(y)

    def draw_member_markers(self):
        self.remove_items(getattr(self, 'member_items', []))
        self.member_items = []
        if (
            not getattr(self, 'member_catalog', None)
            or not hasattr(self, 'show_members_check')
            or not self.show_members_check.isChecked()
        ):
            return
        color = QColor(255, 245, 120, 190) if self.members_enabled() else QColor(170, 170, 170, 120)
        radius = max(0.08, 0.0025 * max(self.extent[1] - self.extent[0], self.extent[3] - self.extent[2]))
        for member in self.member_catalog:
            self.member_items.append(
                self.add_marker(
                    self.image_scene,
                    member['x'],
                    member['y'],
                    color,
                    radius=radius,
                    zvalue=8,
                )
            )

    def load_lenstool_model(self, prefix):
        parfile = prefix + '.par'
        filex = prefix + '_angx.fits'
        filey = prefix + '_angy.fits'
        filepot = prefix + '_pot.fits'
        missing = [path for path in (parfile, filex, filey) if not os.path.exists(path)]
        if missing:
            QMessageBox.warning(
                self,
                'Lenstool model missing files',
                'Could not load Lenstool model. Missing:\n' + '\n'.join(missing),
            )
            return False
        try:
            zl = float(lst.getLensRedshift(parfile))
            zs0 = 1.0
            self.zs_spin.setValue(zs0)
            model = self.create_lenstool_grid_model(prefix, zl, zs0)
        except Exception as exc:
            QMessageBox.warning(self, 'Could not load Lenstool model', str(exc))
            return False

        self.lenstool_model = model
        self.lenstool_model_prefix = prefix
        self.lenstool_model_zl = zl
        self.lenstool_model_zs0 = zs0
        self.co = getattr(model, 'co', self.co)
        self.zl_spin.setValue(zl)
        removed_members = self.clear_member_catalog_for_lenstool()
        status = f'Loaded Lenstool model: {parfile} at zs={zs0:.4g}'
        if removed_members:
            status += '; cleared the cluster-member catalog to avoid double-counting and slow grid updates.'
        self.status_label.setText(status)
        return True

    def create_lenstool_grid_model(self, prefix, zl, zs):
        filepot = prefix + '_pot.fits'
        deflector_model = lst.create_deflector(
            parfile=prefix + '.par',
            filex=prefix + '_angx.fits',
            filey=prefix + '_angy.fits',
            filepot=filepot if os.path.exists(filepot) else None,
            usePotential=False,
            zl=zl,
            zs=float(zs),
            zsnorm=1.0,
            resc_fact=1.0,
            compute_potential=False,
        )
        return LenstoolGridModel(deflector_model)

    def active_lenstool_model(self, zs):
        if not self.use_lenstool_check.isChecked():
            return None
        if self.lenstool_model is None:
            prefix = self.current_prefix
            if not prefix or not self.load_lenstool_model(prefix):
                return None
        if abs(float(self.lenstool_model.zs) - float(zs)) > 1.0e-8:
            self.lenstool_model.change_redshift(float(zs))
        return self.lenstool_model

    def choose_observed_family_catalog(self, _checked=False):
        start_dir = os.path.dirname(self.current_parfile) if self.current_parfile else os.getcwd()
        local_catalog = os.path.join(os.path.dirname(__file__), 'obs_arcs_S1063.dat')
        if os.path.exists(local_catalog):
            start_dir = os.path.dirname(local_catalog)
        path, _ = QFileDialog.getOpenFileName(
            self,
            'Load observed image-family catalog',
            start_dir,
            'Lenstool image catalogs (*.dat *.cat *.txt *.all *.lenstool);;All files (*)',
        )
        if not path:
            return
        try:
            families = self.parse_observed_family_catalog(path)
        except Exception as exc:
            QMessageBox.warning(self, 'Could not load observed families', str(exc))
            return
        self.observed_family_catalog_path = path
        self.observed_families = families
        self.show_observed_families_check.setEnabled(True)
        self.show_observed_families_check.setChecked(True)
        self.redraw_observed_family_markers()
        n_images = sum(len(family['images']) for family in families)
        self.status_label.setText(
            f'Loaded {n_images} observed image(s) in {len(families)} family/families from {path}.'
        )

    def parse_observed_family_catalog(self, path):
        families = {}
        family_order = []
        with open(path, 'r', encoding='utf-8') as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped or stripped.startswith('#'):
                    continue
                values = stripped.split()
                if len(values) < 3:
                    raise ValueError(
                        f'Catalog line {line_number} has {len(values)} columns; expected at least id RA DEC.'
                    )
                image_id = values[0]
                family_id = self.observed_family_id(image_id)
                ra = float(values[1])
                dec = float(values[2])
                x, y = self.radec_to_scene(ra, dec)
                redshift = float(values[6]) if len(values) > 6 else None
                if family_id not in families:
                    families[family_id] = {'id': family_id, 'images': []}
                    family_order.append(family_id)
                families[family_id]['images'].append({
                    'id': image_id,
                    'ra': ra,
                    'dec': dec,
                    'x': x,
                    'y': y,
                    'redshift': redshift,
                })
        if not family_order:
            raise ValueError('The catalog does not contain any image rows.')
        return [families[family_id] for family_id in family_order]

    def observed_family_id(self, image_id):
        token = str(image_id)
        if '.' in token:
            family_id = token.split('.', 1)[0]
            return family_id or token
        prefix = []
        for char in token:
            if char.isdigit():
                prefix.append(char)
            else:
                break
        return ''.join(prefix) or token

    def reset_observed_family_catalog(self):
        self.observed_family_catalog_path = None
        self.observed_families = []
        if hasattr(self, 'observed_family_items'):
            self.remove_items(self.observed_family_items)
        self.observed_family_items = []
        if hasattr(self, 'show_observed_families_check'):
            self.show_observed_families_check.blockSignals(True)
            self.show_observed_families_check.setChecked(True)
            self.show_observed_families_check.setEnabled(False)
            self.show_observed_families_check.blockSignals(False)

    def redraw_observed_family_markers(self):
        self.remove_items(getattr(self, 'observed_family_items', []))
        self.observed_family_items = []
        if (
            not getattr(self, 'observed_families', None)
            or not self.show_observed_families_check.isChecked()
        ):
            return
        for family in self.observed_families:
            for image in family['images']:
                label = image['id']
                if image.get('redshift') is not None:
                    label = f'{label}'
                self.observed_family_items.extend(
                    self.add_labeled_marker(
                        self.image_scene,
                        image['x'],
                        image['y'],
                        label,
                        QColor(255, 150, 60),
                        zvalue=66,
                    )
                )

    def mark_family_point(self, point):
        self.selected_family_marker = None
        self.current_family_points.append(QPointF(point.x(), point.y()))
        self.redraw_current_family_markers()
        label = self.family_marker_label('current', len(self.current_family_points) - 1)
        self.status_label.setText(
            f'Marked image {label}. Press Add family when this family is complete.'
        )

    def redraw_current_family_markers(self):
        self.remove_items(self.current_family_items)
        self.current_family_items = []
        family_id = len(self.image_families) + 1
        for image_index, point in enumerate(self.current_family_points):
            selected = self.selected_family_marker == ('current', None, image_index)
            self.current_family_items.extend(
                self.add_labeled_marker(
                    self.image_scene,
                    point.x(),
                    point.y(),
                    f'{family_id}.{image_index + 1}',
                    self.family_marker_color(selected, current=True),
                    zvalue=80 if selected else 75,
                )
            )

    def add_current_family(self):
        if not self.current_family_points:
            QMessageBox.information(self, 'No images marked', 'Click image positions before adding a family.')
            return
        family = [QPointF(p.x(), p.y()) for p in self.current_family_points]
        self.image_families.append(family)
        family_id = len(self.image_families)
        self.current_family_points = []
        self.selected_family_marker = None
        self.remove_items(self.current_family_items)
        self.current_family_items = []
        self.redraw_family_markers()
        self.update_family_list()
        self.status_label.setText(f'Added family {family_id} with {len(family)} image(s).')

    def update_family_list(self):
        self.family_list_widget.clear()
        for i, family in enumerate(self.image_families, start=1):
            self.family_list_widget.addItem(f'Family {i}: {len(family)} image(s)')
        self.family_list_widget.setVisible(bool(self.image_families) and self.panel_is_visible('Families'))

    def redraw_family_markers(self):
        self.remove_items(self.family_items)
        self.family_items = []
        for family_id, family in enumerate(self.image_families, start=1):
            for image_index, point in enumerate(family):
                selected = self.selected_family_marker == ('saved', family_id - 1, image_index)
                self.family_items.extend(
                    self.add_labeled_marker(
                        self.image_scene,
                        point.x(),
                        point.y(),
                        f'{family_id}.{image_index + 1}',
                        self.family_marker_color(selected, current=False),
                        zvalue=80 if selected else 74,
                    )
                )

    def family_marker_color(self, selected, current):
        if selected:
            return QColor(255, 235, 70)
        if current:
            return QColor(70, 220, 120)
        return QColor(70, 170, 255)

    def family_marker_label(self, marker_kind, image_index, family_index=None):
        if marker_kind == 'current':
            return f'{len(self.image_families) + 1}.{image_index + 1}'
        return f'{family_index + 1}.{image_index + 1}'

    def marker_pick_radius(self):
        return max(0.8, 2.5 * self.handle_radius())

    def select_family_marker_at(self, point):
        marker = self.find_family_marker_at(point)
        if marker is None:
            self.selected_family_marker = None
            self.redraw_current_family_markers()
            self.redraw_family_markers()
            return False
        self.selected_family_marker = marker
        self.selected_sie_index = None
        self.remove_sie_handles()
        self.draw_all_sie_ellipses()
        self.redraw_current_family_markers()
        self.redraw_family_markers()
        marker_kind, family_index, image_index = marker
        label = self.family_marker_label(marker_kind, image_index, family_index)
        self.status_label.setText(f'Selected image marker {label}. Press Delete or Backspace to remove it.')
        return True

    def find_family_marker_at(self, point):
        best_marker = None
        best_distance = self.marker_pick_radius()
        for image_index, candidate in enumerate(self.current_family_points):
            distance = self.distance(point, candidate)
            if distance <= best_distance:
                best_marker = ('current', None, image_index)
                best_distance = distance
        for family_index, family in enumerate(self.image_families):
            for image_index, candidate in enumerate(family):
                distance = self.distance(point, candidate)
                if distance <= best_distance:
                    best_marker = ('saved', family_index, image_index)
                    best_distance = distance
        return best_marker

    def delete_selected_object(self):
        if self.selected_family_marker is not None:
            self.delete_selected_family_marker()
            return
        self.delete_selected_sie()

    def delete_selected_family_marker(self):
        marker_kind, family_index, image_index = self.selected_family_marker
        label = self.family_marker_label(marker_kind, image_index, family_index)
        if marker_kind == 'current':
            if 0 <= image_index < len(self.current_family_points):
                del self.current_family_points[image_index]
        elif 0 <= family_index < len(self.image_families):
            family = self.image_families[family_index]
            if 0 <= image_index < len(family):
                del family[image_index]
            if not family:
                del self.image_families[family_index]
        self.selected_family_marker = None
        self.redraw_current_family_markers()
        self.redraw_family_markers()
        self.update_family_list()
        self.status_label.setText(f'Removed image marker {label}.')

    def export_families(self):
        if not self.image_families:
            QMessageBox.information(self, 'No families', 'No image families have been added yet.')
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            'Export image families',
            os.path.join(os.getcwd(), 'image_families.lenstool'),
            'Lenstool image catalogs (*.lenstool *.all *.txt);;All files (*)',
        )
        if not path:
            return
        if self.reference_radec is None and self.rgb_wcs is None:
            QMessageBox.warning(
                self,
                'No sky coordinates available',
                'No Lenstool reference coordinate or RGB WCS is available. '
                'Exporting scene coordinates in the RA/DEC columns.',
            )
        self.write_lenstool_families(path)
        self.status_label.setText(f'Exported {len(self.image_families)} family/families to {path}.')

    def write_lenstool_families(self, path):
        with open(path, 'w', encoding='utf-8') as handle:
            handle.write('#REFERENCE 3 0.0000000 0.0000000\n')
            for family_id, family in enumerate(self.image_families, start=1):
                for image_id, point in enumerate(family, start=1):
                    ra, dec = self.scene_to_radec(point.x(), point.y())
                    row_id = f'{family_id}.{image_id}'
                    handle.write(
                        f'{row_id} {ra:.8f} {dec:.8f} 1.0 1.0 0.0 0.0 0.0\n'
                    )

    def scene_to_radec(self, x, y):
        if self.reference_radec is not None:
            ra_ref, dec_ref = self.reference_radec
            ra, dec = lst.getRADECfromXY(float(x), float(y), ra_ref, dec_ref, type=0)
            return float(ra), float(dec)
        if self.rgb_wcs is None or self.rgb_shape is None:
            return float(x), float(y)
        ny, nx = self.rgb_shape
        xmin, xmax, ymin, ymax = self.extent
        xpix = (float(x) - xmin) / (xmax - xmin) * (nx - 1)
        ypix = (float(y) - ymin) / (ymax - ymin) * (ny - 1)
        ra, dec = self.rgb_wcs.pixel_to_world_values(xpix, ypix)
        return float(ra), float(dec)

    def selected_component(self):
        if self.selected_sie_index is None:
            return None
        if not (0 <= self.selected_sie_index < len(self.sie_components)):
            return None
        return self.sie_components[self.selected_sie_index]

    def component_from_controls(self, center=None, theta_e=None, existing=None, renormalize=True):
        center = self.sie_center if center is None else center
        theta_e = self.theta_e_spin.value() if theta_e is None else theta_e
        component = {
            'model_type': self.profile_combo.currentText(),
            'center': QPointF(center.x(), center.y()),
            'theta_e': float(theta_e),
            'q': float(self.q_spin.value()),
            'pa_deg': float(self.pa_spin.value()),
            'theta_c': float(self.core_spin.value()),
            'theta_t': float(self.theta_t_spin.value()),
        }
        needs_norm = (
            existing is None
            or 'sigma0' not in existing
            or existing.get('needs_renormalization', False)
            or self.component_normalization_signature(component)
            != self.component_normalization_signature(existing)
        )
        if existing is not None and 'sigma0' in existing and not (renormalize and needs_norm):
            component['sigma0'] = float(existing['sigma0'])
            component['normalization_zs'] = float(existing.get('normalization_zs', self.zs_spin.value()))
            component['needs_renormalization'] = bool(needs_norm)
            return component
        if renormalize and needs_norm:
            zl = float(self.zl_spin.value())
            zs = float(self.zs_spin.value())
            component['sigma0'] = self.component_sigma0(component, zl, zs)
            component['normalization_zs'] = zs
            component['needs_renormalization'] = False
        elif existing is not None and 'sigma0' in existing:
            component['sigma0'] = float(existing['sigma0'])
            component['normalization_zs'] = float(existing.get('normalization_zs', self.zs_spin.value()))
            component['needs_renormalization'] = bool(needs_norm)
        else:
            component['needs_renormalization'] = True
        return component

    def component_normalization_signature(self, component):
        return (
            component.get('model_type', 'SIE'),
            round(float(component['theta_e']), 10),
            round(float(component['q']), 10),
            round(float(component['theta_c']), 10),
            round(float(component.get('theta_t', self.theta_t_spin.value())), 10),
        )

    def component_sigma0(self, component, zl, zs):
        if component.get('model_type', 'SIE') == 'PIEMD':
            return self.sigma0_for_piemd_einstein_radius(component, zl, zs)
        return sigma0_from_theta_e(float(component['theta_e']), float(component['q']), self.co, zl, zs)

    def load_component_into_controls(self, component):
        self.sie_center = QPointF(component['center'].x(), component['center'].y())
        self.sie_theta_e = float(component['theta_e'])
        self.profile_combo.setCurrentText(component.get('model_type', 'SIE'))
        self.theta_e_spin.setValue(float(component['theta_e']))
        self.q_spin.setValue(float(component['q']))
        self.pa_spin.setValue(float(component['pa_deg']))
        self.core_spin.setValue(float(component['theta_c']))
        self.theta_t_spin.setValue(float(component.get('theta_t', self.theta_t_spin.value())))

    def store_selected_component_from_controls(self, renormalize=True):
        component = self.selected_component()
        if component is None:
            return
        component.update(self.component_from_controls(existing=component, renormalize=renormalize))

    def select_sie_component(self, index):
        if index is None or not (0 <= index < len(self.sie_components)):
            self.selected_sie_index = None
            self.remove_sie_handles()
            self.draw_all_sie_ellipses()
            return
        self.selected_family_marker = None
        self.redraw_current_family_markers()
        self.redraw_family_markers()
        self.selected_sie_index = index
        self.pending_new_sie = False
        self.loading_component_controls = True
        self.load_component_into_controls(self.sie_components[index])
        self.loading_component_controls = False
        self.draw_all_sie_ellipses()
        model_type = self.sie_components[index].get('model_type', 'SIE')
        self.status_label.setText(f'Selected {model_type} component {index + 1} of {len(self.sie_components)}.')

    def delete_selected_sie(self):
        if self.selected_sie_index is None:
            self.status_label.setText('No mass component selected.')
            return
        deleted_index = self.selected_sie_index
        del self.sie_components[deleted_index]
        if self.sie_components:
            self.selected_sie_index = min(deleted_index, len(self.sie_components) - 1)
            self.loading_component_controls = True
            self.load_component_into_controls(self.sie_components[self.selected_sie_index])
            self.loading_component_controls = False
        else:
            self.selected_sie_index = None
            self.remove_items(self.sie_items)
            self.sie_items = []
            self.sie_item = None
            self.remove_sie_handles()
            self.update_sie_from_controls()
            self.status_label.setText('Removed the last analytic component.')
            return
        self.update_sie_from_controls()
        self.status_label.setText(f'Removed mass component {deleted_index + 1}.')

    def begin_sie_interaction(self, point):
        self.drag_start_point = QPointF(point.x(), point.y())
        self.drag_start_center = QPointF(self.sie_center.x(), self.sie_center.y())
        self.active_sie_drag = self.sie_edit_target(point)
        self.clear_curve_overlays()
        if self.active_sie_drag == 'draw':
            self.sie_center = QPointF(point.x(), point.y())
            self.remove_sie_handles()
            self.update_sie_preview(point)
        elif self.active_sie_drag == 'select':
            index = self.find_sie_component_at_point(point)
            self.select_sie_component(index)
        elif self.active_sie_drag == 'move':
            self.status_label.setText('Dragging SIE center')
        elif self.active_sie_drag == 'axis_ratio':
            self.status_label.setText('Dragging q handle')
        elif self.active_sie_drag == 'size_orientation':
            self.status_label.setText('Dragging size/orientation handle')

    def update_sie_interaction(self, point):
        if self.active_sie_drag == 'move':
            self.move_sie_preview(point)
        elif self.active_sie_drag == 'axis_ratio':
            self.update_axis_ratio_preview(point)
        elif self.active_sie_drag == 'size_orientation':
            self.update_size_orientation_preview(point)
        elif self.active_sie_drag == 'select':
            return
        else:
            self.update_sie_preview(point)

    def finish_sie_interaction(self, point):
        if self.active_sie_drag == 'move':
            self.move_sie_preview(point)
            self.active_sie_drag = None
            self.update_sie_from_controls()
            return
        if self.active_sie_drag == 'axis_ratio':
            self.update_axis_ratio_preview(point)
            self.active_sie_drag = None
            self.update_sie_from_controls()
            return
        if self.active_sie_drag == 'size_orientation':
            self.update_size_orientation_preview(point)
            self.active_sie_drag = None
            self.update_sie_from_controls()
            return
        if self.active_sie_drag == 'select':
            self.active_sie_drag = None
            self.draw_critical_curves()
            return

        radius, pa_deg = self.sie_radius_and_pa(point)
        self.active_sie_drag = None
        if radius <= 0:
            return
        self.theta_e_spin.setValue(radius)
        self.pa_spin.setValue(pa_deg)
        self.sie_theta_e = radius
        if self.selected_sie_index is None:
            self.sie_components.append(
                self.component_from_controls(center=self.sie_center, theta_e=radius, renormalize=True)
            )
            self.selected_sie_index = len(self.sie_components) - 1
            self.pending_new_sie = False
        else:
            self.store_selected_component_from_controls()
        self.update_sie_from_controls()

    def sie_edit_target(self, point):
        if self.pending_new_sie or not self.sie_components:
            return 'draw'
        if self.selected_component() is not None:
            if self.is_near_point(point, self.major_handle_position(), scale=1.7):
                return 'size_orientation'
            if self.is_near_point(point, self.q_handle_position(), scale=1.7):
                return 'axis_ratio'
            if self.is_point_inside_sie(point) or self.is_near_point(point, self.sie_center, scale=1.7):
                return 'move'
        if self.find_sie_component_at_point(point) is not None:
            return 'select'
        self.status_label.setText('Use Add component to draw an additional component.')
        return 'select'

    def move_sie_preview(self, point):
        if self.drag_start_point is None or self.drag_start_center is None:
            return
        dx = point.x() - self.drag_start_point.x()
        dy = point.y() - self.drag_start_point.y()
        self.sie_center = QPointF(self.drag_start_center.x() + dx, self.drag_start_center.y() + dy)
        self.store_selected_component_from_controls(renormalize=False)
        self.draw_sie_ellipse(float(self.theta_e_spin.value()), preview=False)
        self.status_label.setText(
            f'SIE center=({self.sie_center.x():.3f}, {self.sie_center.y():.3f}), '
            f'q={self.q_spin.value():.3f}, PA={self.pa_spin.value():.1f} deg'
        )

    def update_axis_ratio_preview(self, point):
        theta_e = float(self.theta_e_spin.value())
        if theta_e <= 0:
            return
        _, minor = self.sie_display_axes()
        dx = point.x() - self.sie_center.x()
        dy = point.y() - self.sie_center.y()
        minor_radius = abs(dx * minor[0] + dy * minor[1])
        q = float(np.clip(minor_radius / theta_e, self.q_spin.minimum(), self.q_spin.maximum()))
        self.q_spin.setValue(q)
        self.store_selected_component_from_controls(renormalize=False)
        self.draw_sie_ellipse(theta_e, preview=False)
        self.status_label.setText(
            f'SIE center=({self.sie_center.x():.3f}, {self.sie_center.y():.3f}), '
            f'theta_E={theta_e:.3f}, q={q:.3f}'
        )

    def update_size_orientation_preview(self, point):
        radius, pa_deg = self.sie_radius_and_pa(point)
        if radius <= 0:
            return
        self.theta_e_spin.setValue(radius)
        self.pa_spin.setValue(pa_deg)
        self.sie_theta_e = radius
        self.store_selected_component_from_controls(renormalize=False)
        self.draw_sie_ellipse(radius, preview=False)
        self.status_label.setText(
            f'SIE center=({self.sie_center.x():.3f}, {self.sie_center.y():.3f}), '
            f'theta_E={radius:.3f}, PA={pa_deg:.1f} deg'
        )

    def update_sie_preview(self, point):
        radius, pa_deg = self.sie_radius_and_pa(point)
        if radius <= 0:
            return
        self.pa_spin.setValue(pa_deg)
        self.draw_sie_ellipse(radius, preview=True)
        self.status_label.setText(
            f'SIE center=({self.sie_center.x():.3f}, {self.sie_center.y():.3f}), '
            f'theta_E={radius:.3f}, PA={pa_deg:.1f} deg'
        )

    def distance(self, p1, p2):
        return float(np.hypot(p2.x() - p1.x(), p2.y() - p1.y()))

    def handle_radius(self):
        return max(0.125, 0.005 * max(self.extent[1] - self.extent[0], self.extent[3] - self.extent[2]))

    def is_near_point(self, point, target, scale=1.0):
        return self.distance(point, target) <= scale * self.handle_radius()

    def sie_display_axes(self):
        pa = np.deg2rad(float(self.pa_spin.value()))
        major = np.array([np.cos(pa), np.sin(pa)])
        minor = np.array([-np.sin(pa), np.cos(pa)])
        return major, minor

    def q_handle_position(self):
        _, minor = self.sie_display_axes()
        theta_e = float(self.theta_e_spin.value())
        q = float(self.q_spin.value())
        return QPointF(
            self.sie_center.x() + theta_e * q * minor[0],
            self.sie_center.y() + theta_e * q * minor[1],
        )

    def major_handle_position(self):
        major, _ = self.sie_display_axes()
        theta_e = float(self.theta_e_spin.value())
        return QPointF(
            self.sie_center.x() + theta_e * major[0],
            self.sie_center.y() + theta_e * major[1],
        )

    def find_sie_component_at_point(self, point):
        for index in range(len(self.sie_components) - 1, -1, -1):
            if self.is_point_inside_component(point, self.sie_components[index]):
                return index
        return None

    def is_point_inside_sie(self, point):
        component = self.selected_component()
        if component is not None:
            return self.is_point_inside_component(point, component)
        theta_e = float(self.theta_e_spin.value())
        q = float(self.q_spin.value())
        if theta_e <= 0 or q <= 0:
            return False
        major, minor = self.sie_display_axes()
        dx = point.x() - self.sie_center.x()
        dy = point.y() - self.sie_center.y()
        major_coord = dx * major[0] + dy * major[1]
        minor_coord = dx * minor[0] + dy * minor[1]
        return (major_coord / theta_e) ** 2 + (minor_coord / (theta_e * q)) ** 2 <= 1.0

    def is_point_inside_component(self, point, component):
        theta_e = float(component['theta_e'])
        q = float(component['q'])
        if theta_e <= 0 or q <= 0:
            return False
        major, minor = self.component_axes(component)
        dx = point.x() - component['center'].x()
        dy = point.y() - component['center'].y()
        major_coord = dx * major[0] + dy * major[1]
        minor_coord = dx * minor[0] + dy * minor[1]
        return (major_coord / theta_e) ** 2 + (minor_coord / (theta_e * q)) ** 2 <= 1.0

    def component_axes(self, component):
        pa = np.deg2rad(float(component['pa_deg']))
        major = np.array([np.cos(pa), np.sin(pa)])
        minor = np.array([-np.sin(pa), np.cos(pa)])
        return major, minor

    def sie_radius_and_pa(self, point):
        dx = point.x() - self.sie_center.x()
        dy = point.y() - self.sie_center.y()
        radius = float(np.hypot(dx, dy))
        if radius <= 0:
            return 0.0, float(self.pa_spin.value())
        pa_deg = float(np.rad2deg(np.arctan2(dy, dx)))
        return radius, pa_deg

    def force_update_model(self, _checked=False):
        focus_widget = QApplication.focusWidget()
        if focus_widget is not None:
            focus_widget.clearFocus()
        self.commit_numeric_edits()
        self.update_sie_from_controls(force=True)

    def update_sie_from_controls(self, *_args, force=False):
        if self.loading_component_controls:
            return
        self.commit_numeric_edits()
        zl = float(self.zl_spin.value())
        zs = float(self.zs_spin.value())
        nray = int(self.nray_spin.value())
        base_model = self.active_lenstool_model(zs)
        zl = float(self.zl_spin.value())
        try:
            member_layer = self.member_grid_layer(zl, zs, base_model, nray) if self.members_enabled() else None
        except ValueError as exc:
            QMessageBox.warning(self, 'Invalid member-galaxy configuration', str(exc))
            return
        member_count = self.member_layer_cache_count if member_layer is not None else 0

        if base_model is None and not self.sie_components and member_layer is None:
            self.lens = None
            self.selected_sie_index = None
            self.remove_items(self.sie_items)
            self.sie_items = []
            self.sie_item = None
            self.remove_sie_handles()
            self.clear_curve_overlays()
            self.clear_prediction_markers(clear_table=True, clear_anchor=True)
            self.status_label.setText(
                'No lens model yet. Draw the first component, press Add component, or enable a Lenstool model.'
            )
            return

        if self.selected_component() is not None:
            self.store_selected_component_from_controls()

        models = []
        if base_model is not None:
            models.append(base_model)
        if member_layer is not None:
            models.append(member_layer)
        try:
            models.extend(self.model_from_component(component, zl, zs) for component in self.sie_components)
        except ValueError as exc:
            QMessageBox.warning(self, 'Invalid SIE configuration', str(exc))
            return

        lens = compositeModel(self.co, models=models)
        lens.zl = zl
        lens.zs = zs
        lens.ds = self.co.angular_diameter_distance(zs)
        lens.dls = self.co.angular_diameter_distance_z1z2(zl, zs)
        lens.dl = self.co.angular_diameter_distance(zl)
        if base_model is None:
            theta = np.linspace(self.extent[0], self.extent[1], nray)
            lens.setGrid(theta=theta, compute_potential=False)
        else:
            lens.setGrid(
                thetax=np.asarray(base_model.thetax),
                thetay=np.asarray(base_model.thetay),
                compute_potential=False,
            )
        self.lens = lens
        self.model_update_counter += 1
        selected = self.selected_component()
        if selected is not None:
            self.loading_component_controls = True
            self.load_component_into_controls(selected)
            self.loading_component_controls = False
        self.draw_all_sie_ellipses()
        self.draw_critical_curves()
        if self.last_prediction_anchor is None:
            self.clear_prediction_markers(clear_table=True, clear_anchor=False)
            base_label = ' + Lenstool base' if base_model is not None else ''
            member_label = f' + {member_count} member galaxies' if member_count else ''
            self.status_label.setText(
                f'Lens update #{self.model_update_counter}: zs={zs:.4g}, '
                f'Nray={lens.nray1} with {len(self.sie_components)} analytic component(s)'
                f'{base_label}{member_label}.'
            )
        else:
            self.refresh_prediction_from_anchor()
            if force:
                self.status_label.setText(
                    f'Lens update #{self.model_update_counter}: zs={zs:.4g}, '
                    f'Nray={lens.nray1}; refreshed the current prediction.'
                )

    def commit_numeric_edits(self):
        for spin in [
            self.theta_e_spin,
            self.q_spin,
            self.pa_spin,
            self.core_spin,
            self.theta_t_spin,
            self.zl_spin,
            self.zs_spin,
            self.nray_spin,
        ]:
            spin.interpretText()
        for spin in [
            getattr(self, 'member_mag0_spin', None),
            getattr(self, 'member_sigma0_spin', None),
            getattr(self, 'member_alpha_spin', None),
            getattr(self, 'member_cut0_spin', None),
            getattr(self, 'member_beta_spin', None),
            getattr(self, 'member_core0_spin', None),
            getattr(self, 'member_magmax_spin', None),
        ]:
            if spin is not None:
                spin.interpretText()

    def member_parameter_arrays(self, zl, zs):
        if not self.member_catalog:
            return None
        mag0 = float(self.member_mag0_spin.value())
        sigma0_0 = float(self.member_sigma0_spin.value())
        alpha = float(self.member_alpha_spin.value())
        cut0 = float(self.member_cut0_spin.value())
        beta = float(self.member_beta_spin.value())
        core0 = float(self.member_core0_spin.value())
        mag_max = float(self.member_magmax_spin.value())
        if cut0 <= core0:
            raise ValueError('Member cut_0 must be larger than core_0.')

        selected = [member for member in self.member_catalog if float(member['mag']) <= mag_max]
        if not selected:
            return None

        mag = np.asarray([member['mag'] for member in selected], dtype=float)
        theta_c = np.asarray(
            lst.core_radius_from_mag(mag, mag_0=mag0, core_radius_0=core0),
            dtype=float,
        )
        theta_t = np.asarray(
            lst.cut_radius_from_mag(mag, beta=beta, mag_0=mag0, cut_radius_0=cut0),
            dtype=float,
        )
        invalid = np.where(theta_t <= theta_c)[0]
        if invalid.size:
            member = selected[int(invalid[0])]
            raise ValueError(
                f'Member galaxy {member["id"]} has theta_t <= theta_c. '
                'Increase cut_0, decrease core_0, or reduce beta.'
            )

        return {
            'x1': np.asarray([member['x'] for member in selected], dtype=float),
            'x2': np.asarray([member['y'] for member in selected], dtype=float),
            'q': np.clip(np.asarray([member['q'] for member in selected], dtype=float), 0.01, 1.0),
            'pa': np.asarray(
                [self.sie_model_pa_from_display_pa(member['pa_deg']) for member in selected],
                dtype=float,
            ),
            'sigma0': np.asarray(
                lst.v_disp_from_mag(mag, alpha=alpha, mag_0=mag0, v_disp_0=sigma0_0),
                dtype=float,
            ),
            'theta_c': theta_c,
            'theta_t': theta_t,
        }

    def member_grid_layer(self, zl, zs, base_model, nray):
        cache_key = self.member_layer_key(zl, zs, base_model, nray)
        if cache_key == self.member_layer_cache_key:
            return self.member_layer_cache

        params = self.member_parameter_arrays(zl, zs)
        self.member_layer_cache_count = 0 if params is None else int(params['x1'].size)
        if params is None:
            self.member_layer_cache_key = cache_key
            self.member_layer_cache = None
            return None

        if base_model is None:
            thetax = np.linspace(self.extent[0], self.extent[1], int(nray))
            thetay = thetax
        else:
            thetax = np.asarray(base_model.thetax, dtype=float)
            thetay = np.asarray(base_model.thetay, dtype=float)

        physical_key = self.member_physical_layer_key(zl, base_model, nray)
        layer_zs = float(zs)
        if physical_key == self.member_physical_cache_key and self.member_physical_cache_maps is not None:
            layer_zs = float(self.member_physical_cache_zs)
            a1, a2, ka, g1, g2 = (
                np.asarray(values, dtype=float).copy()
                for values in self.member_physical_cache_maps
            )
        else:
            a1, a2, ka, g1, g2 = self.evaluate_member_grid_maps(params, thetax, thetay, zl, zs)
            self.member_physical_cache_key = physical_key
            self.member_physical_cache_zs = float(zs)
            self.member_physical_cache_maps = (
                a1.copy(),
                a2.copy(),
                ka.copy(),
                g1.copy(),
                g2.copy(),
            )

        self.member_layer_cache_key = cache_key
        self.member_layer_cache = PrecomputedGridModel(
            self.co,
            zl,
            layer_zs,
            thetax,
            thetay,
            a1,
            a2,
            ka,
            g1,
            g2,
        )
        if abs(float(self.member_layer_cache.zs) - float(zs)) > 1.0e-8:
            self.member_layer_cache.change_redshift(float(zs))
        return self.member_layer_cache

    def evaluate_member_grid_maps(self, params, thetax, thetay, zl, zs):
        theta1, theta2 = np.meshgrid(thetax, thetay)
        a1 = np.zeros_like(theta1, dtype=float)
        a2 = np.zeros_like(theta1, dtype=float)
        ka = np.zeros_like(theta1, dtype=float)
        g1 = np.zeros_like(theta1, dtype=float)
        g2 = np.zeros_like(theta1, dtype=float)
        # A single-member chunk keeps temporaries 2D rather than 3D. For the
        # current NumPy backend this is usually faster and much more memory
        # stable than broadcasting many galaxies across the full grid at once.
        chunk_size = 1
        ds = self.co.angular_diameter_distance(zs).value
        dls = self.co.angular_diameter_distance_z1z2(zl, zs).value
        if dls <= 0:
            raise ValueError('Source redshift must be larger than lens redshift.')
        conv = 180.0 / np.pi * 3600.0
        c_kms = 299792.458

        for start in range(0, params['x1'].size, chunk_size):
            stop = min(start + chunk_size, params['x1'].size)
            chunk = {key: value[start:stop] for key, value in params.items()}
            bsie = (
                conv
                * 4.0
                * np.pi
                * chunk['sigma0'] ** 2
                / c_kms ** 2
                * dls
                / ds
                / np.sqrt(chunk['q'])
            )
            core_maps = self.evaluate_sie_chunk(theta1, theta2, chunk, bsie, chunk['theta_c'])
            cut_maps = self.evaluate_sie_chunk(theta1, theta2, chunk, bsie, chunk['theta_t'])
            a1 += core_maps[0] - cut_maps[0]
            a2 += core_maps[1] - cut_maps[1]
            ka += core_maps[2] - cut_maps[2]
            g1 += core_maps[3] - cut_maps[3]
            g2 += core_maps[4] - cut_maps[4]
        return a1, a2, ka, g1, g2

    def evaluate_sie_chunk(self, theta1, theta2, params, bsie, theta_core):
        x1 = params['x1'][:, None, None]
        x2 = params['x2'][:, None, None]
        q = params['q'][:, None, None]
        pa = params['pa'][:, None, None]
        s = (theta_core / np.sqrt(params['q']))[:, None, None]
        b = bsie[:, None, None]
        sin_pa = np.sin(pa)
        cos_pa = np.cos(pa)

        dx = theta1[None, :, :] - x1
        dy = theta2[None, :, :] - x2
        tx = dx * sin_pa - dy * cos_pa
        ty = dx * cos_pa + dy * sin_pa
        psi = np.sqrt(q ** 2 * (s ** 2 + tx ** 2) + ty ** 2)
        ka = b / 2.0 / np.sqrt(s ** 2 + tx ** 2 + ty ** 2 / q ** 2)

        alphax = np.empty_like(tx)
        alphay = np.empty_like(ty)
        circular = np.abs(params['q'] - 1.0) < 1.0e-8
        if np.any(circular):
            alphax[circular] = b[circular] * tx[circular] / (psi[circular] + s[circular])
            alphay[circular] = b[circular] * ty[circular] / (psi[circular] + s[circular])
        if np.any(~circular):
            qn = q[~circular]
            bn = b[~circular]
            sn = s[~circular]
            tx_n = tx[~circular]
            ty_n = ty[~circular]
            psi_n = psi[~circular]
            root = np.sqrt(1.0 - qn ** 2)
            prefactor = bn * qn / root
            alphax[~circular] = prefactor * np.arctan(root * tx_n / (psi_n + sn))
            arg = root * ty_n / (psi_n + qn ** 2 * sn)
            alphay[~circular] = prefactor * np.arctanh(np.clip(arg, -1.0 + 1.0e-12, 1.0 - 1.0e-12))

        a1 = alphax * sin_pa + alphay * cos_pa
        a2 = -alphax * cos_pa + alphay * sin_pa

        den = (1.0 + q ** 2) * s ** 2 + 2.0 * psi * s + tx ** 2 + ty ** 2
        psi11 = b * q / psi * (q ** 2 * s ** 2 + ty ** 2 + s * psi) / den
        psi22 = b * q / psi * (s ** 2 + tx ** 2 + s * psi) / den
        psi12 = -b * q / psi * (tx * ty) / den
        psi11_r = psi11 * sin_pa ** 2 + 2.0 * psi12 * sin_pa * cos_pa + psi22 * cos_pa ** 2
        psi22_r = psi11 * cos_pa ** 2 - 2.0 * psi12 * sin_pa * cos_pa + psi22 * sin_pa ** 2
        psi12_r = (
            -psi11 * sin_pa * cos_pa
            + psi12 * (sin_pa ** 2 - cos_pa ** 2)
            + psi22 * sin_pa * cos_pa
        )
        g1 = 0.5 * (psi11_r - psi22_r)
        g2 = psi12_r
        return (
            np.sum(a1, axis=0),
            np.sum(a2, axis=0),
            np.sum(ka, axis=0),
            np.sum(g1, axis=0),
            np.sum(g2, axis=0),
        )

    def member_layer_key(self, zl, zs, base_model, nray):
        params = (
            round(float(self.member_mag0_spin.value()), 8),
            round(float(self.member_sigma0_spin.value()), 8),
            round(float(self.member_alpha_spin.value()), 8),
            round(float(self.member_cut0_spin.value()), 8),
            round(float(self.member_beta_spin.value()), 8),
            round(float(self.member_core0_spin.value()), 10),
            round(float(self.member_magmax_spin.value()), 8),
        )
        if base_model is None:
            grid_key = (
                'analytic-grid',
                int(nray),
                round(float(self.extent[0]), 8),
                round(float(self.extent[1]), 8),
                round(float(self.extent[2]), 8),
                round(float(self.extent[3]), 8),
            )
        else:
            grid_key = (
                'base-grid',
                int(base_model.nray1),
                int(base_model.nray2),
                round(float(base_model.thetax[0]), 8),
                round(float(base_model.thetax[-1]), 8),
                round(float(base_model.thetay[0]), 8),
                round(float(base_model.thetay[-1]), 8),
                round(float(base_model.pixel_scale), 10),
            )
        return (
            int(self.member_catalog_revision),
            round(float(zl), 8),
            round(float(zs), 8),
            params,
            grid_key,
            id(self.co),
        )

    def member_physical_layer_key(self, zl, base_model, nray):
        params = (
            round(float(self.member_mag0_spin.value()), 8),
            round(float(self.member_sigma0_spin.value()), 8),
            round(float(self.member_alpha_spin.value()), 8),
            round(float(self.member_cut0_spin.value()), 8),
            round(float(self.member_beta_spin.value()), 8),
            round(float(self.member_core0_spin.value()), 10),
            round(float(self.member_magmax_spin.value()), 8),
        )
        if base_model is None:
            grid_key = (
                'analytic-grid',
                int(nray),
                round(float(self.extent[0]), 8),
                round(float(self.extent[1]), 8),
                round(float(self.extent[2]), 8),
                round(float(self.extent[3]), 8),
            )
        else:
            grid_key = (
                'base-grid',
                int(base_model.nray1),
                int(base_model.nray2),
                round(float(base_model.thetax[0]), 8),
                round(float(base_model.thetax[-1]), 8),
                round(float(base_model.thetay[0]), 8),
                round(float(base_model.thetay[-1]), 8),
                round(float(base_model.pixel_scale), 10),
            )
        return (
            int(self.member_catalog_revision),
            round(float(zl), 8),
            params,
            grid_key,
            id(self.co),
        )

    def model_from_component(self, component, zl, zs):
        model_type = component.get('model_type', 'SIE')
        theta_e = float(component['theta_e'])
        q = float(component['q'])
        pa = self.sie_model_pa_from_display_pa(component['pa_deg'])
        common_kwargs = {
            'zl': zl,
            'zs': zs,
            'theta_c': float(component['theta_c']),
            'pa': pa,
            'q': q,
            'x1': component['center'].x(),
            'x2': component['center'].y(),
        }
        if model_type == 'PIEMD':
            theta_t = float(component.get('theta_t', self.theta_t_spin.value()))
            if theta_t <= common_kwargs['theta_c']:
                raise ValueError('PIEMD theta_t must be larger than core radius.')
            if component.get('needs_renormalization', False) or 'sigma0' not in component:
                component['sigma0'] = self.sigma0_for_piemd_einstein_radius(component, zl, zs)
                component['normalization_zs'] = float(zs)
                component['needs_renormalization'] = False
            sigma0 = float(component['sigma0'])
            return piemd(self.co, sigma0=sigma0, theta_t=theta_t, **common_kwargs)

        if component.get('needs_renormalization', False) or 'sigma0' not in component:
            component['sigma0'] = sigma0_from_theta_e(theta_e, q, self.co, zl, zs)
            component['normalization_zs'] = float(zs)
            component['needs_renormalization'] = False
        sigma0 = float(component['sigma0'])
        return sie(self.co, sigma0=sigma0, **common_kwargs)

    def sigma0_for_piemd_einstein_radius(self, component, zl, zs):
        theta_e = float(component['theta_e'])
        if theta_e <= 0:
            raise ValueError('PIEMD theta_E must be positive.')
        theta_t = float(component.get('theta_t', self.theta_t_spin.value()))
        theta_c = float(component['theta_c'])
        q = float(component['q'])
        target_theta_e = theta_e * np.sqrt(q)
        major, _ = self.component_axes(component)
        center = component['center']
        unit_model = piemd(
            self.co,
            zl=zl,
            zs=zs,
            theta_c=theta_c,
            theta_t=theta_t,
            pa=self.sie_model_pa_from_display_pa(component['pa_deg']),
            q=q,
            sigma0=1.0,
            x1=center.x(),
            x2=center.y(),
        )
        x_eval = center.x() + target_theta_e * major[0]
        y_eval = center.y() + target_theta_e * major[1]
        alpha1, alpha2 = unit_model.angle(x_eval, y_eval)
        alpha1 = float(np.ravel(np.asarray(alpha1))[0])
        alpha2 = float(np.ravel(np.asarray(alpha2))[0])
        alpha_radial = alpha1 * major[0] + alpha2 * major[1]
        if not np.isfinite(alpha_radial) or alpha_radial <= 0:
            raise ValueError('Could not normalize PIEMD to the requested theta_E.')
        sigma0 = float(np.sqrt(target_theta_e / alpha_radial))

        # Refine against the same equivalent-radius convention used by the SIE
        # normalization: the drawn ellipse has area-equivalent radius theta_E*sqrt(q).
        for _iteration in range(4):
            measured_theta_e = self.piemd_equivalent_critical_radius(component, zl, zs, sigma0)
            if not np.isfinite(measured_theta_e) or measured_theta_e <= 0:
                sigma0 *= 2.0
                continue
            sigma0 *= np.sqrt(target_theta_e / measured_theta_e)
        return sigma0

    def piemd_equivalent_critical_radius(self, component, zl, zs, sigma0):
        theta = np.linspace(
            self.extent[0],
            self.extent[1],
            min(max(int(self.nray_spin.value()), 128), 256),
        )
        model = piemd(
            self.co,
            zl=zl,
            zs=zs,
            theta_c=float(component['theta_c']),
            theta_t=float(component.get('theta_t', self.theta_t_spin.value())),
            pa=self.sie_model_pa_from_display_pa(component['pa_deg']),
            q=float(component['q']),
            sigma0=float(sigma0),
            x1=component['center'].x(),
            x2=component['center'].y(),
        )
        model.setGrid(theta=theta, compute_potential=False)
        lines = model.tancl(size_principale=0.0)
        if len(lines) == 0:
            return np.nan
        return float(lines[0].getThetaE() * model.pixel_scale)

    def draw_sie_ellipse(self, theta_e, preview=False):
        q = float(self.q_spin.value())
        pa_deg = float(self.pa_spin.value())
        center = self.sie_center
        if preview:
            if self.sie_preview_item is not None and self.sie_preview_item.scene() is not None:
                self.sie_preview_item.scene().removeItem(self.sie_preview_item)
            preview_component = {
                'model_type': self.profile_combo.currentText(),
                'center': center,
                'theta_e': float(theta_e),
                'q': q,
                'pa_deg': pa_deg,
                'theta_c': float(self.core_spin.value()),
                'theta_t': float(self.theta_t_spin.value()),
            }
            self.sie_preview_item = self.add_sie_component_item(
                preview_component,
                QColor(255, 220, 80, 170),
                zvalue=40,
            )
            self.remove_sie_handles()
            return

        self.store_selected_component_from_controls(renormalize=False)
        self.draw_all_sie_ellipses()

    def draw_all_sie_ellipses(self):
        self.remove_items(self.sie_items)
        self.sie_items = []
        self.sie_item = None
        for index, component in enumerate(self.sie_components):
            selected = index == self.selected_sie_index
            color = self.component_color(component, selected)
            zvalue = 34 if selected else 28
            item = self.add_sie_component_item(component, color, zvalue=zvalue)
            self.sie_items.append(item)
            if selected:
                self.sie_item = item

        if self.sie_preview_item is not None and self.sie_preview_item.scene() is not None:
            self.sie_preview_item.scene().removeItem(self.sie_preview_item)
            self.sie_preview_item = None
        if self.selected_component() is None:
            self.remove_sie_handles()
        else:
            self.draw_sie_handles()

    def component_color(self, component, selected):
        model_type = component.get('model_type', 'SIE')
        if model_type == 'PIEMD':
            return QColor(70, 210, 255, 245) if selected else QColor(70, 180, 255, 160)
        return QColor(255, 180, 0, 240) if selected else QColor(255, 150, 0, 150)

    def add_sie_component_item(self, component, color, zvalue):
        center = component['center']
        theta_e = float(component['theta_e'])
        q = float(component['q'])
        rect = QRectF(center.x() - theta_e, center.y() - theta_e * q, 2.0 * theta_e, 2.0 * theta_e * q)
        item = QGraphicsEllipseItem(rect)
        item.setTransformOriginPoint(center)
        item.setRotation(float(component['pa_deg']))
        pen = QPen(color, 2.0)
        pen.setCosmetic(True)
        item.setPen(pen)
        item.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        item.setZValue(zvalue)
        self.image_scene.addItem(item)
        return item

    def draw_sie_handles(self):
        self.remove_sie_handles()
        center_color = QColor(60, 150, 255, 230)
        major_color = QColor(255, 90, 210, 240)
        q_color = QColor(40, 240, 220, 240)
        self.sie_handle_items.append(
            self.add_handle(self.sie_center, center_color, zvalue=80)
        )
        self.sie_handle_items.append(
            self.add_handle(self.major_handle_position(), major_color, zvalue=83)
        )
        self.sie_handle_items.append(
            self.add_handle(self.q_handle_position(), q_color, zvalue=82)
        )

    def ensure_sie_handles_visible(self):
        if self.sie_item is None:
            return
        if len(self.sie_handle_items) != 3 or any(item.scene() is None for item in self.sie_handle_items):
            self.draw_sie_handles()
            return
        for item in self.sie_handle_items:
            item.setZValue(max(item.zValue(), 80))
            item.show()

    def add_handle(self, point, color, zvalue):
        radius = self.handle_radius()
        item = QGraphicsEllipseItem(
            QRectF(point.x() - radius, point.y() - radius, 2.0 * radius, 2.0 * radius)
        )
        pen = QPen(QColor(0, 0, 0), 1.2)
        pen.setCosmetic(True)
        item.setPen(pen)
        item.setBrush(QBrush(color))
        item.setZValue(zvalue)
        self.image_scene.addItem(item)
        return item

    def remove_sie_handles(self):
        self.remove_items(self.sie_handle_items)
        self.sie_handle_items = []

    def sie_model_pa_from_display_pa(self, pa_deg):
        # pyLensLib.sie defines pa for its rotated coordinates; the mass major
        # axis is offset by 90 degrees from Qt's visible ellipse rotation.
        return np.deg2rad(float(pa_deg) + 90.0)

    def draw_critical_curves(self):
        self.remove_items(self.critical_items)
        self.remove_items(self.caustic_items)
        self.critical_items = []
        self.caustic_items = []
        if self.lens is None:
            return

        try:
            tan_lines = self.lens.tancl(size_principale=0.0)
            rad_lines = self.lens.radcl(size_principale=0.0) if self.show_radial_check.isChecked() else []
            tan_caustics = self.lens.getCaustics(tan_lines)
            rad_caustics = self.lens.getCaustics(rad_lines) if self.show_radial_check.isChecked() else []
        except Exception as exc:
            self.status_label.setText(f'Could not compute critical curves: {exc}')
            return

        for cl in tan_lines:
            x, y = self.lens.getCritPoints(cl)
            self.critical_items.append(self.add_polyline(self.image_scene, x, y, QColor(255, 255, 255, 230), 1.8, 20))
        for cl in rad_lines:
            x, y = self.lens.getCritPoints(cl)
            self.critical_items.append(self.add_polyline(self.image_scene, x, y, QColor(110, 210, 255, 220), 1.4, 20))
        for cau in tan_caustics:
            x, y = self.lens.getCausticPoints(cau)
            self.caustic_items.append(self.add_polyline(self.source_scene, x, y, QColor(255, 220, 70, 240), 1.8, 20))
        for cau in rad_caustics:
            x, y = self.lens.getCausticPoints(cau)
            self.caustic_items.append(self.add_polyline(self.source_scene, x, y, QColor(110, 210, 255, 230), 1.4, 20))

    def clear_curve_overlays(self):
        self.remove_items(self.critical_items)
        self.remove_items(self.caustic_items)
        self.critical_items = []
        self.caustic_items = []

    def add_polyline(self, scene, x, y, color, width, zvalue):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)
        x = x[finite]
        y = y[finite]
        path = QPainterPath()
        if x.size > 0:
            path.moveTo(float(x[0]), float(y[0]))
            for xx, yy in zip(x[1:], y[1:]):
                path.lineTo(float(xx), float(yy))
        item = QGraphicsPathItem(path)
        pen = QPen(color, width)
        pen.setCosmetic(True)
        item.setPen(pen)
        item.setZValue(zvalue)
        scene.addItem(item)
        return item

    def refresh_prediction_from_anchor(self):
        if self.last_prediction_anchor is None:
            return

        kind, x, y = self.last_prediction_anchor
        if kind == 'source':
            self.predict_from_source(QPointF(x, y), remember=False)
        elif kind == 'image':
            self.predict_from_image(QPointF(x, y), remember=False)

    def predict_from_source(self, point, remember=True):
        if self.lens is None:
            self.status_label.setText('Draw at least one SIE or enable a Lenstool model before predicting images.')
            return
        beta1, beta2 = point.x(), point.y()
        xi, yi, mui = self.find_images(beta1, beta2)
        if remember:
            self.last_prediction_anchor = ('source', beta1, beta2)
        self.clear_prediction_markers(clear_table=True, clear_anchor=False)
        self.source_marker_items.append(self.add_marker(self.source_scene, beta1, beta2, QColor(255, 60, 60), radius=0.60, zvalue=60))
        for x, y, mu in zip(xi, yi, mui):
            radius = min(1.20, 0.45 + 0.06 * np.sqrt(abs(mu)))
            self.image_marker_items.append(self.add_marker(self.image_scene, x, y, QColor(255, 60, 60), radius=radius, zvalue=60))
        self.ensure_sie_handles_visible()
        self.status_label.setText(f'Source ({beta1:.3f}, {beta2:.3f}) produced {len(xi)} images.')

    def predict_from_image(self, point, remember=True):
        if self.lens is None:
            self.status_label.setText('Draw at least one SIE or enable a Lenstool model before predicting images.')
            return
        theta1, theta2 = point.x(), point.y()
        a1, a2 = self.lens.angle(theta1, theta2)
        beta1 = float(theta1 - np.atleast_1d(a1)[0])
        beta2 = float(theta2 - np.atleast_1d(a2)[0])
        xi, yi, mui = self.find_images(beta1, beta2)
        if remember:
            self.last_prediction_anchor = ('image', theta1, theta2)
        self.clear_prediction_markers(clear_table=True, clear_anchor=False)
        self.source_marker_items.append(self.add_marker(self.source_scene, beta1, beta2, QColor(255, 60, 60), radius=0.60, zvalue=60))
        self.image_marker_items.append(self.add_cross_marker(self.image_scene, theta1, theta2, QColor(255, 60, 60), size=0.95, zvalue=70))
        for x, y, mu in zip(xi, yi, mui):
            radius = min(1.20, 0.45 + 0.06 * np.sqrt(abs(mu)))
            self.image_marker_items.append(self.add_marker(self.image_scene, x, y, QColor(255, 60, 60), radius=radius, zvalue=60))
        self.ensure_sie_handles_visible()
        self.status_label.setText(
            f'Image ({theta1:.3f}, {theta2:.3f}) maps to source ({beta1:.3f}, {beta2:.3f}); {len(xi)} images.'
        )

    def find_images(self, beta1, beta2):
        ps = pointsrc(
            sizex=[self.extent[0], self.extent[1]],
            sizey=[self.extent[2], self.extent[3]],
            Npix=self.lens.nray1,
            gl=self.lens,
            use_lenstronomy=False,
            refine=True,
            zs=self.lens.zs,
            ys1=float(beta1),
            ys2=float(beta2),
            flux=1.0,
        )
        return np.atleast_1d(ps.xi1), np.atleast_1d(ps.xi2), np.atleast_1d(ps.mui)

    def add_marker(self, scene, x, y, color, radius=0.35, zvalue=60):
        item = QGraphicsEllipseItem(QRectF(x - radius, y - radius, 2.0 * radius, 2.0 * radius))
        pen = QPen(QColor(0, 0, 0), 1.4)
        pen.setCosmetic(True)
        item.setPen(pen)
        item.setBrush(QBrush(color))
        item.setZValue(zvalue)
        scene.addItem(item)
        return item

    def add_labeled_marker(self, scene, x, y, label, color, zvalue=70):
        marker = self.add_marker(scene, x, y, color, radius=0.55, zvalue=zvalue)
        text = QGraphicsSimpleTextItem(label)
        text.setBrush(QBrush(QColor(255, 255, 255)))
        offset = 0.25
        text.setPos(x + offset, y - offset)
        text.setScale(0.0006 * max(self.extent[1] - self.extent[0], self.extent[3] - self.extent[2]))
        text.setTransform(QTransform().scale(1.0, -1.0), True)
        text.setZValue(zvalue + 1)
        scene.addItem(text)
        return [marker, text]

    def add_cross_marker(self, scene, x, y, color, size=0.6, zvalue=70):
        path = QPainterPath()
        path.moveTo(x - size, y - size)
        path.lineTo(x + size, y + size)
        path.moveTo(x - size, y + size)
        path.lineTo(x + size, y - size)
        item = QGraphicsPathItem(path)
        pen = QPen(color, 2.0)
        pen.setCosmetic(True)
        item.setPen(pen)
        item.setZValue(zvalue)
        scene.addItem(item)
        return item

    def clear_prediction_markers(self, clear_table=False, clear_anchor=True):
        self.remove_items(self.source_marker_items)
        self.remove_items(self.image_marker_items)
        self.source_marker_items = []
        self.image_marker_items = []
        if clear_anchor:
            self.last_prediction_anchor = None

    def remove_items(self, items):
        for item in items:
            if item is not None and item.scene() is not None:
                item.scene().removeItem(item)

    def update_status_position(self, plane_name, point):
        self.status_label.setText(f'{plane_name}: x={point.x():.3f}, y={point.y():.3f}')


def parse_args(argv):
    parser = argparse.ArgumentParser(description='Interactive critical-line image finder.')
    parser.add_argument('--cluster', default='S1063', choices=list(CLUSTER_PATHS.keys()))
    parser.add_argument('--prefix', default=None, help='Override cluster file prefix, without _rgb.fits/.par suffix.')
    parser.add_argument('--rgb', default=None, help='Override RGB FITS path.')
    parser.add_argument('--fov', type=float, default=60.0, help='Fallback field of view in arcsec.')
    parser.add_argument('--nray', type=int, default=512, help='Analytic-component grid resolution.')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    app = QApplication(sys.argv)
    window = CLImageFinder(args)
    window.show()
    window.show_viewer_window()
    return app.exec()


if __name__ == '__main__':
    raise SystemExit(main())

import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget,
                             QLabel, QSlider, QComboBox, QCheckBox, QPushButton,QLineEdit,
                             QGridLayout, QTableWidget, QTableWidgetItem,QHeaderView,
                             QMessageBox)
from PyQt6.QtGui import QDoubleValidator, QFont
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import numpy as np
import pyLensLib.lenstool as lst
from pyLensLib.sersic_numba import sersic
from pyLensLib.pointsrc import pointsrc
from matplotlib.figure import Figure

from matplotlib.widgets import RectangleSelector
from matplotlib.backend_bases import MouseButton, MouseEvent, Event
from matplotlib.colors import LogNorm, Normalize
from scipy.ndimage import map_coordinates

import yaml
import os

import astropy.io.fits as fits
from PyQt6.QtCore import QLocale


def alpha_blend(base_rgb, overlay_rgba):
    #alpha = overlay_rgba[:, :, 3:4]  # shape (ny, nx, 1)
    #return base_rgb * (1 - alpha) + overlay_rgba[:, :, :3] * alpha
    result = base_rgb.copy()
    alpha = overlay_rgba[:, :, 3:4]  # shape: (H, W, 1)
    overlay_rgb = overlay_rgba[:, :, :3]

    # Optional: mask very faint overlays to reduce noise
    mask = alpha > 0.01

    # Add overlay only where it's visible
    result[mask.squeeze()] += overlay_rgb[mask.squeeze()] * alpha[mask.squeeze()]

    # Clip values to [0, 1]
    return np.clip(result, 0, 1)

def decimal_validator(bottom, top, decimals, parent):
    validator = QDoubleValidator(bottom, top, decimals, parent)
    locale = QLocale.c()
    locale.setNumberOptions(QLocale.NumberOption.RejectGroupSeparator)
    validator.setLocale(locale)
    validator.setNotation(QDoubleValidator.Notation.StandardNotation)
    return validator

class LensInteract(QMainWindow):
    def __init__(self):
        super().__init__()

        self.nodata = 0
        self.setWindowTitle('Lens Interact')
        screen = QApplication.primaryScreen() # reference to the primary screen
        if screen is not None:
            screen_rect = screen.geometry() # get the screen geometry
            self.setGeometry(screen_rect) # set the geometry of the window to the screen geometry
        else:
            self.resize(1200, 800)  # fallback default size if no screen is available
        self.main_widget = QWidget(self) # create a main widget
        self.setCentralWidget(self.main_widget) # set the main widget as the central widget
        self.grid_layout = QGridLayout(self.main_widget) # create a grid layout
        self.initUI()
        self.show()

    def initUI(self):
        self.label_font = QFont() # create a font object
        self.label_font.setPointSize(12) # set the font size

        # assuming a grid layout with 12 columns
        ncols = 12

        # set the stretch factor for each column
        for i in range(ncols):
            self.grid_layout.setColumnStretch(i, 1)

        # create the widgets to be added to the layout
        # effective radius label and  QlineEdit
        self.re_label = QLabel('Effective radius:', self)
        self.re_label.setFont(self.label_font)
        self.grid_layout.addWidget(self.re_label,0,0,1,1)

        self.re_input = QLineEdit(self)
        self.re_input.setText("0.5") # default value for the effective radius
        self.re_input.setValidator(decimal_validator(0.0, 1.0e12, 8, self))
        self.grid_layout.addWidget(self.re_input,0,1,1,1)

        # sersic index
        self.n_label = QLabel('Sersic index:', self)
        self.n_label.setFont(self.label_font)
        self.grid_layout.addWidget(self.n_label,0,2,1,1)
        self.n_input = QLineEdit(self)
        self.n_input.setText("1.0") # default value for the sersic index
        self.n_input.setValidator(decimal_validator(0.0, 1.0e12, 8, self))
        self.grid_layout.addWidget(self.n_input,0,3,1,1)

        # source redshift label and QLineEdit
        self.zs_label = QLabel('Source redshift:', self)
        self.zs_label.setFont(self.label_font)
        self.grid_layout.addWidget(self.zs_label,1,0,1,1)

        self.zs_input = QLineEdit(self)
        self.zs_input.setText("1.0")
        self.zs_input.setValidator(decimal_validator(0.0, 1.0e12, 8, self))
        self.grid_layout.addWidget(self.zs_input,1,1,1,1)

        # update button (activate chages on re and zs)
        self.update_button = QPushButton('Update', self)
        self.update_button.clicked.connect(self.update_plot)
        self.update_button.clicked.connect(self.update_nodata)
        self.grid_layout.addWidget(self.update_button,2,0,1,2)

        # cluster label and QComboBox (drop-down menu)
        self.cluster_label = QLabel('Select Cluster:', self)
        self.cluster_label.setFont(self.label_font)
        self.grid_layout.addWidget(self.cluster_label,0,6,1,1)

        self.cluster_combo = QComboBox(self)
        self.cluster_combo.addItems(['S1063', 'M0416_B22', 'M0416_canucs', 'M1206pl', 'PSZ1G311_200', 'A370', 'A2744', 'M0717', 'M1149', 'M0329', 'M1931', 'M2129', 'R2129','PLCK-G287','elgordo','A2390'])
        self.cluster_combo.currentTextChanged.connect(self.update_cluster)
        self.grid_layout.addWidget(self.cluster_combo,0,7,1,1)

        # use Sersic checkbox#
        self.sersic_checkbox = QCheckBox('Use Sersic', self)
        self.sersic_checkbox.setFont(self.label_font)
        self.sersic_checkbox.stateChanged.connect(self.update_plot)
        self.sersic_checkbox.stateChanged.connect(self.update_nodata)
        self.grid_layout.addWidget(self.sersic_checkbox,2,2,1,1)

        # use TD checkbox#
        self.showtd_checkbox = QCheckBox('Show Time Delay', self)
        self.showtd_checkbox.setFont(self.label_font)
        self.showtd_checkbox.stateChanged.connect(self.update_plot)
        self.showtd_checkbox.stateChanged.connect(self.update_nodata)
        self.grid_layout.addWidget(self.showtd_checkbox,2,3,1,1)

        # use Convergece checkbox#
        self.showconvergence_checkbox = QCheckBox('Show Convergence', self)
        self.showconvergence_checkbox.setFont(self.label_font)
        self.showconvergence_checkbox.stateChanged.connect(self.update_plot)
        self.showconvergence_checkbox.stateChanged.connect(self.update_nodata)
        self.grid_layout.addWidget(self.showconvergence_checkbox,2,4,1,1)

        # use showrgb checkbox#
        self.showrgb_checkbox = QCheckBox('Show RGB image', self)
        self.showrgb_checkbox.setFont(self.label_font)
        self.showrgb_checkbox.stateChanged.connect(self.update_plot)
        self.showrgb_checkbox.stateChanged.connect(self.update_nodata)
        self.grid_layout.addWidget(self.showrgb_checkbox,2,5,1,1)

        # if Sersic is not checked, another checkbox to enable refine of the image positions
        self.refine_checkbox = QCheckBox('Refine Image Positions', self)
        self.refine_checkbox.setFont(self.label_font)
        self.refine_checkbox.stateChanged.connect(self.update_plot)
        self.refine_checkbox.stateChanged.connect(self.update_nodata)
        self.grid_layout.addWidget(self.refine_checkbox,2,6,1,1)

        # optional refinement of point-source images to time-delay stationary points
        self.refine_td_checkbox = QCheckBox('Refine to TD Surface', self)
        self.refine_td_checkbox.setFont(self.label_font)
        self.refine_td_checkbox.stateChanged.connect(self.update_plot)
        self.refine_td_checkbox.stateChanged.connect(self.update_nodata)
        self.grid_layout.addWidget(self.refine_td_checkbox,2,7,1,1)

        self.predict_checkbox = QCheckBox('Predict', self)
        self.predict_checkbox.setFont(self.label_font)
        self.predict_checkbox.stateChanged.connect(self.update_predict_mode)
        self.predict_checkbox.stateChanged.connect(self.update_nodata)
        self.grid_layout.addWidget(self.predict_checkbox, 4, 0, 1, 1)

        self.zmin_label = QLabel('zmin:', self)
        self.zmin_label.setFont(self.label_font)
        self.grid_layout.addWidget(self.zmin_label, 4, 1, 1, 1)

        self.zmin_input = QLineEdit(self)
        self.zmin_input.setText("1.0")
        self.zmin_input.setValidator(decimal_validator(0.0, 1.0e12, 8, self))
        self.grid_layout.addWidget(self.zmin_input, 4, 2, 1, 1)

        self.zmax_label = QLabel('zmax:', self)
        self.zmax_label.setFont(self.label_font)
        self.grid_layout.addWidget(self.zmax_label, 4, 3, 1, 1)

        self.zmax_input = QLineEdit(self)
        self.zmax_input.setText("6.0")
        self.zmax_input.setValidator(decimal_validator(0.0, 1.0e12, 8, self))
        self.grid_layout.addWidget(self.zmax_input, 4, 4, 1, 1)

        self.update_predict_button = QPushButton('Update Predict', self)
        self.update_predict_button.clicked.connect(self.update_prediction_plot)
        self.update_predict_button.clicked.connect(self.update_nodata)
        self.grid_layout.addWidget(self.update_predict_button, 4, 5, 1, 2)

        # table to display info about the clicked source
        # self.source_table = QTableWidget(self)
        # self.layout.addWidget(self.source_table, 0, 7, 1, 2)
        # self.source_table.setRowCount(1)
        # self.source_table.setColumnCount(2)
        # self.source_table.setHorizontalHeaderLabels(['Source Position X', 'Source Position Y'])
        # horizontal_header = self.source_table.horizontalHeader()
        # vertical_header = self.source_table.verticalHeader()
        # horizontal_header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        # vertical_header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        # table to display info about multiple images
        self.image_table = QTableWidget(self)
        self.image_table.setRowCount(10)  # Adjust as needed
        self.image_table.setColumnCount(4)
        # Set the headers
        self.image_table.setHorizontalHeaderLabels(
            ['Image Position X', 'Image Position Y', 'Magnification', 'Time Delay - Earliest [d]']
        )
        # Get the horizontal and vertical headers
        horizontal_header = self.image_table.horizontalHeader()
        vertical_header = self.image_table.verticalHeader()

        # Set the resize mode to stretch so that all sections will be resized equally
        if horizontal_header is not None:
            horizontal_header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        if vertical_header is not None:
            vertical_header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        
        self.grid_layout.addWidget(self.image_table, 0, 9, 3, 3)

        # initialize source position
        self.beta1 = 0.0
        self.beta2 = 0.0
        self.current_td_surface = None
        self.last_prediction_click = None
        self._predict_mode_initialized = False
        self.prediction_artists = []
        self.prediction_annotation = None
        self.source_position = QLabel('Source Position: {:.2f} {:.2f}'.format(self.beta1, self.beta2), self)
        self.source_position.setFont(self.label_font)
        self.grid_layout.addWidget(self.source_position, 1, 2, 1, 1)

        # optional manual coordinate input. The default keeps the original
        # mouse-click behavior, while the other modes override clicked points.
        self.coord_mode_label = QLabel('Coordinate input:', self)
        self.coord_mode_label.setFont(self.label_font)
        self.grid_layout.addWidget(self.coord_mode_label, 3, 0, 1, 1)

        self.coord_mode_combo = QComboBox(self)
        self.coord_mode_combo.addItems(['Use mouse clicks', 'Use source coords', 'Use image coords'])
        self.grid_layout.addWidget(self.coord_mode_combo, 3, 1, 1, 2)

        self.coord_x_label = QLabel('x:', self)
        self.coord_x_label.setFont(self.label_font)
        self.grid_layout.addWidget(self.coord_x_label, 3, 3, 1, 1)

        self.coord_x_input = QLineEdit(self)
        self.coord_x_input.setText("0.0")
        self.coord_x_input.setValidator(decimal_validator(-1.0e12, 1.0e12, 8, self))
        self.grid_layout.addWidget(self.coord_x_input, 3, 4, 1, 1)

        self.coord_y_label = QLabel('y:', self)
        self.coord_y_label.setFont(self.label_font)
        self.grid_layout.addWidget(self.coord_y_label, 3, 5, 1, 1)

        self.coord_y_input = QLineEdit(self)
        self.coord_y_input.setText("0.0")
        self.coord_y_input.setValidator(decimal_validator(-1.0e12, 1.0e12, 8, self))
        self.grid_layout.addWidget(self.coord_y_input, 3, 6, 1, 1)

        self.apply_coord_button = QPushButton('Apply Coordinates', self)
        self.apply_coord_button.clicked.connect(self.apply_coordinate_inputs)
        self.apply_coord_button.clicked.connect(self.update_nodata)
        self.grid_layout.addWidget(self.apply_coord_button, 3, 7, 1, 2)
        self.coord_x_input.returnPressed.connect(self.apply_coordinate_inputs)
        self.coord_y_input.returnPressed.connect(self.apply_coordinate_inputs)

        size = self.size()
        fig_width = size.width() / 80
        fig_height = size.height() / 80

        self.fig = Figure(figsize=(fig_width,fig_height))
        self.canvas = FigureCanvas(self.fig)
        self.grid_layout.addWidget(self.canvas, 5, 0, 1, ncols)
        self.canvas.mpl_connect('button_press_event', self.mouse_event)
        self.canvas.mpl_connect('motion_notify_event', self.prediction_hover_event)

        self.re = self._parse_float_input(self.re_input, 'effective radius')
        self.n_index = self._parse_float_input(self.n_input, 'Sersic index')
        self.zs = self._parse_float_input(self.zs_input, 'source redshift')
        # initialize some logical variables (using default values
        self.sersic = self.sersic_checkbox.isChecked()
        self.showtd = self.showtd_checkbox.isChecked()
        self.showconvergence = self.showconvergence_checkbox.isChecked()
        self.showrgb = self.showrgb_checkbox.isChecked()
        self.refine = self.refine_checkbox.isChecked()
        self.refine_td = self.refine_td_checkbox.isChecked()
        self.predict = self.predict_checkbox.isChecked()
        self.cluster = self.cluster_combo.currentText()

        # set up lens and axes
        self.lens_init()
        self.update_plot()
        self.update_predict_mode()
        self.refresh_zoom_selectors()

        # add a reset button to reset the zoom
        self.reset_button = QPushButton('Reset Zoom', self)
        self.reset_button.clicked.connect(self.reset_zoom)
        self.grid_layout.addWidget(self.reset_button,6,0,1,ncols)

    def reset_zoom(self):
        self.ax1.set_xlim(*self.extent1)
        self.ax1.set_ylim(*self.extent2)
        self.ax2.set_xlim(*self.extent1)
        self.ax2.set_ylim(*self.extent2)
        self.beta1_lim = self.extent1
        self.beta2_lim = self.extent2
        self.theta1_lim = self.extent1
        self.theta2_lim = self.extent2
        self.canvas.draw_idle()

    def refresh_zoom_selectors(self):
        for selector_name in ('RS1', 'RS2'):
            selector = getattr(self, selector_name, None)
            if selector is not None:
                selector.disconnect_events()

        # Rectangle selectors must be rebound whenever fig.clear() creates new axes.
        self.RS1 = ShiftRectangleSelector(self.ax1, self.onselect1,
                                          interactive=True, useblit=True, spancoords='pixels')
        self.RS2 = ShiftRectangleSelector(self.ax2, self.onselect2,
                                          interactive=True, useblit=True, spancoords='pixels')

    def update_nodata(self):
        self.nodata += 1

    def update_predict_mode(self):
        predict_on = self.predict_checkbox.isChecked()
        if predict_on:
            self.coord_mode_combo.setCurrentText('Use mouse clicks')
        self.coord_mode_combo.setEnabled(not predict_on)
        self.coord_x_input.setEnabled(not predict_on)
        self.coord_y_input.setEnabled(not predict_on)
        self.apply_coord_button.setEnabled(not predict_on)
        self.zmin_label.setEnabled(predict_on)
        self.zmin_input.setEnabled(predict_on)
        self.zmax_label.setEnabled(predict_on)
        self.zmax_input.setEnabled(predict_on)
        self.update_predict_button.setEnabled(predict_on)
        if getattr(self, '_predict_mode_initialized', False):
            self.update_plot()
        self._predict_mode_initialized = True

    def _parse_float_input(self, line_edit, field_name):
        text = line_edit.text().strip()
        try:
            value = float(text)
        except ValueError:
            self._show_invalid_numeric_input(line_edit, field_name, text, positive=True)
            return None

        if not np.isfinite(value) or value <= 0.0:
            self._show_invalid_numeric_input(line_edit, field_name, text, positive=True)
            return None

        line_edit.setStyleSheet('')
        return value

    def _parse_coordinate_input(self, line_edit, field_name):
        text = line_edit.text().strip()
        try:
            value = float(text)
        except ValueError:
            self._show_invalid_numeric_input(line_edit, field_name, text, positive=False)
            return None

        if not np.isfinite(value):
            self._show_invalid_numeric_input(line_edit, field_name, text, positive=False)
            return None

        line_edit.setStyleSheet('')
        return value

    def _show_invalid_numeric_input(self, line_edit, field_name, text, positive=True):
        line_edit.setStyleSheet('QLineEdit { border: 2px solid #c43; }')
        line_edit.setFocus()
        line_edit.selectAll()
        number_kind = 'a positive number' if positive else 'a finite number'
        QMessageBox.warning(
            self,
            'Invalid numeric input',
            f'Please enter {number_kind} for {field_name}. Received: "{text}"',
        )

    def _as_float_scalar(self, value):
        return float(np.atleast_1d(value)[0])

    def _using_manual_coordinates(self):
        return self.coord_mode_combo.currentText() != 'Use mouse clicks'

    def _apply_coordinate_override(self):
        mode = self.coord_mode_combo.currentText()
        if mode == 'Use mouse clicks':
            return True

        x = self._parse_coordinate_input(self.coord_x_input, 'coordinate x')
        if x is None:
            return False
        y = self._parse_coordinate_input(self.coord_y_input, 'coordinate y')
        if y is None:
            return False

        if mode == 'Use source coords':
            self.beta1 = x
            self.beta2 = y
        elif mode == 'Use image coords':
            self.set_source_from_image_position(x, y)

        return True

    def apply_coordinate_inputs(self):
        self.update_plot()

    def image_plane_deflection_for_source_update(self, theta1, theta2):
        if (self.showtd or self.refine_td) and getattr(self.lens, 'computed_potential', False):
            return self.lens.getAngleFromPotential(theta1, theta2)
        return self.lens.getAngle(theta1, theta2)

    def set_source_from_image_position(self, theta1, theta2):
        a1, a2 = self.image_plane_deflection_for_source_update(theta1, theta2)
        self.beta1 = self._as_float_scalar(theta1 - a1)
        self.beta2 = self._as_float_scalar(theta2 - a2)

    def prediction_redshifts(self):
        zmin = self._parse_float_input(self.zmin_input, 'prediction zmin')
        if zmin is None:
            return None
        zmax = self._parse_float_input(self.zmax_input, 'prediction zmax')
        if zmax is None:
            return None

        if zmax <= zmin:
            QMessageBox.warning(
                self,
                'Invalid redshift range',
                'Prediction requires zmax to be greater than zmin.',
            )
            return None

        if zmin <= self.lens.zl:
            QMessageBox.warning(
                self,
                'Invalid redshift range',
                f'Prediction zmin must be greater than the lens redshift z_l={self.lens.zl:.3f}.',
            )
            return None

        return np.linspace(zmin, zmax, 15)

    def sync_options_from_controls(self):
        zs = self._parse_float_input(self.zs_input, 'source redshift')
        if zs is None:
            return False

        self.zs = zs
        self.sersic = self.sersic_checkbox.isChecked()
        self.showtd = self.showtd_checkbox.isChecked()
        self.showconvergence = self.showconvergence_checkbox.isChecked()
        self.showrgb = self.showrgb_checkbox.isChecked()
        self.refine = self.refine_checkbox.isChecked()
        self.refine_td = self.refine_td_checkbox.isChecked()
        self.predict = self.predict_checkbox.isChecked()
        self.lens.change_redshift(self.zs)
        return True

    def prepare_prediction_plot(self):
        self.fig.clear()
        self.ax1 = self.fig.add_subplot(121)
        self.ax2 = self.fig.add_subplot(122)
        self.refresh_zoom_selectors()
        self.prediction_artists = []
        self.prediction_annotation = None
        self.image_table.clearContents()
        self.image_table.setRowCount(0)
        self.current_td_surface = None

        if self.showconvergence:
            self.plot_convergence()
        if self.showrgb:
            self.plot_rgb(alpha=0.55)
        self.plot_critlines_caustics()

    def add_prediction_hover_artist(self, artist, redshifts, label):
        artist.set_picker(True)
        if hasattr(artist, 'set_pickradius'):
            artist.set_pickradius(6)
        self.prediction_artists.append(
            {
                'artist': artist,
                'redshifts': np.asarray(redshifts, dtype=float),
                'label': label,
            }
        )

    def ensure_prediction_annotation(self, axes):
        if self.prediction_annotation is not None and self.prediction_annotation.axes == axes:
            return self.prediction_annotation
        if self.prediction_annotation is not None:
            self.prediction_annotation.set_visible(False)

        self.prediction_annotation = axes.annotate(
            '',
            xy=(0, 0),
            xytext=(12, 12),
            textcoords='offset points',
            bbox={'boxstyle': 'round', 'fc': 'white', 'ec': 'black', 'alpha': 0.9},
            arrowprops={'arrowstyle': '->', 'color': 'black'},
            zorder=200,
        )
        self.prediction_annotation.set_visible(False)
        return self.prediction_annotation

    def hide_prediction_annotation(self):
        if self.prediction_annotation is not None and self.prediction_annotation.get_visible():
            self.prediction_annotation.set_visible(False)
            self.canvas.draw_idle()

    def prediction_hover_event(self, event):
        if not self.predict_checkbox.isChecked() or event.inaxes is None:
            self.hide_prediction_annotation()
            return

        for item in self.prediction_artists:
            artist = item['artist']
            if artist.axes != event.inaxes:
                continue
            contains, info = artist.contains(event)
            if not contains:
                continue

            ind = info.get('ind', [])
            if len(ind) == 0:
                continue
            point_index = int(ind[0])
            offsets = artist.get_offsets()
            if point_index >= len(offsets) or point_index >= len(item['redshifts']):
                continue

            x, y = offsets[point_index]
            z = item['redshifts'][point_index]
            annotation = self.ensure_prediction_annotation(event.inaxes)
            annotation.xy = (x, y)
            annotation.set_text(f'{item["label"]}\nz = {z:.3f}')
            annotation.set_visible(True)
            self.canvas.draw_idle()
            return

        self.hide_prediction_annotation()

    def predicted_images_for_source(self, beta1, beta2, z_values):
        image_x, image_y, image_z = [], [], []
        restore_z = self.lens.zs

        try:
            for z in z_values:
                self.lens.change_redshift(float(z))
                ps = pointsrc(
                    sizex=self.extent1,
                    sizey=self.extent2,
                    Npix=self.lens.nray1,
                    gl=self.lens,
                    use_lenstronomy=False,
                    refine=self.refine,
                    refine_to_td=False,
                    zs=self.lens.zs,
                    ys1=beta1,
                    ys2=beta2,
                    flux=1.0,
                )
                xi = np.atleast_1d(ps.xi1)
                yi = np.atleast_1d(ps.xi2)
                finite = np.isfinite(xi) & np.isfinite(yi)
                image_x.extend(xi[finite])
                image_y.extend(yi[finite])
                image_z.extend(np.full(np.count_nonzero(finite), z))
        finally:
            self.lens.change_redshift(restore_z)

        return np.asarray(image_x), np.asarray(image_y), np.asarray(image_z)

    def predicted_positions_for_image(self, theta1, theta2, z_values):
        source_x, source_y, source_z = [], [], []
        image_x, image_y, image_z = [], [], []
        restore_z = self.lens.zs

        try:
            for z in z_values:
                self.lens.change_redshift(float(z))
                a1, a2 = self.lens.getAngle(theta1, theta2)
                beta1 = self._as_float_scalar(theta1 - a1)
                beta2 = self._as_float_scalar(theta2 - a2)
                source_x.append(beta1)
                source_y.append(beta2)
                source_z.append(z)

                ps = pointsrc(
                    sizex=self.extent1,
                    sizey=self.extent2,
                    Npix=self.lens.nray1,
                    gl=self.lens,
                    use_lenstronomy=False,
                    refine=self.refine,
                    refine_to_td=False,
                    zs=self.lens.zs,
                    ys1=beta1,
                    ys2=beta2,
                    flux=1.0,
                )
                xi = np.atleast_1d(ps.xi1)
                yi = np.atleast_1d(ps.xi2)
                finite = np.isfinite(xi) & np.isfinite(yi)
                image_x.extend(xi[finite])
                image_y.extend(yi[finite])
                image_z.extend(np.full(np.count_nonzero(finite), z))
        finally:
            self.lens.change_redshift(restore_z)

        return (
            np.asarray(source_x), np.asarray(source_y), np.asarray(source_z),
            np.asarray(image_x), np.asarray(image_y), np.asarray(image_z)
        )

    def add_prediction_colorbar(self, z_values, cmap):
        norm = Normalize(vmin=float(np.min(z_values)), vmax=float(np.max(z_values)))
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        self.fig.colorbar(sm, ax=[self.ax1, self.ax2], label='Source redshift')
        return norm

    def plot_source_prediction(self, beta1, beta2, z_values):
        self.prepare_prediction_plot()
        cmap = plt.colormaps.get_cmap('viridis')
        norm = self.add_prediction_colorbar(z_values, cmap)
        image_x, image_y, image_z = self.predicted_images_for_source(beta1, beta2, z_values)

        self.ax1.scatter(beta1, beta2, marker='*', s=130, color='red', edgecolors='black', zorder=120)
        if image_x.size > 0:
            artist = self.ax2.scatter(
                image_x, image_y, c=image_z, cmap=cmap, norm=norm,
                s=36, alpha=0.92, edgecolors='black', linewidths=0.35, zorder=120,
            )
            self.add_prediction_hover_artist(artist, image_z, 'Predicted image')

        self.beta1 = beta1
        self.beta2 = beta2
        self.update_axes()
        self.source_position.setText(
            'Prediction from source: {:.2f} {:.2f}'.format(beta1, beta2)
        )
        self.canvas.draw_idle()

    def plot_image_prediction(self, theta1, theta2, z_values):
        self.prepare_prediction_plot()
        cmap = plt.colormaps.get_cmap('plasma')
        norm = self.add_prediction_colorbar(z_values, cmap)
        sx, sy, sz, ix, iy, iz = self.predicted_positions_for_image(theta1, theta2, z_values)

        if sx.size > 0:
            artist = self.ax1.scatter(
                sx, sy, c=sz, cmap=cmap, norm=norm,
                s=36, alpha=0.95, edgecolors='black', linewidths=0.35, zorder=120,
            )
            self.add_prediction_hover_artist(artist, sz, 'Predicted source')
        if ix.size > 0:
            artist = self.ax2.scatter(
                ix, iy, c=iz, cmap=cmap, norm=norm,
                s=34, alpha=0.88, edgecolors='black', linewidths=0.35, zorder=120,
            )
            self.add_prediction_hover_artist(artist, iz, 'Predicted image')
        self.ax2.scatter(theta1, theta2, marker='x', s=90, color='black', linewidths=2.0, zorder=130)

        self.update_axes()
        self.source_position.setText(
            'Prediction from image: {:.2f} {:.2f}'.format(theta1, theta2)
        )
        self.canvas.draw_idle()

    def handle_prediction_click(self, event):
        if event.inaxes not in [self.ax1, self.ax2] or event.xdata is None or event.ydata is None:
            return
        if not self.sync_options_from_controls():
            return

        z_values = self.prediction_redshifts()
        if z_values is None:
            return

        x = self._as_float_scalar(event.xdata)
        y = self._as_float_scalar(event.ydata)
        self.last_prediction_click = ('source' if event.inaxes == self.ax1 else 'image', x, y)
        self.draw_prediction_from_last_click(z_values)

    def update_prediction_plot(self):
        if not self.predict_checkbox.isChecked():
            return
        if self.last_prediction_click is None:
            QMessageBox.information(
                self,
                'No prediction anchor',
                'Click a source or image position first, then update zmin/zmax.',
            )
            return
        if not self.sync_options_from_controls():
            return
        z_values = self.prediction_redshifts()
        if z_values is None:
            return
        self.draw_prediction_from_last_click(z_values)

    def draw_prediction_from_last_click(self, z_values):
        if self.last_prediction_click is None:
            return

        plane, x, y = self.last_prediction_click
        if plane == 'source':
            self.plot_source_prediction(x, y, z_values)
        else:
            self.plot_image_prediction(x, y, z_values)

    def lens_init(self):
        # path to the deflection angle maps: RECOMMENDATION: should be stored in a configuration file
        self.path_to_angles = {
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
            'A2390': '/Users/maxmen3/stiva/abriola_models/Deflection_Maps_A2390_Gravity/'
        }

        self.file_prefix = self.path_to_angles[self.cluster] + self.cluster
        # get the lens redshift from the lenstool parameter file and create the deflector
        # RECOMMENDATION: we should  get these info in a different way if the GUI has to be used with
        # lens models (e.g., from gravity.jl)
        zl = float(lst.getLensRedshift(self.file_prefix + '.par'))
        self.lens = lst.create_deflector(
            parfile=self.file_prefix + '.par',
            filex=self.file_prefix + '_angx.fits',
            filey=self.file_prefix + '_angy.fits',
            filepot=self.file_prefix + '_pot.fits',
            usePotential=False,
            zl=zl, zs=self.zs, zsnorm=1.0, resc_fact=1.0, compute_potential=False)

        # if the file exists, load rgb.fits image of the cluster
        if os.path.exists(self.file_prefix + '_rgb.fits'):
            with fits.open(self.file_prefix + '_rgb.fits') as hdul:
                rgb_fits_data = hdul[0].data  # type: ignore
            rgb_image = np.transpose(rgb_fits_data, (1, 2, 0))
            #rgb_image = np.flip(rgb_image, axis=0)
            rgb_image = rgb_image / np.max(rgb_image)
            self.rgb_image = rgb_image
        else:
            self.rgb_image = self.lens.ka

        # get the field of view from the lenstool parameter file
        # RECOMMENDATION: see above
        lims = lst.getFoV(self.file_prefix + '.par')
        fov = lims[1] - lims[0]
        print ('FOV:', fov, lims)
        # set the grid for the deflector

        # set limits in the source and lens planes. These limits will be adjusted
        # when the user zooms in the plots
        self.beta1_lim = [-fov/2.,fov/2.] #[lims[0], lims[1]]
        self.beta2_lim = [-fov/2.,fov/2.] #[lims[2], lims[3]]
        self.theta1_lim = self.beta1_lim
        self.theta2_lim = self.beta2_lim

        # set the extent of the plots. These values will be used to reset the zoom
        # and to freeze the size of displayed images in imshow calls

        #self.extent = [self.beta1_lim[0], self.beta1_lim[1], self.beta2_lim[0], self.beta2_lim[1]]
        self.extent = (-fov/2., fov/2., -fov/2., fov/2.)
        self.extent1 =  self.theta1_lim.copy()
        self.extent2 =  self.theta2_lim.copy()


    def update_cluster(self, cluster):
        """
        Update the cluster and the lens object
        :param cluster:  cluster name
        :return: None
        """
        self.cluster = cluster
        self.lens_init()
        self.update_plot()

    def update_plot(self):
        """
        Update the plot when re or zs are changed
        :return:  None
        """
        re = self._parse_float_input(self.re_input, 'effective radius')
        if re is None:
            return
        n_index = self._parse_float_input(self.n_input, 'Sersic index')
        if n_index is None:
            return
        zs = self._parse_float_input(self.zs_input, 'source redshift')
        if zs is None:
            return

        self.re = re
        self.n_index = n_index
        self.zs = zs
        self.sersic = self.sersic_checkbox.isChecked()
        self.showtd = self.showtd_checkbox.isChecked()
        self.showconvergence = self.showconvergence_checkbox.isChecked()
        self.showrgb = self.showrgb_checkbox.isChecked()
        self.refine = self.refine_checkbox.isChecked()
        self.refine_td = self.refine_td_checkbox.isChecked()
        self.predict = self.predict_checkbox.isChecked()
        self.lens.change_redshift(self.zs)
        if not self._apply_coordinate_override():
            return

        self.fig.clear()
        self.ax1 = self.fig.add_subplot(121)
        self.ax2 = self.fig.add_subplot(122)
        self.refresh_zoom_selectors()

        self.plot_lens()
        self.update_axes()
        self.canvas.draw_idle()

        #self.initUI()
        self.update_table()
        self.update_source_table()

    def update_source_table(self):
        # self.source_table.setItem(0, 0, QTableWidgetItem(str(self.beta1)))
        # self.source_table.setItem(0, 1, QTableWidgetItem(str(self.beta2)))
        # self.source_table.setRowHeight(1, 10)
        self.source_position.setText('Source Position: {:.2f} {:.2f}'.format(self.beta1, self.beta2))

    def update_table(self):
        # Clear the table
        self.image_table.clearContents()
        # Get the image positions and magnifications
        image_positions_x = np.atleast_1d(self.thetai_1)
        image_positions_y = np.atleast_1d(self.thetai_2)
        magnifications = np.atleast_1d(self.mui)
        relative_delays = self.image_time_delays(image_positions_x, image_positions_y)

        if relative_delays is not None:
            sort_values = np.where(np.isfinite(relative_delays), relative_delays, np.inf)
            order = np.argsort(sort_values, kind='stable')
            image_positions_x = image_positions_x[order]
            image_positions_y = image_positions_y[order]
            magnifications = magnifications[order]
            relative_delays = relative_delays[order]

        self.image_table.setRowCount(len(image_positions_x))

        # Update the table
        for i in range(len(image_positions_x)):
            self.image_table.setItem(i, 0, QTableWidgetItem("{:.2f}".format(image_positions_x[i])))
            self.image_table.setItem(i, 1, QTableWidgetItem("{:.2f}".format(image_positions_y[i])))
            self.image_table.setItem(i, 2, QTableWidgetItem("{:.2f}".format(magnifications[i])))
            if relative_delays is not None and np.isfinite(relative_delays[i]):
                delay_text = "{:.3f}".format(relative_delays[i])
            else:
                delay_text = "--"
            self.image_table.setItem(i, 3, QTableWidgetItem(delay_text))

    def image_time_delays(self, image_positions_x, image_positions_y):
        if not getattr(self.lens, 'computed_potential', False):
            return None

        image_positions_x = np.asarray(image_positions_x, dtype=float)
        image_positions_y = np.asarray(image_positions_y, dtype=float)
        if image_positions_x.size == 0:
            return np.array([], dtype=float)

        td_surface = self.current_td_surface
        if td_surface is None:
            td_surface = self.lens.t_delay_surf(beta=(self.beta1, self.beta2))
            self.current_td_surface = td_surface

        xpix = (image_positions_x - self.lens.thetax[0]) / self.lens.pixel_scale
        ypix = (image_positions_y - self.lens.thetay[0]) / self.lens.pixel_scale
        delays = map_coordinates(td_surface, [ypix, xpix], order=1, mode='nearest')
        return delays - np.nanmin(delays)

    def update_axes(self, axes='all'):
        if axes == 'all':
            self.ax1.set_xlabel(r'$\beta_1$', fontsize=15)
            self.ax1.set_ylabel(r'$\beta_2$', fontsize=15)
            self.ax2.set_xlabel(r'$\theta_1$', fontsize=15)
            self.ax2.set_ylabel(r'$\theta_2$', fontsize=15)
            self.ax1.set_xlim(tuple(self.beta1_lim))
            self.ax1.set_ylim(tuple(self.beta2_lim))
            self.ax2.set_xlim(tuple(self.theta1_lim))
            self.ax2.set_ylim(tuple(self.theta2_lim))

            self.ax1.set_aspect('equal')
            self.ax2.set_aspect('equal')
            self.ax1.set_xlabel(r'$\beta_1$', fontsize=15)
            self.ax1.set_ylabel(r'$\beta_2$', fontsize=15)
            self.ax2.set_xlabel(r'$\theta_1$', fontsize=15)
            self.ax2.set_ylabel(r'$\theta_2$', fontsize=15)
            self.ax1.set_xlim(tuple(self.beta1_lim))
            self.ax1.set_ylim(tuple(self.beta2_lim))
            self.ax2.set_xlim(tuple(self.theta1_lim))
            self.ax2.set_ylim(tuple(self.theta2_lim))

            self.ax1.set_aspect('equal')
            self.ax2.set_aspect('equal')
        elif axes == 'source':
            self.ax1.set_xlabel(r'$\beta_1$', fontsize=15)
            self.ax1.set_ylabel(r'$\beta_2$', fontsize=15)
            self.ax1.set_xlim(tuple(self.beta1_lim))
            self.ax1.set_ylim(tuple(self.beta2_lim))
            self.ax1.set_aspect('equal')
        elif axes == 'lens':
            self.ax2.set_xlabel(r'$\theta_1$', fontsize=15)
            self.ax2.set_ylabel(r'$\theta_2$', fontsize=15)
            self.ax2.set_xlim(tuple(self.theta1_lim))
            self.ax2.set_ylim(tuple(self.theta2_lim))
            self.ax2.set_aspect('equal')

    def plot_td(self, td=None):
        if td is None:
            td = self.lens.t_delay_surf(beta=(self.beta1, self.beta2))
        tds = np.sqrt((td - td.min()) / (td.max() - td.min()))

        #ax.contour(np.sqrt((tds - tds.min()) / (tds.max() - tds.min())), levels=np.logspace(-2, 0, 30),
         #          extent=[-fov / 2., fov / 2., -fov / 2, fov / 2.], colors='black', alpha=0.2)

        #td_min = td.min()
        #td -= td_min
        #max_td = 1.5 * ((-self.lens.pot).max() * self.lens.conv_fact_time.value - td_min)
        levels = np.logspace(-3, 0, 30)#np.linspace(0, max_td, 30) if max_td > 0 else 30

        if self.showrgb or self.showconvergence:
            self.ax2.contourf(tds, levels=levels, cmap=plt.colormaps.get_cmap('coolwarm'), extent=self.extent,zorder=18, alpha=0.7)
        else:
            self.ax2.contourf(tds, levels=levels, cmap=plt.colormaps.get_cmap('coolwarm'), extent=self.extent, zorder=18)
        self.ax2.contour(tds, levels=levels, extent=self.extent, colors='white', zorder=18,alpha=0.7)
    
    def plot_convergence(self):
        self.ax2.imshow(self.lens.ka, origin='lower', cmap='afmhot_r', alpha=0.7, zorder=2, extent=self.extent,
                        norm=LogNorm(vmax=2.0))

    def plot_rgb(self, alpha=1.0):
        self.ax2.imshow(self.rgb_image, origin='lower', zorder=1, extent=self.extent, alpha=alpha)

    def plot_critlines_caustics(self):
        zorder_cau =20
        tancau = self.lens.getCaustics(self.lens.tancl())
        radcau = self.lens.getCaustics(self.lens.radcl())

        for cl in self.lens.tancl():
            thetac1, thetac2 = self.lens.getCritPoints(cl)
            if self.sersic or self.showconvergence or self.showrgb:
                self.ax2.plot(thetac1, thetac2, '-', color='lightgray', zorder=zorder_cau)
            else:
                self.ax2.plot(thetac1, thetac2, '-', color='black', zorder=zorder_cau)

        for cl in tancau:
            betac1, betac2 = self.lens.getCausticPoints(cl)
            if self.sersic:
                self.ax1.plot(betac1, betac2, '-', color='lightgray', zorder=zorder_cau)
            else:
                self.ax1.plot(betac1, betac2, '-', color='black', zorder=zorder_cau)

        for cl in self.lens.radcl():
            thetac1, thetac2 = self.lens.getCritPoints(cl)
            if self.sersic or self.showconvergence or self.showrgb:
                self.ax2.plot(thetac1, thetac2, '-', color='lightgray', zorder=zorder_cau)
            else:
                self.ax2.plot(thetac1, thetac2, '-', color='black', zorder=zorder_cau)

        for cl in radcau:
            betac1, betac2 = self.lens.getCausticPoints(cl)
            if self.sersic:
                self.ax1.plot(betac1, betac2, '-', color='lightgray', zorder=zorder_cau)
            else:
                self.ax1.plot(betac1, betac2, '-', color='black', zorder=zorder_cau)


    def plot_sersics(self):
        kwargs_se = {
            'n': self.n_index,
            're': self.re,
            'q': 1.0,
            'pa': 0.0,
            'ys1': self.beta1,
            'ys2': self.beta2,
            'zs': self.lens.zs
        }
        se = sersic(sizex=self.extent1, sizey=self.extent2, Npix=self.lens.nray1, gl=self.lens, save_unlensed=True,
                    **kwargs_se)
        if self.showtd and not self.showconvergence:
            self.ax1.imshow(se.image_unlensed, origin='lower', cmap='afmhot', alpha=0.7, zorder=19, extent=self.extent)
            self.ax2.imshow(se.image, origin='lower', cmap='afmhot', alpha=0.7, zorder=19, extent=self.extent)
        elif self.showtd and self.showconvergence:
            self.ax1.imshow(se.image_unlensed, origin='lower', cmap='afmhot', alpha=0.7, zorder=0, extent=self.extent)
            # Combine the three images into an RGB image
            rgb_image = self.createRGB(se.image, self.lens.ka, self.lens.ka)
            self.ax2.imshow(rgb_image, origin='lower',
                            alpha=0.7, zorder=19,
                            extent=self.extent)
        else:
            self.ax1.imshow(se.image_unlensed, origin='lower', cmap='afmhot', alpha=0.7, zorder=0, extent=self.extent)

            if self.showconvergence and not self.showrgb:
                # Combine the three images into an RGB image
                rgb_image = self.createRGB(se.image, self.lens.ka, self.lens.ka)
                self.ax2.imshow(rgb_image, origin='lower',
                                alpha=1.0, zorder=0,
                                extent=self.extent)
            elif self.showrgb:
                #overlay_color = np.array([1.0, 0.0, 1.0])
                rgba_overlay = np.zeros((se.image.shape[0], se.image.shape[1], 4))
                #for i in range(3):  # R, G, B
                #    rgba_overlay[:, :, i] = overlay_color[i]
                normed_image = se.image / np.max(se.image)
                colormap = plt.colormaps.get_cmap("afmhot")  # Future-safe for matplotlib >= 3.7
                rgba_overlay = colormap(normed_image)  # shape: (2048, 2048, 4)
                composite = alpha_blend(self.rgb_image, rgba_overlay)
                self.ax2.imshow(composite, origin='lower', zorder=19, extent=self.extent, alpha=0.9)

                #self.ax2.imshow(se.image, origin='lower', cmap='afmhot', alpha=0.5, zorder=19, extent=self.extent)
            else:
                self.ax2.imshow(se.image, origin='lower', cmap='afmhot', alpha=0.7, zorder=0, extent=self.extent)

    def plot_point_source_images(self):
        self.ax1.plot(self.beta1, self.beta2, 'o', ms=5, color='red',zorder=100)

        thetai_1_arr = np.atleast_1d(self.thetai_1)
        if len(thetai_1_arr) > 0:
            isel = self.mui_plot > 25
            self.mui_plot[isel] = 25
            if self.showtd:
                self.ax2.scatter(self.thetai_1, self.thetai_2, s=(self.mui_plot * 5 + 15.0), marker='o', edgecolors='black', color='yellow',zorder=100)
            else:
                self.ax2.scatter(self.thetai_1, self.thetai_2, s=(self.mui_plot * 5 + 15.0), marker='o', edgecolors='black', color='red',zorder=100)

    def plot_lens(self):
        """
        Plot the lens and the source. This function is called when the plot is updated or when the user
        zooms in the plot
        :return:
        """
        td_surface = None
        if self.showtd or self.refine_td:
            td_surface = self.lens.t_delay_surf(beta=(self.beta1, self.beta2))
        self.current_td_surface = td_surface

        kwargs_psr = {
            'zs': self.lens.zs,
            'ys1': self.beta1,
            'ys2': self.beta2,
            'flux': 1.0
        }
        ps = pointsrc(
            sizex=self.extent1,
            sizey=self.extent2,
            Npix=self.lens.nray1,
            gl=self.lens,
            use_lenstronomy=False,
            refine=self.refine,
            refine_to_td=self.refine_td or self.showtd,
            td_surface=td_surface,
            **kwargs_psr,
        )
        self.thetai_1, self.thetai_2, self.mui = ps.xi1, ps.xi2, ps.mui
        self.mui_plot = np.atleast_1d(self.mui).copy()

        if self.showtd:
            self.plot_td(td=td_surface)

        if self.showconvergence and not self.sersic:
            self.plot_convergence()

        if self.showrgb:
            self.plot_rgb()

        self.plot_critlines_caustics()

        if self.sersic:
            self.plot_sersics()
        else:
            self.plot_point_source_images()

    def mouse_event(self, event):
        if self.predict_checkbox.isChecked():
            if event.button == 1:
                self.handle_prediction_click(event)
            return

        if self._using_manual_coordinates():
            return

        if event.button == 1:
            if event.inaxes in [self.ax1]:
                self.ax1.clear()
                self.ax2.clear()
                if hasattr(event.xdata, '__iter__'):
                    event.xdata = event.xdata[0]
                if hasattr(event.ydata, '__iter__'):
                    event.ydata = event.ydata[0]
                self.beta1 = event.xdata
                self.beta2 = event.ydata
                self.update_td_plot()
                self.update_table()
                self.update_source_table()
            elif event.inaxes in [self.ax2]:
                if hasattr(event.xdata, '__iter__'):
                    event.xdata = event.xdata[0]
                if hasattr(event.ydata, '__iter__'):
                    event.ydata = event.ydata[0]
                theta1 = event.xdata
                theta2 = event.ydata

                self.set_source_from_image_position(theta1, theta2)

                if hasattr(self.beta1, '__iter__'):
                    self.beta1 = self.beta1[0]
                if hasattr(self.beta2, '__iter__'):
                    self.beta2 = self.beta2[0]
                self.ax1.clear()
                self.ax2.clear()
                self.update_td_plot()
                self.update_table()
                self.update_source_table()

    def update_td_plot(self):
        self.plot_lens()
        self.update_axes()
        self.canvas.draw_idle()

    def createRGB(self,img1,img2,img3):
        # take sqrt of img2 and img3
        img2 = np.sqrt(img2)
        img3 = np.sqrt(img3)
        # Normalize each channel to the range [0, 1]
        img1 = (img1 - img1.min()) / (img1.max() - img1.min())
        img2 = (img2 - img2.min()) / (img2.max() - img2.min())
        img3 = (img3 - img3.min()) / (img3.max() - img3.min())
        # Set the maximum value for the blue channel
        max_blue_value = 0.3  # Set the maximum value as needed
        img3 = np.clip(img3, 0, max_blue_value)

        # Amplify the blue channel for better visibility
        blue_amplification_factor = 2.0
        img3 = img3 * blue_amplification_factor
        img3 = np.clip(img3, 0, 1)  # Ensure the values are still in the range [0, 1]

        # Combine the three images into an RGB image
        rgb_image = np.zeros((img1.shape[0], img1.shape[1], 3), dtype=img1.dtype)
        rgb_image[..., 0] = img1  # Red channel
        rgb_image[..., 1] = img2  # Green channel
        rgb_image[..., 2] = img3  # Blue channel

        # Apply gamma correction to enhance the overall brightness
        gamma = 0.9
        rgb_image = np.power(rgb_image, gamma)
        return rgb_image

    def onselect1(self, eclick, erelease):
        if eclick.inaxes == self.ax1:
            self.beta1_lim = sorted([eclick.xdata, erelease.xdata])
            self.beta2_lim = sorted([eclick.ydata, erelease.ydata])
            self.update_axes('source')
            #self.ax1.set_xlim(self.beta1_lim)
            #self.ax1.set_ylim(self.beta2_lim)
            #self.ax1.set_aspect('equal')
            #self.ax2.set_aspect('equal')
            #self.ax1.set_xlabel(r'$\beta_1$', fontsize=15)
            #self.ax1.set_ylabel(r'$\beta_2$', fontsize=15)
            #self.ax2.set_xlabel(r'$\theta_1$', fontsize=15)
            #elf.ax2.set_ylabel(r'$\theta_2$', fontsize=15)
            self.canvas.draw_idle()

    def onselect2(self, eclick, erelease):
        if eclick.inaxes == self.ax2:
            self.theta1_lim = sorted([eclick.xdata, erelease.xdata])
            self.theta2_lim = sorted([eclick.ydata, erelease.ydata])

            self.ax2.set_xlim(tuple(self.theta1_lim))
            self.ax2.set_ylim(tuple(self.theta2_lim))
            self.update_axes('lens')
            self.canvas.draw_idle()

    def convert_units(self, value, theta_label):
        # conversion from  [0,1] to physical units (beta or theta)
        converted_value = value  # default fallback value
        if theta_label == 'beta1':
            converted_value = (value - 0.5)*(self.beta1_lim[1]-self.beta1_lim[0])+np.mean(self.beta1_lim)
        elif theta_label == 'beta2':
            converted_value = (value - 0.5)*(self.beta2_lim[1]-self.beta2_lim[0])+np.mean(self.beta2_lim)
        elif theta_label == 'theta1':
            converted_value = (value - 0.5)*(self.theta1_lim[1]-self.theta1_lim[0])+np.mean(self.theta1_lim)
        elif theta_label == 'theta2':
            converted_value = (value - 0.5)*(self.theta2_lim[1]-self.theta2_lim[0])+np.mean(self.theta2_lim)
        return converted_value

class ShiftRectangleSelector(RectangleSelector):
    def __init__(self, *args, **kwargs):
        print (kwargs)
        super().__init__(*args, **kwargs)

    def press(self, event: Event) -> bool:
        if isinstance(event, MouseEvent) and event.button == MouseButton.RIGHT:
            return super().press(event)
        return False

    def release(self, event: Event) -> bool:
        if isinstance(event, MouseEvent) and event.button == MouseButton.RIGHT:
            result = super().release(event)
            self.set_visible(False)
            if self.canvas is not None:
                self.canvas.draw()
            return result
        return False

if __name__ == '__main__':
    #with open(os.getenv('LENSINTERACT_CONFIG', 'lensint_config.yml'), 'r') as file:
    #   config = yaml.safe_load(file)

    # section "clusters" of config yaml file should contain the name of the clusters and the paths to their deflection angles"
    #clusters = config['clusters']

    #print (clusters)

    app = QApplication(sys.argv)
    ex = LensInteract()
    sys.exit(app.exec())

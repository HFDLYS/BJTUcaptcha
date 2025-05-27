import sys
import os
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageQt, ImageOps

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QFileDialog, QSizePolicy, QGraphicsView, QGraphicsScene,
    QGraphicsPixmapItem, QLineEdit, QGroupBox, QTabWidget, QMessageBox,
    QColorDialog
)
from PyQt6.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QBrush
from PyQt6.QtCore import Qt, QPointF, QRectF

CHARSET = [' '] + ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] + ['+', '-', '*'] + ['=']
MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT = 130, 42

def decode_sequence(sequence, charset_list):
    decoded_chars = []
    prev_char_idx = -1
    blank_idx = charset_list.index(' ') if ' ' in charset_list else 0

    for idx in sequence:
        if idx == blank_idx:
            prev_char_idx = blank_idx
            continue
        if idx == prev_char_idx:
            continue
        if idx < 0 or idx >= len(charset_list):
            continue
        decoded_chars.append(charset_list[idx])
        prev_char_idx = idx
    return "".join(decoded_chars)

def preprocess_pil_image_for_onnx(pil_img, target_width, target_height):
    if pil_img is None:
        return None
    if pil_img.mode != 'RGB':
        pil_img = pil_img.convert('RGB')
    resample_method = Image.Resampling.LANCZOS if pil_img.width > target_width else Image.Resampling.BICUBIC
    img_resized = pil_img.resize((target_width, target_height), resample_method)
    img_np = np.array(img_resized, dtype=np.float32)
    img_np = img_np.transpose((2, 0, 1))
    img_np = img_np / 255.0
    img_np = np.expand_dims(img_np, axis=0)
    return img_np


class DrawingCanvas(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)

        self.image = QImage(MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT, QImage.Format.Format_RGB32)
        self.image.fill(Qt.GlobalColor.white) # Default background
        self.background_color = Qt.GlobalColor.white # Store background color

        self.pixmap_item = QGraphicsPixmapItem(QPixmap.fromImage(self.image))
        self.scene.addItem(self.pixmap_item)

        self.last_point = QPointF()
        self.drawing = False
        self.pen_color = Qt.GlobalColor.black # Default pen color
        self.pen_width = 3
        self.pen = QPen(self.pen_color, self.pen_width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)

        # --- MODIFICATION START ---
        # Removed setFixedSize to allow the canvas to resize with the window.
        # self.setFixedSize(MODEL_INPUT_WIDTH * 2 + 20, MODEL_INPUT_HEIGHT * 2 + 20)

        # Optionally, set a minimum size for the canvas.
        # This prevents it from becoming too small, while still allowing it to grow.
        self.setMinimumSize(MODEL_INPUT_WIDTH + 20, MODEL_INPUT_HEIGHT + 20)
        # --- MODIFICATION END ---

        self.setSceneRect(0, 0, MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT) # Scene matches image
        self.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio) # Initial fit


    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.last_point = self.pixmap_item.mapFromScene(self.mapToScene(event.pos()))
            self.drawing = True

    def mouseMoveEvent(self, event):
        if (event.buttons() & Qt.MouseButton.LeftButton) and self.drawing:
            current_point = self.pixmap_item.mapFromScene(self.mapToScene(event.pos()))
            painter = QPainter(self.image)
            painter.setPen(self.pen)
            painter.drawLine(self.last_point, current_point)
            painter.end()

            self.pixmap_item.setPixmap(QPixmap.fromImage(self.image))
            self.last_point = current_point

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.drawing:
            self.drawing = False

    def clear_canvas(self):
        self.image.fill(self.background_color) # Use stored background color
        self.pixmap_item.setPixmap(QPixmap.fromImage(self.image))

    def get_image(self):
        pil_img = ImageQt.fromqimage(self.image)
        # If your model expects inverted (white on black), and you draw black on white:
        # if self.pen_color == Qt.GlobalColor.black and self.background_color == Qt.GlobalColor.white:
        #     pil_img = ImageOps.invert(pil_img.convert('L')).convert('RGB')
        return pil_img

    def resizeEvent(self, event):
        # This ensures the content scales when the view is resized
        self.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        super().resizeEvent(event) # Call the base class implementation

    def set_pen_color(self, color):
        if isinstance(color, QColor) and color.isValid():
            self.pen_color = color
            self.pen.setColor(self.pen_color)

    def set_background_color(self, color):
        if isinstance(color, QColor) and color.isValid():
            self.background_color = color
            self.clear_canvas() # Redraw background with new color


class CaptchaInferenceApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Captcha CRNN Inference")
        self.setGeometry(100, 100, 650, 550)

        self.onnx_model_path = None
        self.ort_session = None
        self.current_pil_image = None

        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)

        model_group = QGroupBox("ONNX Model")
        model_layout = QHBoxLayout()
        self.btn_load_model = QPushButton("Load Model (.onnx)")
        self.btn_load_model.clicked.connect(self.load_model_dialog)
        self.lbl_model_path = QLabel("No model loaded.")
        self.lbl_model_path.setWordWrap(True)
        model_layout.addWidget(self.btn_load_model)
        model_layout.addWidget(self.lbl_model_path, 1)
        model_group.setLayout(model_layout)
        main_layout.addWidget(model_group)

        self.tabs = QTabWidget()
        self.tab_file = QWidget()
        self.tab_draw = QWidget()
        self.tabs.addTab(self.tab_file, "Load from File")
        self.tabs.addTab(self.tab_draw, "Draw Captcha")

        file_tab_layout = QVBoxLayout(self.tab_file)
        self.btn_load_image = QPushButton("Load Image File")
        self.btn_load_image.clicked.connect(self.load_image_dialog)
        self.lbl_image_display_file = QLabel("No image loaded.")
        self.lbl_image_display_file.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_image_display_file.setMinimumSize(MODEL_INPUT_WIDTH + 40, MODEL_INPUT_HEIGHT + 40)
        self.lbl_image_display_file.setStyleSheet("border: 1px solid gray; background-color: #f0f0f0;")
        file_tab_layout.addWidget(self.btn_load_image)
        file_tab_layout.addWidget(self.lbl_image_display_file, 1)

        draw_tab_layout = QVBoxLayout(self.tab_draw)
        self.drawing_canvas = DrawingCanvas()
        canvas_container_layout = QHBoxLayout()
        canvas_container_layout.addStretch()
        canvas_container_layout.addWidget(self.drawing_canvas) # DrawingCanvas will now expand
        canvas_container_layout.addStretch()
        draw_tab_layout.addLayout(canvas_container_layout)

        drawing_controls_layout = QHBoxLayout()
        self.btn_pen_color = QPushButton("Pen Color")
        self.btn_pen_color.clicked.connect(self.choose_pen_color)
        self.lbl_pen_color_preview = QLabel()
        self.lbl_pen_color_preview.setFixedSize(20, 20)
        self.update_pen_color_preview(self.drawing_canvas.pen_color)

        self.btn_bg_color = QPushButton("Background Color")
        self.btn_bg_color.clicked.connect(self.choose_background_color)
        self.lbl_bg_color_preview = QLabel()
        self.lbl_bg_color_preview.setFixedSize(20, 20)
        self.update_bg_color_preview(self.drawing_canvas.background_color)

        self.btn_clear_drawing = QPushButton("Clear Drawing")
        self.btn_clear_drawing.clicked.connect(self.drawing_canvas.clear_canvas)

        drawing_controls_layout.addStretch()
        drawing_controls_layout.addWidget(self.btn_pen_color)
        drawing_controls_layout.addWidget(self.lbl_pen_color_preview)
        drawing_controls_layout.addSpacing(20)
        drawing_controls_layout.addWidget(self.btn_bg_color)
        drawing_controls_layout.addWidget(self.lbl_bg_color_preview)
        drawing_controls_layout.addSpacing(20)
        drawing_controls_layout.addWidget(self.btn_clear_drawing)
        drawing_controls_layout.addStretch()
        draw_tab_layout.addLayout(drawing_controls_layout)

        main_layout.addWidget(self.tabs)

        infer_group = QGroupBox("Inference")
        infer_layout = QVBoxLayout()
        self.btn_infer = QPushButton("Run Inference")
        self.btn_infer.clicked.connect(self.run_inference)
        self.lbl_prediction = QLineEdit("Prediction will appear here.")
        self.lbl_prediction.setReadOnly(True)
        self.lbl_prediction.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = self.lbl_prediction.font()
        font.setPointSize(16)
        font.setBold(True)
        self.lbl_prediction.setFont(font)

        infer_layout.addWidget(self.btn_infer)
        infer_layout.addWidget(self.lbl_prediction)
        infer_group.setLayout(infer_layout)
        main_layout.addWidget(infer_group)

        self.setLayout(main_layout)

    def update_pen_color_preview(self, color):
        pixmap = QPixmap(self.lbl_pen_color_preview.size())
        pixmap.fill(color)
        self.lbl_pen_color_preview.setPixmap(pixmap)

    def update_bg_color_preview(self, color):
        pixmap = QPixmap(self.lbl_bg_color_preview.size())
        pixmap.fill(color)
        self.lbl_bg_color_preview.setPixmap(pixmap)

    def choose_pen_color(self):
        color = QColorDialog.getColor(self.drawing_canvas.pen_color, self, "Choose Pen Color")
        if color.isValid():
            self.drawing_canvas.set_pen_color(color)
            self.update_pen_color_preview(color)

    def choose_background_color(self):
        color = QColorDialog.getColor(self.drawing_canvas.background_color, self, "Choose Background Color")
        if color.isValid():
            self.drawing_canvas.set_background_color(color)
            self.update_bg_color_preview(color)

    def load_model_dialog(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Load ONNX Model", "", "ONNX Model Files (*.onnx)")
        if fname:
            self.onnx_model_path = fname
            try:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                if 'CUDAExecutionProvider' not in ort.get_available_providers():
                    print("CUDAExecutionProvider not available, using CPU.")
                    providers = ['CPUExecutionProvider']
                self.ort_session = ort.InferenceSession(self.onnx_model_path, providers=providers)
                self.lbl_model_path.setText(f"Loaded: {os.path.basename(fname)}")
                print(f"ONNX model loaded successfully from {fname} using {self.ort_session.get_providers()[0]}.")
            except Exception as e:
                self.lbl_model_path.setText("Error loading model.")
                QMessageBox.critical(self, "Model Load Error", f"Could not load ONNX model:\n{e}")
                self.ort_session = None
                self.onnx_model_path = None

    def load_image_dialog(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Load Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if fname:
            try:
                self.current_pil_image = Image.open(fname)
                # Display the image scaled to fit the label
                pixmap = QPixmap(fname)
                # Ensure label is visible and has a size before scaling
                if self.lbl_image_display_file.width() > 10 and self.lbl_image_display_file.height() > 10:
                    scaled_pixmap = pixmap.scaled(
                        self.lbl_image_display_file.width() - 10 ,
                        self.lbl_image_display_file.height() - 10,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation
                    )
                    self.lbl_image_display_file.setPixmap(scaled_pixmap)
                else: # Fallback if label size is not yet determined
                    self.lbl_image_display_file.setPixmap(pixmap.scaledToWidth(200, Qt.TransformationMode.SmoothTransformation))


            except Exception as e:
                QMessageBox.critical(self, "Image Load Error", f"Could not load image:\n{e}")
                self.current_pil_image = None
                self.lbl_image_display_file.setText("Error loading image.")


    def run_inference(self):
        if not self.ort_session:
            QMessageBox.warning(self, "Error", "Please load an ONNX model first.")
            return

        active_tab_index = self.tabs.currentIndex()
        image_to_infer = None

        if active_tab_index == 0: # Load from File tab
            if self.current_pil_image:
                image_to_infer = self.current_pil_image.copy()
            else:
                QMessageBox.warning(self, "Error", "Please load an image from file first.")
                return
        elif active_tab_index == 1: # Draw Captcha tab
            image_to_infer = self.drawing_canvas.get_image()
            if image_to_infer is None:
                 QMessageBox.warning(self, "Error", "Could not get image from drawing canvas.")
                 return
        else:
            QMessageBox.warning(self, "Error", "Unknown tab selected.")
            return

        try:
            input_data = preprocess_pil_image_for_onnx(image_to_infer, MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT)
            if input_data is None:
                QMessageBox.critical(self, "Error", "Image preprocessing failed.")
                return

            input_name = self.ort_session.get_inputs()[0].name
            ort_inputs = {input_name: input_data}
            ort_outs = self.ort_session.run(None, ort_inputs)

            preds_tensor = ort_outs[0] # Assuming output is the first element
            # The CRNN model typically outputs logits of shape (seq_len, batch_size, num_classes)
            # For inference with batch_size=1, it's (seq_len, 1, num_classes)
            # We need to take argmax over the last dimension (num_classes)
            preds_idx = np.argmax(preds_tensor[:, 0, :], axis=1) # Get char indices for each time step
            predicted_text = decode_sequence(preds_idx, CHARSET)

            self.lbl_prediction.setText(predicted_text if predicted_text else "[No text detected]")

        except Exception as e:
            QMessageBox.critical(self, "Inference Error", f"An error occurred during inference:\n{e}")
            self.lbl_prediction.setText("Error during inference.")
            print(f"Inference error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = CaptchaInferenceApp()
    main_window.show()
    sys.exit(app.exec())

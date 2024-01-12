import argparse
import io
import json
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import paramiko
from PIL import Image
from PIL.ImageQt import ImageQt
from PyQt5.QtCore import QEvent, Qt, pyqtSignal
from PyQt5.QtGui import (
    QFont,
    QFontMetrics,
    QImage,
    QIntValidator,
    QKeyEvent,
    QKeySequence,
    QPalette,
    QPixmap,
)
from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QScrollArea,
    QScrollBar,
    QShortcut,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


def key_based_connect(host, username, port, private_key_path):
    pkey = paramiko.RSAKey.from_private_key_file(private_key_path)
    client = paramiko.SSHClient()
    policy = paramiko.AutoAddPolicy()
    client.set_missing_host_key_policy(policy)
    client.connect(host, username=username, pkey=pkey, port=port)
    sftp_client = client.open_sftp()
    return sftp_client


def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


class SwitchSignal(QWidget):
    next = pyqtSignal()
    prev = pyqtSignal()

    def keyPresseEvent(self, ev: QKeyEvent):
        super().keyPressEvent(ev)
        if ev.key() == Qt.Key_Up:
            print("KEy UP")
            self.prev.emit()
        elif ev.key() == Qt.Key_Down:
            print("KEy Down")
            self.next.emit()


class Dataset:
    def __init__(
        self,
        list_file: Optional[Path] = None,
        delete_file_logging: Optional[Path] = None,
        image_dir: Optional[Path] = None,
        replace_texts: Optional[List] = [],
        label_dir: Optional[Path] = None,
        client: Optional[paramiko.SFTPClient] = None,
    ):
        if list_file is not None and list_file.suffix == ".json":
            textlines = []
            predict_text = []

            context = (
                client.open(list_file.as_posix(), mode="r")
                if client is not None
                else open(list_file.as_posix(), encoding="utf-8")
            )
            file_info = json.load(context)
            context.close()

            for item in file_info["data"]:
                image_path = Path(item["image_path"])
                textline = TextLine(
                    image_path, replace_texts, label_dir=None, client=client
                )
                textlines.append(textline)
                if item.get("text", None) is not None:
                    predict_text.append(item["text"])

            self.textlines = textlines
            if len(predict_text) > 0:
                self.predict_text = predict_text
        else:
            if list_file:
                file_obj = (
                    client.open(list_file.as_posix(), mode="r").readlines()
                    if client is not None
                    else open(list_file, encoding="utf-8").readlines()
                )
                image_lines = [image_dir.joinpath(line.strip()) for line in file_obj]
            else:
                if client is not None:
                    image_lines = client.listdir(image_dir.as_posix())
                    image_lines = [
                        image_dir.joinpath(image_line)
                        for image_line in image_lines
                        if not os.path.splitext(image_line)[-1] == ".txt"
                    ]
                else:
                    image_lines = list(image_dir.rglob("*g"))

                image_lines = sorted(image_lines, key=lambda x: x.name)

                if label_dir:
                    if client is not None:
                        label_paths = client.listdir(label_dir.as_posix())
                        label_paths = list(
                            map(lambda x: label_dir.joinpath(x), label_paths)
                        )
                    else:
                        label_paths = list(label_dir.rglob("*txt"))

                    assert len(image_lines) == len(
                        label_paths
                    ), f"len image {len(image_lines)} vs len label {len(label_paths)}"

                    label_paths = sorted(label_paths, key=lambda x: x.name)
                else:
                    label_paths = list(
                        map(lambda x: x.with_suffix(".txt"), image_lines)
                    )

            textlines = []
            for idx, image_path in enumerate(image_lines):
                if list_file:
                    textline = TextLine(image_path, replace_texts, client=client)
                else:
                    textline = TextLine(
                        image_path, replace_texts, label_paths[idx], client=client
                    )
                textlines.append(textline)

            self.textlines = textlines

        self.delete_file_logging = delete_file_logging
        self.path2idx = {
            textline.image_path.as_posix(): idx
            for idx, textline in enumerate(textlines)
        }

    def __getitem__(self, idx):
        if hasattr(self, "predict_text"):
            return self.textlines[idx], self.predict_text[idx]
        else:
            return self.textlines[idx]

    def __len__(self):
        return len(self.textlines)


class TextLine:
    def __init__(
        self,
        image_path: Path,
        replace_texts: Optional[List] = None,
        label_dir: Optional[Path] = None,
        client: Optional[paramiko.SFTPClient] = None,
    ):
        if len(replace_texts):
            image_path = str(image_path).replace(*replace_texts[0])
            image_path = Path(image_path)

        if len(replace_texts) > 1:
            assert label_dir is not None
            label_dir = str(label_dir).replace(*replace_texts[-1])
            label_dir = Path(label_dir)

        self.image_path = image_path
        self.client = client

        if not label_dir:
            self.txt_path = self.image_path.with_suffix(".txt")
        else:
            self.txt_path = label_dir

        self.is_fix = None

    def textline(self):
        if self.client:
            textline: str = self.client.open(
                self.txt_path.as_posix(), mode="r"
            ).readline()
            return textline.strip()
        else:
            return open(self.txt_path, encoding="utf-8").readline().strip()

    def save(self, new_label):
        self.is_fix = True
        if self.client:
            self.client.open(self.txt_path.as_posix(), "w").write(new_label)
        else:
            open(self.txt_path, "wt", encoding="utf-8").write(new_label)

    def save_image(self, new_image: Image.Image):
        if self.client:
            buf = io.BytesIO()
            new_image.save(buf, format="PNG")
            byte_im = buf.getvalue()
            self.client.open(self.image_path.as_posix(), mode="w").write(byte_im)
        else:
            new_image.save(self.image_path)


class App(QMainWindow):
    def __init__(
        self,
        list_file: Optional[Path] = None,
        delete_file_logging: Optional[Path] = None,
        image_dir: Optional[Path] = None,
        replace_texts: Optional[List] = [],
        label_dir: Optional[Path] = None,
        ssh_client: Optional[paramiko.SFTPClient] = None,
    ):
        super().__init__()
        assert (
            list_file or image_dir
        ), f"Either list_file or image_dir should be provided"

        self.list_file = list_file
        self.ssh_client = ssh_client

        if list_file is not None and list_file.suffix == ".json":
            WIN_SIZE = (1024, 260)
        else:
            WIN_SIZE = (1024, 128)

        self.image = QImage()
        self.scaleFactor = 1.0

        self.imageLabel = QLabel()
        self.imageLabel.setBackgroundRole(QPalette.Base)
        self.imageLabel.setFixedSize(1024, 64)
        self.imageLabel.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.imageLabel.setScaledContents(False)

        layout = QVBoxLayout()

        ##################
        # Index
        ##################
        index_widget = QWidget()
        index_layout = QHBoxLayout()
        index_layout.setAlignment(Qt.AlignLeft)

        self.current_line_index = QLineEdit(f"{0:07d}")
        self.current_line_index.setValidator(QIntValidator())
        self.current_line_index.setMaxLength(6)
        self.current_line_index.setFixedWidth(60)
        self.current_line_index.editingFinished.connect(self.jump_to_line_index)
        self.total_line_label = QLabel("/0")
        index_layout.addWidget(self.current_line_index)
        index_layout.addWidget(self.total_line_label)

        self.current_path_label = QLineEdit("path")
        self.current_path_label.setFixedWidth(1000)
        self.current_path_label.setReadOnly(False)
        self.current_path_label.editingFinished.connect(self.jump_to_path)
        index_layout.addWidget(self.current_path_label)

        index_widget.setLayout(index_layout)
        layout.addWidget(index_widget)

        label = QLabel("Image:")
        layout.addWidget(label)

        self.scrollArea = QScrollArea()
        self.scrollArea.setBackgroundRole(QPalette.Dark)
        self.scrollArea.setWidget(self.imageLabel)
        self.scrollArea.setVisible(False)
        self.scrollArea.setWidgetResizable(True)
        layout.addWidget(self.scrollArea)

        label = QLabel("Label:")
        layout.addWidget(label)

        self.label_text = QLineEdit("label")
        f = self.label_text.font()
        f.setPointSize(27)  # sets the size to 27
        f.setStyleHint(QFont.Monospace)
        self.label_text.setFont(f)
        self.label_text.setFocus()
        self.label_text.setReadOnly(False)
        layout.addWidget(self.label_text)

        if list_file and list_file.suffix == ".json":
            predict = QLabel("Predict:")
            layout.addWidget(predict)

            self.predict_text = QLineEdit("predict")
            f = self.predict_text.font()
            f.setPointSize(27)  # sets the size to 27
            f.setStyleHint(QFont.Monospace)
            self.predict_text.setFont(f)
            self.predict_text.setFocus()
            self.predict_text.setReadOnly(False)
            layout.addWidget(self.predict_text)

        signal_widget = SwitchSignal()
        signal_widget.next.connect(self.next_image)
        signal_widget.prev.connect(self.prev_image)
        layout.addWidget(signal_widget)

        root = QWidget()
        root.setLayout(layout)
        self.setCentralWidget(root)

        self.resize(WIN_SIZE[0], WIN_SIZE[1])

        self.current_index = 0

        self.dataset = Dataset(
            list_file,
            delete_file_logging,
            image_dir,
            replace_texts,
            label_dir,
            ssh_client,
        )

        if len(self.dataset) == 0:
            print("Nothing to do! Nice!")
            exit(0)

        self.label_text.installEventFilter(self)

        self.need_save = False
        if list_file is not None and list_file.suffix == ".json":
            result = self.dataset[0]
            if isinstance(result, tuple):
                self.current_textline, self.predict_textline = result
            else:
                self.current_textline = result
        else:
            self.current_textline = self.dataset[0]
        self.total_line_label.setText(f"{len(self.dataset) - 1:05d}")
        self.set_step(0)

        #######################
        # Set shortcut
        shortcut = QShortcut(QKeySequence("Ctrl+S"), self)
        shortcut.activated.connect(self.save_current_line)

        shortcut_rotate = QShortcut(QKeySequence("Ctrl+R"), self)
        shortcut_rotate.activated.connect(self.rotate)

        shortcut_flip = QShortcut(QKeySequence("Ctrl+F"), self)
        shortcut_flip.activated.connect(self.flip)

        shortcut_vflip = QShortcut(QKeySequence("Ctrl+H"), self)
        shortcut_vflip.activated.connect(self.vflip)

        shortcut_delete = QShortcut(QKeySequence("Ctrl+D"), self)
        shortcut_delete.activated.connect(self.delete)
        self.delete_file_logging = self.dataset.delete_file_logging

        self._is_rotate = False
        self._is_flip = False

    def save_current_line(self):
        if self.current_textline.textline() != self.label_text.text():
            print("Save text")
            self.current_textline.save(self.label_text.text())

        if self._is_rotate or self._is_flip:
            self._is_rotate = False
            self._is_flip = False
            print("Save image")
            self.current_textline.save_image(self.pillow_image)

    def jump_to_line_index(self):
        step = int(self.current_line_index.text())
        self.set_step(step)

    def jump_to_path(self):
        path_ = str(self.current_path_label.text().strip())
        path_idx = self.dataset.path2idx[path_]
        self.set_step(path_idx)

    def eventFilter(self, source, event):
        if event.type() == QEvent.KeyPress and source is self.label_text:
            if event.key() == Qt.Key_Down:
                self.next_image()
            elif event.key() == Qt.Key_Up:
                self.prev_image()
        return super().eventFilter(source, event)

    def next_image(self):
        self.set_step(self.current_index + 1)

    def prev_image(self):
        self.set_step(self.current_index - 1)

    def is_able_to_next(self, step):
        if step >= len(self.current_textline):
            if self.current_step == len(self.account) - 1:
                return False
        return True

    def is_able_to_back(self, step):
        if step < 0:
            if self.current_step == 0:
                return False
        return True

    def set_step(self, step):
        if step >= len(self.dataset) or step < 0:
            return

        if step != self.current_index:
            # check if next/prev image without save
            if self.current_textline.textline() != self.label_text.text():
                buttonReply = QMessageBox.question(
                    self,
                    "Text not saved yet?",
                    "Do you like to save?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )
                if buttonReply == QMessageBox.Yes:
                    self.save_current_line()

        if self.list_file and self.list_file.suffix == ".json":
            result = self.dataset[step]
            if isinstance(result, tuple):
                self.current_textline, self.predict_textline = result
            else:
                self.current_textline = result
        else:
            self.current_textline = self.dataset[step]

        self.current_index = step

        if self.ssh_client:
            img_byte = self.ssh_client.open(
                self.current_textline.image_path.as_posix(), mode="r"
            ).read()
            img_byte = io.BytesIO(img_byte)
            image = Image.open(img_byte)
        else:
            image = Image.open(self.current_textline.image_path)

        label = self.current_textline.textline()

        if image.size[0] * image.size[1] == 0:
            print(f"Width or height is 0. WxH = {image.size[0]}x{image.size[1]}")
            if self.is_able_to_next(step):
                self.next_image()
                return
            elif self.is_able_to_back(step):
                self.prev_image()
                return
            else:
                print("Done!")
                exit(0)

        self.pillow_image: Image.Image = image
        self.current_angle = 0

        self.current_line_index.setText(f"{self.current_index:05d}")
        self.label_text.setText(label)
        if self.list_file and self.list_file.suffix == ".json":
            self.predict_text.setText(self.predict_textline)
        self.loadImage(image)

        #### Resize font
        # Use binary search to efficiently find the biggest font that will fit.
        max_size = 27
        min_size = 1
        font = self.label_text.font()
        while 1 < max_size - min_size:
            new_size = (min_size + max_size) // 2
            font.setPointSize(new_size)
            metrics = QFontMetrics(font)

            target_rect = self.label_text.contentsRect()

            # Be careful which overload of boundingRect() you call.
            rect = metrics.boundingRect(target_rect, Qt.AlignLeft, label)
            if (
                rect.width() > target_rect.width()
                or rect.height() > target_rect.height()
            ):
                max_size = new_size
            else:
                min_size = new_size

        font.setPointSize(min_size)
        self.label_text.setFont(font)
        if self.list_file and self.list_file.suffix == ".json":
            self.predict_text.setFont(font)

    def loadImage(self, pillow_image: Image.Image):
        image_w, image_h = pillow_image.size
        target_h = 64
        factor = target_h / image_h
        image_w = factor * image_w
        image_h = factor * image_h
        image_w, image_h = int(image_w), int(image_h)
        pillow_image = pillow_image.resize((image_w, image_h))

        self.scrollArea.setVisible(True)
        self.image = ImageQt(pillow_image)
        self.imageLabel.setPixmap(QPixmap.fromImage(self.image))
        self.imageLabel.setFixedSize(image_w, image_h)

        self.adjustScrollBar(self.scrollArea.horizontalScrollBar(), 0)
        self.adjustScrollBar(self.scrollArea.verticalScrollBar(), 1.0)

        self.current_path_label.setText(str(self.current_textline.image_path))
        message = "{}, {}x{}, Depth: {}".format(
            self.current_textline.image_path,
            self.image.width(),
            self.image.height(),
            self.image.depth(),
        )
        self.statusBar().showMessage(message)
        return True

    def rotate(self):
        self.current_angle += 90
        self.pillow_image = self.pillow_image.rotate(self.current_angle, expand=True)
        self.loadImage(self.pillow_image)
        self._is_rotate = True

    def flip(self):
        self.pillow_image = self.pillow_image.transpose(Image.FLIP_LEFT_RIGHT)
        self.loadImage(self.pillow_image)
        self._is_flip = True

    def vflip(self):
        self.pillow_image = self.pillow_image.transpose(Image.FLIP_TOP_BOTTOM)
        self.loadImage(self.pillow_image)
        self._is_flip = True

    def adjustScrollBar(self, scrollBar: QScrollBar, factor: float):
        scrollBar.setValue(
            int(factor * scrollBar.value() + ((factor - 1) * scrollBar.pageStep() / 2))
        )

    def delete(self):
        current_path = self.current_textline.image_path
        logging_dir = self.delete_file_logging.parent
        if not logging_dir.exists():
            logging_dir.mkdir(parents=True)
        print(f"remove_path: {current_path}")
        if self.ssh_client:
            self.ssh_client.open(self.delete_file_logging.as_posix(), mode="a").write(
                f"{current_path}\n"
            )
        else:
            with open(self.delete_file_logging, mode="a") as f:
                f.write(f"{current_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--list_file",
        default=None,
        type=Path,
        help="Path to file contain list of image paths or JSON file contain image prediction information from LIB OCR",
    )
    parser.add_argument(
        "--delete_file_logging",
        default=None,
        type=Path,
        help="Path to logging file to store removed image path",
    )
    parser.add_argument(
        "--image_dir",
        default=None,
        type=Path,
        help="Path to directory of image and label (optional) of dataset, do not provide if already set list_file argument",
    )
    parser.add_argument(
        "--replace_text",
        default=None,
        nargs="*",
        help="Replace parent path of data directory between local and server",
    )
    parser.add_argument(
        "--label_dir",
        default=None,
        type=Path,
        help="Directory of label corresponding to the image (optional), useful when apply for pseudo label scenario, where folder contain pseudo label differ from folder contain images",
    )
    parser.add_argument(
        "--server_name",
        default=None,
        help="Name of server in case of using data directly in server",
    )
    args = parser.parse_args()

    if args.server_name:
        ssh_path = Path(os.getenv("HOME")) / ".ssh" / "config"
        ssh_config = list(map(lambda x: x.strip(), open(ssh_path).readlines()))
        server_info = [
            ssh_config[idx + 1 : idx + 5]
            for idx in range(len(ssh_config))
            if ssh_config[idx] == f"Host {args.server_name}"
        ]
        server_info = [line.strip().split()[-1] for line in server_info[0]]
        server_dict = {
            param_name: ssh_value
            for param_name, ssh_value in zip(
                ["host", "username", "port", "private_key_path"], server_info
            )
        }
        server_dict.update(
            {"private_key_path": ssh_path.parent.joinpath("id_rsa").as_posix()}
        )
        print("Connect to ssh server with info")
        print([f"{k} : {v}" for k, v in server_dict.items()])

        client = key_based_connect(**server_dict)
    else:
        client = None

    app = QApplication([])
    window = App(
        list_file=args.list_file,
        delete_file_logging=args.delete_file_logging,
        image_dir=args.image_dir,
        replace_texts=[] if not args.replace_text else [tuple(args.replace_text)],
        label_dir=args.label_dir,
        ssh_client=client,
    )
    window.show()
    app.exec_()

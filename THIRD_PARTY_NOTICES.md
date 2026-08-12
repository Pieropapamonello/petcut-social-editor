# Third-party model notice

PetCut includes `data/models/yolov8n-seg.onnx`, an ONNX export of the
Ultralytics YOLOv8n segmentation model trained on COCO. It is used only to
separate people and animals from their backgrounds.

Ultralytics YOLOv8 is distributed under the GNU Affero General Public License
v3.0. Its source and licence are available at:

- https://github.com/ultralytics/ultralytics
- https://www.gnu.org/licenses/agpl-3.0.html

The model file was exported at 640×640 with ONNX opset 17. PetCut performs
inference locally on the server; uploaded media is not sent to a segmentation
API.

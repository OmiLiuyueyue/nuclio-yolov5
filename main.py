import json
import base64
from PIL import Image
import io

import cv2
import numpy as np
import argparse
import onnxruntime as ort


class YOLO:
    def __init__(self, model_path, input_w=640, input_h=640,
                 conf_threshold=0.35, nms_threshold=None):
        if nms_threshold is None:
            nms_threshold = [0.5, 0.35, 0.35]
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.model = ort.InferenceSession(model_path, so)
        self.classes = {0: "person", 1: "bicycle", 2: "motorbike", 3: "tricycle",
                        4: "car", 5: "bus", 6: "truck", 7: "plate", 8: "R", 9: "G", 10: "Y"}
        self.vehicle_colors = [[103, 196, 209], [50, 205, 50], [180, 105, 255], [38, 71, 139],
                               [251, 127, 36], [64, 64, 255], [219, 112, 147], [255, 191, 0],
                               [0, 0, 255], [0, 255, 255], [0, 255, 0]]
        self.groups = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 1, 8: 2, 9: 2, 10: 2}

        self.input_w, self.input_h = input_w, input_h
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.det_size = 4 + len(self.classes) + 1
        self.output_size = 25200 * self.det_size

    def preprocess(self, frame):
        rows, cols = frame.shape[:2]
        r_w = self.input_w / cols
        r_h = self.input_h / rows
        if r_h > r_w:
            w = self.input_w
            h = int(r_w * rows)
            x = 0
            y = int((self.input_h - h) / 2)
        else:
            w = int(r_h * cols)
            h = self.input_h
            x = int((self.input_w - w) / 2)
            y = 0
        re = cv2.resize(frame, (w, h))
        pr_img = 128 * np.ones((self.input_h, self.input_w, 3), dtype=np.uint8)
        pr_img[y:(y + h), x:(x + w), :] = re

        pr_img = pr_img / 255.0
        pr_img = pr_img[:, :, ::-1].transpose((2, 0, 1))
        pr_img = np.expand_dims(pr_img, axis=0)

        return pr_img

    def infer(self, data):
        input_name = self.model.get_inputs()[0].name
        output_name = self.model.get_outputs()[0].name
        pred = self.model.run([output_name], {input_name: data.astype(np.float32)})[0]
        output = np.squeeze(pred)
        return output

    def iou(self, lbox, rbox):
        interBox = [max(lbox[0] - lbox[2] / 2.0, rbox[0] - rbox[2] / 2.0),
                    min(lbox[0] + lbox[2] / 2.0, rbox[0] + rbox[2] / 2.0),
                    max(lbox[1] - lbox[3] / 2.0, rbox[1] - rbox[3] / 2.0),
                    min(lbox[1] + lbox[3] / 2.0, rbox[1] + rbox[3] / 2.0)]
        if (interBox[2] > interBox[3]) or (interBox[0] > interBox[1]):
            return 0.0
        interBoxS = (interBox[1] - interBox[0]) * (interBox[3] - interBox[2])
        return interBoxS / (lbox[2] * lbox[3] + rbox[2] * rbox[3] - interBoxS)

    def get_rect(self, image, results):
        rows, cols = image.shape[:2]
        for i, target in enumerate(results):
            bbox = target[:4]
            r_w = float(self.input_w / (cols * 1.0))
            r_h = float(self.input_h / (rows * 1.0))
            if r_h > r_w:
                l = int(bbox[0] - bbox[2] / 2.0)
                r = int(bbox[0] + bbox[2] / 2.0)
                t = int(bbox[1] - bbox[3] / 2.0 - (self.input_h - r_w * rows) / 2)
                b = int(bbox[1] + bbox[3] / 2.0 - (self.input_h - r_w * rows) / 2)
                l = int(l / r_w)
                r = int(r / r_w)
                t = int(t / r_w)
                b = int(b / r_w)
            else:
                l = int(bbox[0] - bbox[2] / 2.0 - (self.input_w - r_h * cols) / 2)
                r = int(bbox[0] + bbox[2] / 2.0 - (self.input_w - r_h * cols) / 2)
                t = int(bbox[1] - bbox[3] / 2.0)
                b = int(bbox[1] + bbox[3] / 2.0)
                l = int(l / r_h)
                r = int(r / r_h)
                t = int(t / r_h)
                b = int(b / r_h)
            results[i][0] = l
            results[i][1] = t
            results[i][2] = r
            results[i][3] = b
            results[i][4] = target[4]
            results[i][5] = target[5]
        return results

    def post_process(self, output):
        res = []
        m = {0: [], 1: [], 2: []}
        for det in output:
            if det[4] <= self.conf_threshold:
                continue
            class_id = np.argmax(det[5:])
            group_id = self.groups[class_id]
            m[group_id].append(det[:5].tolist() + [class_id])
        for group_id, dets in m.items():
            dets = np.array(dets)
            dets = dets[np.argsort(dets[:, 4])][::-1]
            i = 0
            while True:
                det = dets[i]
                res.append(det)
                j = i + 1
                while True:
                    if self.iou(det, dets[j]) > self.nms_threshold[group_id]:
                        dets = np.delete(dets, j, axis=0)
                        j -= 1
                    j += 1
                    if j >= len(dets) -1:
                        break
                i += 1
                if i >= len(dets)-1:
                    break
        return res

    def plot(self, frame, result):
        # 绘制检测目标
        for target in result:
            # 绘制当前目标
            x1, y1, x2, y2, score, class_id = target
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.vehicle_colors[int(class_id)], 2)
            cv2.putText(frame, "%s" % self.classes[class_id], (x1, y1 - 12),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, self.vehicle_colors[int(class_id)], 2)
        return frame

    def detect(self, frame):
        data = self.preprocess(frame)
        output = self.infer(data)
        res = self.post_process(output)
        result = self.get_rect(frame, res)
        return result


yolo = YOLO(model_path="./yolov5n.onnx")

if __name__ == '__main__':
    frame = cv2.imread("test.jpg")
    frame = cv2.resize(frame, (1280, 720))
    result = yolo.detect(frame)
    image = yolo.plot(frame, result)
    cv2.imshow("test", image)
    cv2.waitKey(0)


def init_context(context):
    context.logger.info("Init context...  0%")
    context.logger.info("Init context...100%")


def handler(context, event):
    context.logger.info("Run yolo-v5 model")
    data = event.body
    buf = io.BytesIO(base64.b64decode(data["image"]))
    image = np.array(Image.open(buf))
    yolo_results = yolo.detect(image)

    encoded_results = []
    for result in yolo_results:
        x1, y1, x2, y2, score, class_id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        encoded_results.append({
            'confidence': score,
            'label': yolo.classes[class_id],
            'points': [x1, y1, x2, y2],
            'type': 'rectangle'
        })

    return context.Response(body=json.dumps(encoded_results), headers={},
                            content_type='application/json', status_code=200)

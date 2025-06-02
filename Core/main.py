from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import os
from email.mime.base import MIMEBase
from email import encoders
import smtplib
import geocoder
from time import time
import cv2
import torch
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from dotenv import load_dotenv

load_dotenv()
key = os.getenv("MAIL_TOKEN") # should be your token/password, generated for your application.
from_email = "tdummy027@gmail.com"  # must match the email used to generate the password.
to_email = "superinfinity5@gmail.com"  # receiver email.

with open(key) as f:
    password = f.read()

server = smtplib.SMTP("smtp.gmail.com: 587")
server.starttls()
server.login(from_email, password)


def get_location():
    """Fetch the current location (latitude, longitude) using geocoder based on the device's IP address."""
    try:
        g = geocoder.ipinfo('me')
        if g.ok:
            return g.latlng
        else:
            print("Error fetching location:", g.error)
    except geocoder.GeocoderServiceError as e:
        print("Geocoder service error:", e)
    except geocoder.GeocoderNotFound as e:
        print("Geocoder not found:", e)
    return None

def send_email(to_email, from_email, object_detected=1, image_path=None):
    """Sends an email notification with the number of detected objects and attaches the image of violence detected."""
    message = MIMEMultipart()
    message["From"] = from_email
    message["To"] = to_email
    message["Subject"] = "Security Alert"

    # Add the message body
    # Add the message body with location
    location = get_location()
    if location:
        message_body = f"ALERT - {object_detected} objects have been detected!!\n\nLocation: {location[0]}, {location[1]}"
    else:
        message_body = f"ALERT - {object_detected} objects have been detected!!\n\nLocation: Not available"

    message.attach(MIMEText(message_body, "plain"))

    # Attach the image file if provided
    if image_path:
        with open(image_path, "rb") as f:
            mime = MIMEBase("application", "octet-stream")
            mime.set_payload(f.read())
        encoders.encode_base64(mime)
        mime.add_header("Content-Disposition", f'attachment; filename="{os.path.basename(image_path)}"')
        message.attach(mime)

    # Send the email
    server.sendmail(from_email, to_email, message.as_string())



class ObjectDetection:
    def __init__(self, capture_index):
        """Initializes an ObjectDetection instance with a given camera index."""
        self.capture_index = capture_index
        self.email_sent = False

        # model information
        self.model = YOLO("../Models/violenceD.pt")

        # visual information
        self.annotator = None
        self.start_time = 0
        self.end_time = 0
        self.violence_class_id = 1

        # device information
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Firebase init
        # self.db_main = DbMain()

    def predict(self, im0):
        """Run prediction using a YOLO model for the input image `im0`."""
        results = self.model(im0)
        return results

    def display_fps(self, im0):
        """Displays the FPS on an image `im0` by calculating and overlaying as white text on a black rectangle."""
        self.end_time = time()
        fps = 1 / round(self.end_time - self.start_time, 2)
        text = f"FPS: {int(fps)}"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        gap = 10
        cv2.rectangle(
            im0,
            (20 - gap, 70 - text_size[1] - gap),
            (20 + text_size[0] + gap, 70 + gap),
            (255, 255, 255),
            -1,
        )
        cv2.putText(im0, text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

    def plot_bboxes(self, results, im0):
        """Plots bounding boxes on an image given detection results; returns annotated image and class IDs."""
        class_ids = []
        self.annotator = Annotator(im0, 3, results[0].names)
        boxes = results[0].boxes.xyxy.cpu()
        clss = results[0].boxes.cls.cpu().tolist()
        names = results[0].names
        for box, cls in zip(boxes, clss):
            class_ids.append(cls)
            if int(cls) == 0:
                color_id = 13
                self.annotator.box_label(box, label=names[int(cls)], color=colors(color_id, True))
            elif int(cls) == 1:
                color_id = 6
                self.annotator.box_label(box, label=names[int(cls)], color=colors(color_id, True))
        return im0, class_ids

    def __call__(self):
        """Run object detection on video frames from a camera stream, plotting and showing the results."""
        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1920)
        frame_count = 0
        while True:
            self.start_time = time()
            ret, im0 = cap.read()
            assert ret
            results = self.predict(im0)
            im0, class_ids = self.plot_bboxes(results, im0)

            if self.violence_class_id in class_ids:
                if not self.email_sent:
                    # Save the frame where violence is detected
                    image_path = f"Proof/violence_detected_frame_{frame_count}.jpg"
                    cv2.imwrite(image_path, im0)

                    # Send email with the image attached
                    send_email(to_email, from_email, len(class_ids), image_path=image_path)
                    self.email_sent = True

            else:
                self.email_sent = False
            self.display_fps(im0)
            cv2.imshow("YOLOv8 Detection", im0)
            frame_count += 1
            if cv2.waitKey(5) & 0xFF == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
        server.quit()


d1 = 0
d = "../Data/v2.mp4"
detector = ObjectDetection(capture_index=d)
detector()
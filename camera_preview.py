import cv2
from pathlib import Path
from datetime import datetime
import time


def main():
    cap = cv2.VideoCapture(0)
    writer = None
    record_fps = 20
    record_frames = 0
    record_start = 0
    output_path = None

    try:
        if not cap.isOpened():
            print("Cannot open camera.")
            return
        lower_red1 = (0, 180, 50)
        upper_red1 = (10, 255, 255)
        lower_red2 = (170, 180, 50)
        upper_red2 = (179, 255, 255)
        min_area = 1000
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Cannot read a frame.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask = cv2.bitwise_or(mask1, mask2)

            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

            height, width, channels = frame.shape
            center_x = width // 2
            center_y = height // 2
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
            text = "No target"            

            if contours:
                largest = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest)

                if area >= min_area:
                    x, y, w, h = cv2.boundingRect(largest)
                    target_x = x + w // 2
                    target_y = y + h // 2

                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.circle(frame, (target_x, target_y), 5, (0, 255, 0), -1)

                    text = f"X: ({target_x}, Y: {target_y})"
            cv2.putText(
                frame, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    

            if writer is not None:
                elapsed = time.monotonic() - record_start
                due_frames = min(600, int(elapsed * record_fps))
                while record_frames < due_frames:
                    writer.write(frame)
                    record_frames += 1
                if record_frames >= 600:
                    writer.release()
                    writer = None
                    print("Saved 30-second video:", output_path)

            cv2.imshow("Camera Preview", frame)
            cv2.imshow("Grayscale", gray)
            cv2.imshow("Mask", mask)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
            elif key in (ord("i"), ord("I")):
                print("frame.shape (height, width, channels):", frame.shape)
            elif key in (ord("r"), ord("R")) and writer is None:
                video_dir = Path(__file__).resolve().parent / "recordings"
                video_dir.mkdir(exist_ok=True)
                stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
                output_path = video_dir / f"color-tracker-{stamp}.mp4"
                writer = cv2.VideoWriter(
                    str(output_path), cv2.VideoWriter_fourcc(*"mp4v"),
                    record_fps, (width, height)
                )
                if not writer.isOpened():
                    writer.release()
                    writer = None
                    print("Cannot start video recording.")
                else:
                    record_start = time.monotonic()
                    record_frames = 0
                    print("Recording started. Auto-stop in 30 seconds.")
    finally:
        if writer is not None:
            writer.release()
            print("Recording stopped early:", output_path)
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

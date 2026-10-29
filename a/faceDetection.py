"""Detect frontal faces in a webcam feed or image using OpenCV."""

import argparse
from pathlib import Path

import cv2


def load_detector():
    cascade = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(cascade))
    if detector.empty():
        raise RuntimeError(f"Could not load face detector: {cascade}")
    return detector


def annotate_faces(frame, detector):
    """Return an annotated copy and the number of detected faces."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = detector.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    result = frame.copy()
    for x, y, width, height in faces:
        cv2.rectangle(result, (x, y), (x + width, y + height), (0, 255, 0), 2)
    cv2.putText(
        result, f"Faces: {len(faces)}", (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
    )
    return result, len(faces)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, help="Image path; defaults to webcam mode")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument("--output", type=Path, help="Save the annotated image (requires --image)")
    parser.add_argument("--no-display", action="store_true", help="Process an image without opening a window")
    args = parser.parse_args(argv)
    if not args.image and (args.output or args.no_display):
        parser.error("--output and --no-display require --image")

    capture = None
    try:
        detector = load_detector()
        if args.image:
            frame = cv2.imread(str(args.image))
            if frame is None:
                raise RuntimeError(f"Could not read image: {args.image}")
            result, count = annotate_faces(frame, detector)
            if args.output:
                if not cv2.imwrite(str(args.output), result):
                    raise RuntimeError(f"Could not save image: {args.output}")
                print(f"Saved: {args.output}")
            print(f"Faces detected: {count}")
            if not args.no_display:
                cv2.imshow("Face detection - press any key to close", result)
                cv2.waitKey(0)
        else:
            capture = cv2.VideoCapture(args.camera)
            if not capture.isOpened():
                raise RuntimeError(f"Could not open camera {args.camera}")
            print("Press Q or Esc to quit.")
            while True:
                success, frame = capture.read()
                if not success:
                    raise RuntimeError("Could not read a frame from the camera")
                result, _ = annotate_faces(frame, detector)
                cv2.imshow("Face detection", result)
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                    break
                if cv2.getWindowProperty("Face detection", cv2.WND_PROP_VISIBLE) < 1:
                    break
        return 0
    except (RuntimeError, cv2.error) as error:
        print(f"Error: {error}")
        return 1
    finally:
        if capture is not None:
            capture.release()
        if not args.no_display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())

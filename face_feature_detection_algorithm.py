import cv2
import mediapipe as mp

# Initialize Mediapipe FaceMesh and Drawing utilities
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Create a video capture object for the webcam
cap = cv2.VideoCapture(0)

# Define feature landmark indices (from MediaPipe 468 landmark map)
FEATURE_POINTS = {
    "Nose": 1,
    "Left Eye": 33,
    "Right Eye": 263,
    "Left Lip": 61,
    "Right Lip": 291,
    "Chin": 152,
    "Left Eyebrow": 70,
    "Right Eyebrow": 300
}

# Start Face Mesh detection
with mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Camera not found.")
            break

        # Convert the BGR image to RGB for Mediapipe processing
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)

        # Convert RGB back to BGR for OpenCV display
        image.flags.writeable = True

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Draw all mesh landmarks
                mp_drawing.draw_landmarks(
                    image,
                    face_landmarks,
                    mp_face_mesh.FACEMESH_TESSELATION,
                    mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=1),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1)
                )

                # Label major facial features
                h, w, _ = image.shape
                for feature, index in FEATURE_POINTS.items():
                    landmark = face_landmarks.landmark[index]
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(image, (x, y), 3, (0, 255, 0), -1)
                    cv2.putText(image, feature, (x + 5, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Display output
        cv2.imshow("Face Feature Labeling - MediaPipe", image)

        # Exit on pressing 'q'
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

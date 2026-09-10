import cv2
import mediapipe as mp
import numpy as np
import math
import os
from collections import deque


# =========================================================
# CONFIG
# =========================================================

CAMERA_INDEX = 0

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

CAMERA_PANEL_WIDTH = 820
MEME_PANEL_WIDTH = 420

STABLE_FRAMES = 5

ASSET_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "assets",
    "memes"
)


# =========================================================
# MEDIAPIPE
# =========================================================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


# =========================================================
# UTILITY
# =========================================================

def distance(a, b):
    """
    Menghitung jarak antara dua landmark.
    """
    return math.sqrt(
        ((a.x - b.x) ** 2) +
        ((a.y - b.y) ** 2)
    )


def calculate_angle(a, b, c):
    """
    Menghitung sudut A-B-C. b adalah titik tengah.
    """
    ba = np.array([
        a.x - b.x,
        a.y - b.y
    ])

    bc = np.array([
        c.x - b.x,
        c.y - b.y
    ])

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if norm_ba == 0 or norm_bc == 0:
        return 0

    cosine_angle = np.dot(ba, bc) / (
        norm_ba * norm_bc
    )

    cosine_angle = np.clip(
        cosine_angle,
        -1.0,
        1.0
    )

    angle = np.arccos(
        cosine_angle
    )

    return np.degrees(angle)


# =========================================================
# DETEKSI JARI
# =========================================================

def finger_open(
    landmarks,
    mcp,
    pip,
    dip,
    tip
):
    """
    Mengecek apakah suatu jari sedang lurus / terbuka.
    """

    angle1 = calculate_angle(
        landmarks[mcp],
        landmarks[pip],
        landmarks[dip]
    )

    angle2 = calculate_angle(
        landmarks[pip],
        landmarks[dip],
        landmarks[tip]
    )

    if (
        angle1 > 150
        and
        angle2 > 145
    ):
        return True

    return False


def thumb_open(landmarks):
    """
    Deteksi ibu jari terbuka.
    """

    angle1 = calculate_angle(
        landmarks[1],
        landmarks[2],
        landmarks[3]
    )

    angle2 = calculate_angle(
        landmarks[2],
        landmarks[3],
        landmarks[4]
    )

    palm_size = distance(
        landmarks[0],
        landmarks[9]
    )

    if palm_size == 0:
        palm_size = 0.0001

    thumb_distance = (
        distance(
            landmarks[4],
            landmarks[5]
        )
        /
        palm_size
    )

    if (
        angle1 > 135
        and
        angle2 > 140
        and
        thumb_distance > 0.55
    ):
        return True

    return False


# =========================================================
# KLASIFIKASI SATU TANGAN
# =========================================================

def classify_hand(landmarks):

    thumb = thumb_open(
        landmarks
    )

    index = finger_open(
        landmarks,
        5,
        6,
        7,
        8
    )

    middle = finger_open(
        landmarks,
        9,
        10,
        11,
        12
    )

    ring = finger_open(
        landmarks,
        13,
        14,
        15,
        16
    )

    pinky = finger_open(
        landmarks,
        17,
        18,
        19,
        20
    )


    # =====================================================
    # OK
    # =====================================================

    palm_size = distance(
        landmarks[0],
        landmarks[9]
    )

    if palm_size == 0:
        palm_size = 0.0001

    thumb_index_distance = (
        distance(
            landmarks[4],
            landmarks[8]
        )
        /
        palm_size
    )

    if (
        thumb_index_distance < 0.40
        and
        middle
        and
        ring
        and
        pinky
    ):
        return "OK"


    # =====================================================
    # SHAKA / CALL ME 🤙
    # =====================================================

    if (
        thumb
        and
        pinky
        and
        not index
        and
        not middle
        and
        not ring
    ):
        return "SHAKA"


    # =====================================================
    # THUMBS UP 👍
    # =====================================================

    if (
        thumb
        and
        not index
        and
        not middle
        and
        not ring
        and
        not pinky
    ):

        # ujung jempol harus lebih tinggi
        # dibanding pangkal jempol

        if (
            landmarks[4].y
            <
            landmarks[2].y
        ):
            return "THUMBS_UP"

        return "THUMB"


    # =====================================================
    # PEACE ✌
    # =====================================================

    if (
        index
        and
        middle
        and
        not ring
        and
        not pinky
    ):
        return "PEACE"


    # =====================================================
    # SATU TELUNJUK
    # =====================================================

    if (
        index
        and
        not middle
        and
        not ring
        and
        not pinky
    ):
        return "INDEX"


    # =====================================================
    # OPEN PALM
    # =====================================================

    if (
        index
        and
        middle
        and
        ring
        and
        pinky
    ):
        return "OPEN_PALM"


    # =====================================================
    # FIST
    # =====================================================

    if (
        not thumb
        and
        not index
        and
        not middle
        and
        not ring
        and
        not pinky
    ):
        return "FIST"


    return "UNKNOWN"


# =========================================================
# DETEKSI GESTURE 1 / 2 TANGAN
# =========================================================

def classify_scene(hand_landmarks_list):

    if len(hand_landmarks_list) == 0:
        return "NEUTRAL"


    gestures = []

    for hand_landmarks in hand_landmarks_list:

        gesture = classify_hand(
            hand_landmarks.landmark
        )

        gestures.append(
            gesture
        )


    # =====================================================
    # GESTURE 2 TANGAN 👉👈
    # =====================================================

    if len(hand_landmarks_list) >= 2:

        if (
            gestures[0] == "INDEX"
            and
            gestures[1] == "INDEX"
        ):

            index1 = (
                hand_landmarks_list[0]
                .landmark[8]
            )

            index2 = (
                hand_landmarks_list[1]
                .landmark[8]
            )

            index_distance = distance(
                index1,
                index2
            )

            if index_distance < 0.45:
                return "SHY"


    # =====================================================
    # PRIORITAS GESTURE
    # =====================================================

    priority = [

        "SHAKA",

        "OK",

        "THUMBS_UP",

        "PEACE",

        "INDEX",

        "OPEN_PALM",

        "FIST",

        "THUMB"
    ]


    for item in priority:

        if item in gestures:

            return item


    return "UNKNOWN"


# =========================================================
# STABILISASI GESTURE
# =========================================================

class GestureStabilizer:

    def __init__(
        self,
        stable_frames=5
    ):

        self.stable_frames = (
            stable_frames
        )

        self.history = deque(
            maxlen=15
        )

        self.current = "NEUTRAL"


    def update(
        self,
        gesture
    ):

        self.history.append(
            gesture
        )


        if (
            len(self.history)
            <
            self.stable_frames
        ):

            return self.current


        recent = list(
            self.history
        )[
            -self.stable_frames:
        ]


        if (
            len(set(recent))
            ==
            1
        ):

            self.current = (
                recent[0]
            )


        return self.current


# =========================================================
# FILE GAMBAR MEME
# =========================================================

MEME_FILES = {

    "NEUTRAL":
        "neutral.png",

    "INDEX":
        "index.jpg",

    "PEACE":
        "peace.jpg",

    "THUMBS_UP":
        "thumbs_up.jpg",

    "SHAKA":
        "shaka.png",

    "OK":
        "ok.jpg",

    "FIST":
        "fist.jpg",

    "OPEN_PALM":
        "open_palm.jpg",
        # "open_palm.gif",

    "SHY":
        "shy.jpg"
}


# =========================================================
# LOAD MEME
# =========================================================

def load_meme(
    gesture,
    width,
    height
):

    filename = MEME_FILES.get(
        gesture
    )


    if filename:

        full_path = os.path.join(
            ASSET_DIR,
            filename
        )

        if os.path.exists(
            full_path
        ):

            image = cv2.imread(
                full_path
            )

            if image is not None:

                return fit_image(
                    image,
                    width,
                    height
                )


    # =====================================================
    # FALLBACK
    # kalau gambar tidak ditemukan
    # =====================================================

    panel = np.zeros(
        (
            height,
            width,
            3
        ),
        dtype=np.uint8
    )


    titles = {

        "NEUTRAL":
            "READY",

        "INDEX":
            "ONE FINGER",

        "PEACE":
            "PEACE",

        "THUMBS_UP":
            "NICE!",

        "SHAKA":
            "CALL ME",

        "OK":
            "OK!",

        "FIST":
            "FIST",

        "OPEN_PALM":
            "HELLO!",

        "SHY":
            "SHY",

        "UNKNOWN":
            "..."
    }


    text = titles.get(
        gesture,
        gesture
    )


    font = (
        cv2.FONT_HERSHEY_SIMPLEX
    )


    scale = 1.5

    thickness = 3


    text_size = (
        cv2.getTextSize(
            text,
            font,
            scale,
            thickness
        )[0]
    )


    text_x = (
        width
        -
        text_size[0]
    ) // 2


    text_y = (
        height
        +
        text_size[1]
    ) // 2


    cv2.putText(

        panel,

        text,

        (
            text_x,
            text_y
        ),

        font,

        scale,

        (
            255,
            255,
            255
        ),

        thickness,

        cv2.LINE_AA
    )


    return panel


# =========================================================
# FIT IMAGE
# =========================================================

def fit_image(
    image,
    width,
    height
):

    image_height, image_width = (
        image.shape[:2]
    )


    scale = min(

        width
        /
        image_width,

        height
        /
        image_height
    )


    new_width = int(
        image_width
        *
        scale
    )

    new_height = int(
        image_height
        *
        scale
    )


    resized = cv2.resize(

        image,

        (
            new_width,
            new_height
        ),

        interpolation=
        cv2.INTER_AREA
    )


    canvas = np.zeros(

        (
            height,
            width,
            3
        ),

        dtype=
        np.uint8
    )


    x = (
        width
        -
        new_width
    ) // 2


    y = (
        height
        -
        new_height
    ) // 2


    canvas[
        y:y + new_height,
        x:x + new_width
    ] = resized


    return canvas


# =========================================================
# LABEL
# =========================================================

def draw_label(
    frame,
    text,
    x,
    y
):

    font = (
        cv2.FONT_HERSHEY_SIMPLEX
    )


    font_scale = 0.65

    thickness = 2


    text_size = (
        cv2.getTextSize(
            text,
            font,
            font_scale,
            thickness
        )[0]
    )


    cv2.rectangle(

        frame,

        (
            x,
            y - 30
        ),

        (
            x
            +
            text_size[0]
            +
            18,

            y + 8
        ),

        (
            20,
            20,
            20
        ),

        -1
    )


    cv2.putText(

        frame,

        text,

        (
            x + 9,
            y
        ),

        font,

        font_scale,

        (
            255,
            255,
            255
        ),

        thickness,

        cv2.LINE_AA
    )


# =========================================================
# MAIN PROGRAM
# =========================================================

def main():

    # =====================================================
    # CAMERA
    # =====================================================

    cap = cv2.VideoCapture(
        CAMERA_INDEX
    )


    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        1280
    )


    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        720
    )


    if not cap.isOpened():

        print(
            "ERROR: Kamera tidak bisa dibuka."
        )

        return


    # =====================================================
    # STABILIZER
    # =====================================================

    stabilizer = (
        GestureStabilizer(
            STABLE_FRAMES
        )
    )


    # =====================================================
    # WARNA LANDMARK
    # =====================================================

    landmark_style = (

        mp_draw.DrawingSpec(

            color=(
                0,
                255,
                0
            ),

            thickness=2,

            circle_radius=3
        )
    )


    connection_style = (

        mp_draw.DrawingSpec(

            color=(
                0,
                220,
                0
            ),

            thickness=2,

            circle_radius=2
        )
    )


    # =====================================================
    # MEDIAPIPE HANDS
    # =====================================================

    with mp_hands.Hands(

        static_image_mode=False,

        max_num_hands=2,

        model_complexity=1,

        min_detection_confidence=0.65,

        min_tracking_confidence=0.60

    ) as hands:


        while True:

            success, frame = (
                cap.read()
            )


            if not success:

                print(
                    "Frame kamera gagal dibaca."
                )

                break


            # =================================================
            # MIRROR CAMERA
            # =================================================

            frame = cv2.flip(
                frame,
                1
            )


            # =================================================
            # BGR -> RGB
            # =================================================

            rgb = cv2.cvtColor(

                frame,

                cv2.COLOR_BGR2RGB
            )


            # =================================================
            # DETEKSI TANGAN
            # =================================================

            result = hands.process(
                rgb
            )


            if (
                result.multi_hand_landmarks
            ):

                detected_hands = (
                    result.multi_hand_landmarks
                )

            else:

                detected_hands = []


            # =================================================
            # DETEKSI GESTURE
            # =================================================

            raw_gesture = classify_scene(
                detected_hands
            )


            stable_gesture = (
                stabilizer.update(
                    raw_gesture
                )
            )


            # =================================================
            # DRAW LANDMARK
            # =================================================

            for hand_landmarks in detected_hands:

                mp_draw.draw_landmarks(

                    frame,

                    hand_landmarks,

                    mp_hands.HAND_CONNECTIONS,

                    landmark_style,

                    connection_style
                )


            # =================================================
            # BUAT CANVAS UTAMA
            # =================================================

            canvas = np.zeros(

                (
                    WINDOW_HEIGHT,
                    WINDOW_WIDTH,
                    3
                ),

                dtype=
                np.uint8
            )


            canvas[:] = (
                18,
                18,
                18
            )


            # =================================================
            # CAMERA VIEW
            # =================================================

            camera_view = fit_image(

                frame,

                780,

                570
            )


            canvas[
                80:650,
                25:805
            ] = camera_view


            # =================================================
            # MEME VIEW
            # =================================================

            meme = load_meme(

                stable_gesture,

                380,

                500
            )


            canvas[
                100:600,
                870:1250
            ] = meme


            # =================================================
            # PANEL BORDER
            # =================================================

            cv2.rectangle(

                canvas,

                (
                    15,
                    15
                ),

                (
                    820,
                    700
                ),

                (
                    70,
                    70,
                    70
                ),

                2
            )


            cv2.rectangle(

                canvas,

                (
                    850,
                    15
                ),

                (
                    1265,
                    700
                ),

                (
                    70,
                    70,
                    70
                ),

                2
            )


            # =================================================
            # TITLE
            # =================================================

            cv2.putText(

                canvas,

                "CAMERA",

                (
                    35,
                    55
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.8,

                (
                    230,
                    230,
                    230
                ),

                2,

                cv2.LINE_AA
            )


            cv2.putText(

                canvas,

                "EL GATO REACTION",

                (
                    875,
                    55
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.8,

                (
                    230,
                    230,
                    230
                ),

                2,

                cv2.LINE_AA
            )


            # =================================================
            # GESTURE LABEL
            # =================================================

            draw_label(

                canvas,

                "Gesture: "
                +
                stable_gesture,

                875,

                640
            )


            draw_label(

                canvas,

                "Raw: "
                +
                raw_gesture,

                875,

                680
            )


            # =================================================
            # JUMLAH TANGAN
            # =================================================

            cv2.putText(

                canvas,

                "Hands detected: "
                +
                str(
                    len(
                        detected_hands
                    )
                ),

                (
                    35,
                    680
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.55,

                (
                    180,
                    180,
                    180
                ),

                1,

                cv2.LINE_AA
            )


            # =================================================
            # DISPLAY
            # =================================================

            cv2.imshow(

                "Hand Gesture Meme",

                canvas
            )


            # =================================================
            # KEYBOARD
            # =================================================

            key = (
                cv2.waitKey(1)
                &
                0xFF
            )


            if (
                key
                ==
                ord("q")
            ):

                break


            if key == 27:

                break


    # =====================================================
    # CLEANUP
    # =====================================================

    cap.release()

    cv2.destroyAllWindows()


# =========================================================
# START
# =========================================================

if __name__ == "__main__":

    main()
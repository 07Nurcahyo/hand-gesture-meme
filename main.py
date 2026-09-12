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

HAND_STABLE_FRAMES = 5
FACE_STABLE_FRAMES = 5

SHOW_FACE_MESH = False


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


MEME_DIR = os.path.join(
    BASE_DIR,
    "assets",
    "memes"
)


FACE_OVERLAY_DIR = os.path.join(
    BASE_DIR,
    "assets",
    "face_overlays"
)


# =========================================================
# MEDIAPIPE
# =========================================================

mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


# =========================================================
# HAND MEME
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

    "SHY":
        "shy.jpg"
}


# =========================================================
# FACE OVERLAY
# =========================================================

FACE_OVERLAY_FILES = {

    "MOUTH_OPEN":
        "mouth_open.png",

    "SMILE":
        "smile.png",

    "SQUINT":
        "squint.jpg",

    "BROW_RAISE":
        "brow_raise.jpg",

    "HEAD_LEFT":
        "head_left.jpg",

    "HEAD_RIGHT":
        "head_right.png"
}


# =========================================================
# BASIC UTILITY
# =========================================================

def distance(a, b):

    return math.sqrt(

        ((a.x - b.x) ** 2)

        +

        ((a.y - b.y) ** 2)

    )


def calculate_angle(a, b, c):

    ba = np.array(

        [
            a.x - b.x,
            a.y - b.y
        ],

        dtype=np.float32

    )


    bc = np.array(

        [
            c.x - b.x,
            c.y - b.y
        ],

        dtype=np.float32

    )


    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)


    if (
        norm_ba < 0.000001
        or
        norm_bc < 0.000001
    ):

        return 0.0


    cosine_angle = (

        np.dot(
            ba,
            bc
        )

        /

        (
            norm_ba
            *
            norm_bc
        )

    )


    cosine_angle = np.clip(

        cosine_angle,

        -1.0,

        1.0

    )


    angle = np.arccos(
        cosine_angle
    )


    return float(

        np.degrees(
            angle
        )

    )


# =========================================================
# HAND DETECTION
# =========================================================

def finger_open(
    landmarks,
    mcp,
    pip,
    dip,
    tip
):

    angle_1 = calculate_angle(

        landmarks[mcp],

        landmarks[pip],

        landmarks[dip]

    )


    angle_2 = calculate_angle(

        landmarks[pip],

        landmarks[dip],

        landmarks[tip]

    )


    return (

        angle_1 > 150

        and

        angle_2 > 145

    )


def thumb_open(
    landmarks
):

    angle_1 = calculate_angle(

        landmarks[1],

        landmarks[2],

        landmarks[3]

    )


    angle_2 = calculate_angle(

        landmarks[2],

        landmarks[3],

        landmarks[4]

    )


    palm_size = max(

        distance(

            landmarks[0],

            landmarks[9]

        ),

        0.0001

    )


    thumb_distance = (

        distance(

            landmarks[4],

            landmarks[5]

        )

        /

        palm_size

    )


    return (

        angle_1 > 135

        and

        angle_2 > 140

        and

        thumb_distance > 0.55

    )


def classify_hand(
    landmarks
):

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

    palm_size = max(

        distance(

            landmarks[0],

            landmarks[9]

        ),

        0.0001

    )


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
    # SHAKA
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
    # THUMBS UP
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

        if (

            landmarks[4].y

            <

            landmarks[2].y

        ):

            return "THUMBS_UP"


        return "THUMB"


    # =====================================================
    # PEACE
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
    # INDEX
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


def classify_hand_scene(
    hand_landmarks_list
):

    if not hand_landmarks_list:

        return "NEUTRAL"


    gestures = []


    for hand_landmarks in hand_landmarks_list:

        gestures.append(

            classify_hand(

                hand_landmarks.landmark

            )

        )


    # =====================================================
    # SHY / TWO INDEX FINGERS
    # =====================================================

    if len(hand_landmarks_list) >= 2:

        if (

            gestures[0] == "INDEX"

            and

            gestures[1] == "INDEX"

        ):

            index_1 = (

                hand_landmarks_list[0]
                .landmark[8]

            )


            index_2 = (

                hand_landmarks_list[1]
                .landmark[8]

            )


            if (

                distance(

                    index_1,

                    index_2

                )

                <

                0.45

            ):

                return "SHY"


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


    for gesture in priority:

        if gesture in gestures:

            return gesture


    return "UNKNOWN"


# =========================================================
# FACE METRICS
# =========================================================

def calculate_face_metrics(
    landmarks
):

    face_width = max(

        distance(

            landmarks[234],

            landmarks[454]

        ),

        0.0001

    )


    face_height = max(

        distance(

            landmarks[10],

            landmarks[152]

        ),

        0.0001

    )


    # =====================================================
    # MOUTH
    # =====================================================

    mouth_width = distance(

        landmarks[61],

        landmarks[291]

    )


    mouth_open = distance(

        landmarks[13],

        landmarks[14]

    )


    mouth_open_ratio = (

        mouth_open

        /

        max(

            mouth_width,

            0.0001

        )

    )


    mouth_width_ratio = (

        mouth_width

        /

        face_width

    )


    # =====================================================
    # LEFT EYE
    # =====================================================

    left_eye_width = max(

        distance(

            landmarks[33],

            landmarks[133]

        ),

        0.0001

    )


    left_eye_height = distance(

        landmarks[159],

        landmarks[145]

    )


    left_eye_ratio = (

        left_eye_height

        /

        left_eye_width

    )


    # =====================================================
    # RIGHT EYE
    # =====================================================

    right_eye_width = max(

        distance(

            landmarks[362],

            landmarks[263]

        ),

        0.0001

    )


    right_eye_height = distance(

        landmarks[386],

        landmarks[374]

    )


    right_eye_ratio = (

        right_eye_height

        /

        right_eye_width

    )


    eye_open_ratio = (

        left_eye_ratio

        +

        right_eye_ratio

    ) / 2


    # =====================================================
    # BROW
    # =====================================================

    left_brow_eye = distance(

        landmarks[105],

        landmarks[159]

    )


    right_brow_eye = distance(

        landmarks[334],

        landmarks[386]

    )


    brow_ratio = (

        (

            left_brow_eye

            +

            right_brow_eye

        )

        /

        2

        /

        face_height

    )


    # =====================================================
    # HEAD TURN
    # =====================================================

    left_cheek_x = (
        landmarks[234].x
    )


    right_cheek_x = (
        landmarks[454].x
    )


    nose_x = (
        landmarks[1].x
    )


    min_cheek = min(

        left_cheek_x,

        right_cheek_x

    )


    max_cheek = max(

        left_cheek_x,

        right_cheek_x

    )


    cheek_span = max(

        max_cheek
        -
        min_cheek,

        0.0001

    )


    nose_position = (

        nose_x
        -
        min_cheek

    ) / cheek_span


    # =====================================================
    # MOUTH CORNER
    # =====================================================

    mouth_corner_y = (

        landmarks[61].y

        +

        landmarks[291].y

    ) / 2


    mouth_center_y = (

        landmarks[13].y

        +

        landmarks[14].y

    ) / 2


    corner_lift = (

        mouth_center_y

        -

        mouth_corner_y

    ) / face_height


    return {

        "mouth_open":
            mouth_open_ratio,

        "mouth_width":
            mouth_width_ratio,

        "eye_open":
            eye_open_ratio,

        "brow":
            brow_ratio,

        "nose_position":
            nose_position,

        "corner_lift":
            corner_lift

    }


# =========================================================
# FACE CLASSIFICATION
# =========================================================

def classify_face_action(
    metrics
):

    mouth_open = (
        metrics["mouth_open"]
    )


    mouth_width = (
        metrics["mouth_width"]
    )


    eye_open = (
        metrics["eye_open"]
    )


    brow = (
        metrics["brow"]
    )


    nose_position = (
        metrics["nose_position"]
    )


    corner_lift = (
        metrics["corner_lift"]
    )


    # =====================================================
    # HEAD TURN
    # =====================================================

    if nose_position < 0.40:

        return "HEAD_LEFT"


    if nose_position > 0.60:

        return "HEAD_RIGHT"


    # =====================================================
    # MOUTH OPEN
    # =====================================================

    if mouth_open > 0.30:

        return "MOUTH_OPEN"


    # =====================================================
    # SQUINT
    # =====================================================

    if eye_open < 0.18:

        return "SQUINT"


    # =====================================================
    # BROW RAISE
    # =====================================================

    if brow > 0.090:

        return "BROW_RAISE"


    # =====================================================
    # SMILE
    # =====================================================

    if (

        mouth_width > 0.40

        and

        corner_lift > 0.008

        and

        mouth_open < 0.28

    ):

        return "SMILE"


    return "FACE_NEUTRAL"


# =========================================================
# FACE BOUNDING BOX
# =========================================================

def face_bbox_pixels(
    face_landmarks,
    frame_width,
    frame_height
):

    xs = [

        lm.x

        for lm
        in face_landmarks.landmark

    ]


    ys = [

        lm.y

        for lm
        in face_landmarks.landmark

    ]


    x1 = int(
        min(xs)
        *
        frame_width
    )


    y1 = int(
        min(ys)
        *
        frame_height
    )


    x2 = int(
        max(xs)
        *
        frame_width
    )


    y2 = int(
        max(ys)
        *
        frame_height
    )


    pad_x = int(
        (x2 - x1)
        *
        0.08
    )


    pad_y = int(
        (y2 - y1)
        *
        0.08
    )


    x1 = max(
        0,
        x1 - pad_x
    )


    y1 = max(
        0,
        y1 - pad_y
    )


    x2 = min(
        frame_width - 1,
        x2 + pad_x
    )


    y2 = min(
        frame_height - 1,
        y2 + pad_y
    )


    return (
        x1,
        y1,
        x2,
        y2
    )


# =========================================================
# DRAW FACE BOX
# =========================================================

def draw_face_box(
    frame,
    face_landmarks,
    face_state
):

    height, width = (
        frame.shape[:2]
    )


    x1, y1, x2, y2 = (

        face_bbox_pixels(

            face_landmarks,

            width,

            height

        )

    )


    cv2.rectangle(

        frame,

        (
            x1,
            y1
        ),

        (
            x2,
            y2
        ),

        (
            0,
            255,
            0
        ),

        2

    )


    # garis dekorasi corner
    corner_length = 20


    cv2.line(

        frame,

        (x1, y1),

        (
            x1 + corner_length,
            y1
        ),

        (
            0,
            255,
            0
        ),

        4

    )


    cv2.line(

        frame,

        (x1, y1),

        (
            x1,
            y1 + corner_length
        ),

        (
            0,
            255,
            0
        ),

        4

    )


    cv2.line(

        frame,

        (x2, y1),

        (
            x2 - corner_length,
            y1
        ),

        (
            0,
            255,
            0
        ),

        4

    )


    cv2.line(

        frame,

        (x2, y1),

        (
            x2,
            y1 + corner_length
        ),

        (
            0,
            255,
            0
        ),

        4

    )


    cv2.line(

        frame,

        (x1, y2),

        (
            x1 + corner_length,
            y2
        ),

        (
            0,
            255,
            0
        ),

        4

    )


    cv2.line(

        frame,

        (x1, y2),

        (
            x1,
            y2 - corner_length
        ),

        (
            0,
            255,
            0
        ),

        4

    )


    cv2.line(

        frame,

        (x2, y2),

        (
            x2 - corner_length,
            y2
        ),

        (
            0,
            255,
            0
        ),

        4

    )


    cv2.line(

        frame,

        (x2, y2),

        (
            x2,
            y2 - corner_length
        ),

        (
            0,
            255,
            0
        ),

        4

    )


# =========================================================
# LOAD FACE OVERLAY
# =========================================================

def load_face_overlay_png(
    face_state
):

    filename = (
        FACE_OVERLAY_FILES.get(
            face_state
        )
    )


    if not filename:

        return None


    full_path = os.path.join(

        FACE_OVERLAY_DIR,

        filename

    )


    if not os.path.exists(
        full_path
    ):

        return None


    return cv2.imread(

        full_path,

        cv2.IMREAD_UNCHANGED

    )


# =========================================================
# PNG TRANSPARENT OVERLAY
# =========================================================

def overlay_png(
    background,
    overlay,
    x,
    y,
    width=None,
    height=None
):

    if overlay is None:

        return background


    if width is not None and height is not None:

        overlay = cv2.resize(

            overlay,

            (
                width,
                height
            ),

            interpolation=
                cv2.INTER_AREA

        )


    if len(
        overlay.shape
    ) < 3:

        return background


    overlay_height, overlay_width = (
        overlay.shape[:2]
    )


    background_height, background_width = (
        background.shape[:2]
    )


    # =====================================================
    # POSISI OUTSIDE FRAME
    # =====================================================

    if (
        x >= background_width
        or
        y >= background_height
    ):

        return background


    if (
        x + overlay_width <= 0
        or
        y + overlay_height <= 0
    ):

        return background


    # =====================================================
    # CROP AREA
    # =====================================================

    x1 = max(
        0,
        x
    )


    y1 = max(
        0,
        y
    )


    x2 = min(

        background_width,

        x + overlay_width

    )


    y2 = min(

        background_height,

        y + overlay_height

    )


    overlay_x1 = max(

        0,

        -x

    )


    overlay_y1 = max(

        0,

        -y

    )


    overlay_x2 = (

        overlay_x1

        +

        (x2 - x1)

    )


    overlay_y2 = (

        overlay_y1

        +

        (y2 - y1)

    )


    if (

        x1 >= x2

        or

        y1 >= y2

    ):

        return background


    overlay_crop = overlay[

        overlay_y1:overlay_y2,

        overlay_x1:overlay_x2

    ]


    # =====================================================
    # PNG TANPA ALPHA
    # =====================================================

    if overlay_crop.shape[2] < 4:

        background[

            y1:y2,

            x1:x2

        ] = overlay_crop[:, :, :3]


        return background


    # =====================================================
    # ALPHA BLENDING
    # =====================================================

    alpha = (

        overlay_crop[:, :, 3]

        .astype(
            np.float32
        )

        /

        255.0

    )


    alpha = alpha[:, :, np.newaxis]


    foreground = (

        overlay_crop[:, :, :3]

        .astype(
            np.float32
        )

    )


    background_crop = (

        background[

            y1:y2,

            x1:x2

        ]

        .astype(
            np.float32
        )

    )


    blended = (

        alpha
        *
        foreground

        +

        (
            1.0
            -
            alpha
        )

        *
        background_crop

    )


    background[

        y1:y2,

        x1:x2

    ] = blended.astype(
        np.uint8
    )


    return background


# =========================================================
# FACE REACTION OVERLAY
# =========================================================

def draw_face_reaction_on_box(
    frame,
    face_landmarks,
    face_state
):

    if face_state == "FACE_NEUTRAL":

        return


    frame_height, frame_width = (
        frame.shape[:2]
    )


    x1, y1, x2, y2 = (

        face_bbox_pixels(

            face_landmarks,

            frame_width,

            frame_height

        )

    )


    face_width = (
        x2 - x1
    )


    face_height = (
        y2 - y1
    )


    if (
        face_width <= 0
        or
        face_height <= 0
    ):

        return


    overlay = (
        load_face_overlay_png(
            face_state
        )
    )


    # =====================================================
    # UKURAN REACTION
    #
    # 92% lebar wajah
    # 32% tinggi wajah
    # =====================================================

    # overlay_width = max(
    #     120,
    #     int(face_width * 0.92)
    # )

    # overlay_height = max(
    #     45,
    #     int(face_height * 0.30)
    # )
    overlay_width = face_width
    overlay_height = face_height


    # =====================================================
    # POSISI:
    # menempel pada bagian atas kotak hijau
    # =====================================================

    # overlay_x = (x1 + int((face_width - overlay_width) / 2))
    # overlay_y = (y1 - overlay_height - 6)
    overlay_x = x1
    overlay_y = y1

    # kalau terlalu dekat atas layar,
    # masukkan ke bagian atas wajah

    if overlay_y < 5:

        overlay_y = (
            y1 + 8
        )


    # =====================================================
    # ADA PNG
    # =====================================================

    if overlay is not None:

        overlay_png(

            frame,

            overlay,

            overlay_x,

            overlay_y,

            overlay_width,

            overlay_height

        )


    # =====================================================
    # FALLBACK TEXT
    # =====================================================

    else:

        label = face_state.replace(

            "_",

            " "

        )


        # background transparan palsu
        temp = frame.copy()


        cv2.rectangle(

            temp,

            (
                overlay_x,
                overlay_y
            ),

            (
                overlay_x
                +
                overlay_width,

                overlay_y
                +
                overlay_height
            ),

            (
                0,
                0,
                0
            ),

            -1

        )


        frame[:] = cv2.addWeighted(

            temp,

            0.70,

            frame,

            0.30,

            0

        )


        cv2.rectangle(

            frame,

            (
                overlay_x,
                overlay_y
            ),

            (
                overlay_x
                +
                overlay_width,

                overlay_y
                +
                overlay_height
            ),

            (
                0,
                255,
                0
            ),

            2

        )


        font_scale = max(

            0.45,

            min(

                0.75,

                face_width
                /
                350

            )

        )


        cv2.putText(

            frame,

            label,

            (
                overlay_x + 8,

                overlay_y
                +
                int(
                    overlay_height
                    *
                    0.65
                )

            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            font_scale,

            (
                0,
                255,
                0
            ),

            2,

            cv2.LINE_AA

        )


# =========================================================
# OPTIONAL FACE MESH
# =========================================================

def draw_face_mesh(
    frame,
    face_landmarks
):

    mp_draw.draw_landmarks(

        image=frame,

        landmark_list=
            face_landmarks,

        connections=
            mp_face_mesh
            .FACEMESH_CONTOURS,

        landmark_drawing_spec=None,

        connection_drawing_spec=
            mp_drawing_styles
            .get_default_face_mesh_contours_style()

    )


# =========================================================
# STATE STABILIZER
# =========================================================

class StateStabilizer:

    def __init__(
        self,
        stable_frames=5,
        default_state="NEUTRAL"
    ):

        self.stable_frames = (
            stable_frames
        )


        self.history = deque(

            maxlen=max(

                stable_frames * 3,

                15

            )

        )


        self.current = (
            default_state
        )


    def update(
        self,
        state
    ):

        self.history.append(
            state
        )


        if (

            len(
                self.history
            )

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
            len(
                set(
                    recent
                )
            )
            ==
            1
        ):

            self.current = (
                recent[0]
            )


        return self.current


# =========================================================
# IMAGE FIT
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


    new_width = max(

        1,

        int(
            image_width
            *
            scale
        )

    )


    new_height = max(

        1,

        int(
            image_height
            *
            scale
        )

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

        dtype=np.uint8

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
# LOAD HAND MEME
# =========================================================

def load_hand_meme(
    state,
    width,
    height
):

    filename = MEME_FILES.get(
        state
    )


    if filename:

        full_path = os.path.join(

            MEME_DIR,

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
    # =====================================================

    panel = np.zeros(

        (
            height,
            width,
            3
        ),

        dtype=np.uint8

    )


    label = state.replace(

        "_",

        " "

    )


    text_size = cv2.getTextSize(

        label,

        cv2.FONT_HERSHEY_SIMPLEX,

        0.9,

        2

    )[0]


    text_x = max(

        10,

        (

            width
            -
            text_size[0]

        ) // 2

    )


    text_y = (
        height // 2
    )


    cv2.putText(

        panel,

        label,

        (
            text_x,
            text_y
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.9,

        (
            255,
            255,
            255
        ),

        2,

        cv2.LINE_AA

    )


    return panel


# =========================================================
# PANEL KANAN KHUSUS HAND
# =========================================================

def choose_hand_reaction(
    hand_state
):

    if hand_state in (

        "NEUTRAL",

        "UNKNOWN",

        "THUMB"

    ):

        return "NEUTRAL"


    return hand_state


# =========================================================
# HUD
# =========================================================

def draw_camera_hud(
    frame,
    raw_hand,
    hand_state,
    raw_face,
    face_state,
    metrics,
    hand_count,
    face_count
):

    lines = [

        (
            f"HANDS : {hand_count}   "
            f"FACES : {face_count}"
        ),

        (
            f"HAND  : {hand_state}"
        ),

        (
            f"FACE  : {face_state}"
        )

    ]


    if metrics:

        lines.extend(

            [

                (
                    "MOUTH "
                    f"{metrics['mouth_open']:.2f}"
                ),

                (
                    "EYE   "
                    f"{metrics['eye_open']:.2f}"
                ),

                (
                    "BROW  "
                    f"{metrics['brow']:.3f}"
                ),

                (
                    "TURN  "
                    f"{metrics['nose_position']:.2f}"
                )

            ]

        )


    y = 25


    for line in lines:

        cv2.putText(

            frame,

            line,

            (
                10,
                y
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.45,

            (
                0,
                255,
                0
            ),

            1,

            cv2.LINE_AA

        )


        y += 21


# =========================================================
# MAIN
# =========================================================

def main():

    global SHOW_FACE_MESH


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
            "ERROR: Kamera tidak dapat dibuka."
        )

        print(
            "Coba ubah CAMERA_INDEX = 1."
        )

        return


    # =====================================================
    # STABILIZERS
    # =====================================================

    hand_stabilizer = StateStabilizer(

        HAND_STABLE_FRAMES,

        "NEUTRAL"

    )


    face_stabilizer = StateStabilizer(

        FACE_STABLE_FRAMES,

        "FACE_NEUTRAL"

    )


    # =====================================================
    # HAND DRAW STYLE
    # =====================================================

    landmark_style = mp_draw.DrawingSpec(

        color=(
            0,
            255,
            0
        ),

        thickness=2,

        circle_radius=3

    )


    connection_style = mp_draw.DrawingSpec(

        color=(
            0,
            210,
            0
        ),

        thickness=2,

        circle_radius=2

    )


    # =====================================================
    # START MODELS
    # =====================================================

    with mp_hands.Hands(

        static_image_mode=False,

        max_num_hands=2,

        model_complexity=1,

        min_detection_confidence=0.65,

        min_tracking_confidence=0.60


    ) as hands, mp_face_mesh.FaceMesh(


        static_image_mode=False,

        max_num_faces=1,

        refine_landmarks=True,

        min_detection_confidence=0.60,

        min_tracking_confidence=0.60


    ) as face_mesh:


        while True:


            # =================================================
            # READ CAMERA
            # =================================================

            success, frame = (
                cap.read()
            )


            if not success:

                print(
                    "Gagal mengambil frame kamera."
                )

                break


            # =================================================
            # MIRROR
            # =================================================

            frame = cv2.flip(

                frame,

                1

            )


            # =================================================
            # RGB
            # =================================================

            rgb = cv2.cvtColor(

                frame,

                cv2.COLOR_BGR2RGB

            )


            # =================================================
            # PROCESS MEDIAPIPE
            # =================================================

            hand_result = hands.process(
                rgb
            )


            face_result = face_mesh.process(
                rgb
            )


            # =================================================
            # HAND RESULT
            # =================================================

            detected_hands = (

                hand_result
                .multi_hand_landmarks

                or []

            )


            raw_hand = classify_hand_scene(

                detected_hands

            )


            hand_state = hand_stabilizer.update(

                raw_hand

            )


            # =================================================
            # DRAW HAND
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
            # FACE RESULT
            # =================================================

            detected_faces = (

                face_result
                .multi_face_landmarks

                or []

            )


            raw_face = "FACE_NEUTRAL"

            face_state = "FACE_NEUTRAL"

            metrics = {}


            if detected_faces:

                face_landmarks = (
                    detected_faces[0]
                )


                metrics = calculate_face_metrics(

                    face_landmarks.landmark

                )


                raw_face = classify_face_action(

                    metrics

                )


                face_state = face_stabilizer.update(

                    raw_face

                )


                # =============================================
                # OPTIONAL FACE MESH
                # =============================================

                if SHOW_FACE_MESH:

                    draw_face_mesh(

                        frame,

                        face_landmarks

                    )


                # =============================================
                # GREEN FACE BOX
                # =============================================

                draw_face_box(

                    frame,

                    face_landmarks,

                    face_state

                )


                # =============================================
                # FACE REACTION
                # LANGSUNG MENEMPEL KE WAJAH
                # =============================================

                draw_face_reaction_on_box(

                    frame,

                    face_landmarks,

                    face_state

                )


            else:

                face_state = face_stabilizer.update(

                    "FACE_NEUTRAL"

                )


            # =================================================
            # DEBUG HUD
            # =================================================

            draw_camera_hud(

                frame,

                raw_hand,

                hand_state,

                raw_face,

                face_state,

                metrics,

                len(
                    detected_hands
                ),

                len(
                    detected_faces
                )

            )


            # =================================================
            # HAND REACTION PANEL
            # =================================================

            hand_reaction_state = (

                choose_hand_reaction(

                    hand_state

                )

            )


            # =================================================
            # UI CANVAS
            # =================================================

            canvas = np.zeros(

                (
                    WINDOW_HEIGHT,
                    WINDOW_WIDTH,
                    3
                ),

                dtype=np.uint8

            )


            canvas[:] = (

                16,
                16,
                16

            )


            # =================================================
            # LEFT CAMERA
            # =================================================

            camera_view = fit_image(

                frame,

                790,

                570

            )


            canvas[

                80:650,

                25:815

            ] = camera_view


            # =================================================
            # RIGHT HAND REACTION
            # =================================================

            reaction = load_hand_meme(

                hand_reaction_state,

                380,

                500

            )


            canvas[

                90:590,

                875:1255

            ] = reaction


            # =================================================
            # BORDERS
            # =================================================

            cv2.rectangle(

                canvas,

                (
                    15,
                    15
                ),

                (
                    825,
                    700
                ),

                (
                    60,
                    60,
                    60
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
                    60,
                    60,
                    60
                ),

                2

            )


            # =================================================
            # TITLES
            # =================================================

            cv2.putText(

                canvas,

                "CAMERA / AI TRACKING",

                (
                    35,
                    52
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.72,

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

                "HAND REACTION",

                (
                    875,
                    52
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.72,

                (
                    230,
                    230,
                    230
                ),

                2,

                cv2.LINE_AA

            )


            # =================================================
            # STATUS RIGHT
            # =================================================

            cv2.putText(

                canvas,

                (
                    "HAND : "
                    +
                    hand_state
                ),

                (
                    875,
                    625
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.55,

                (
                    0,
                    255,
                    0
                ),

                1,

                cv2.LINE_AA

            )


            cv2.putText(

                canvas,

                (
                    "FACE : "
                    +
                    face_state
                ),

                (
                    875,
                    655
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.55,

                (
                    0,
                    255,
                    0
                ),

                1,

                cv2.LINE_AA

            )


            cv2.putText(

                canvas,

                "M = Face Mesh   Q / ESC = Exit",

                (
                    875,
                    685
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.42,

                (
                    160,
                    160,
                    160
                ),

                1,

                cv2.LINE_AA

            )


            # =================================================
            # SHOW WINDOW
            # =================================================

            cv2.imshow(

                "AI Reaction Cam",

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


            if key in (

                ord("q"),

                27

            ):

                break


            if key == ord("m"):

                SHOW_FACE_MESH = (

                    not
                    SHOW_FACE_MESH

                )


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
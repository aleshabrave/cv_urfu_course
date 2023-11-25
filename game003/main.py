from dataclasses import dataclass
from datetime import datetime
import itertools
import cv2
import mediapipe as mp
import random as rnd
from typing import Callable, Optional
from functools import wraps


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


def on_timer(interval: int):
    """Выполнение по таймеру, можно было бы расспаралелить или ассинхронно замутить, но лень и по скорости все норм."""

    def actual_decorator(func):
        start_dttm = datetime.now()

        @wraps(func)
        def wrapper(*args, **kwargs):
            """Обёртка над функцией."""
            nonlocal start_dttm

            if (datetime.now() - start_dttm).total_seconds() >= interval:
                start_dttm = datetime.now()

                func(*args, **kwargs)

        return wrapper

    return actual_decorator


@dataclass
class Figure:
    """Класс сущности фигуры."""

    position: list
    x_sup: int
    y_sup: int
    color: tuple
    liveness_ticks: int = 5
    _move_radius: int = 20
    _goal_center: Optional[tuple[int, int]] = None

    def _can_move(self, x2: int, y2: int, radius: int) -> bool:
        """Проверка на возможность движения."""
        l, r, d, u = False, False, False, False

        for p in self.position:
            if p[0] <= x2:
                l = True
            elif p[0] >= x2:
                r = True

            if p[1] >= y2:
                u = True
            elif p[1] <= y2:
                d = True

        return l and r and u and d

    def get_position_for_lines(self) -> tuple[tuple[int, int]]:
        """Получить координаты отрезков, описывающих фигуру.."""

        return itertools.combinations(self.position, 2)

    def update_position(self, x_step: int, y_step: int) -> None:
        """Обновить позицию фигуры."""

        for idx, vertex in enumerate(self.position):
            self.position[idx] = vertex[0] + x_step, vertex[1] + y_step

    def get_center(self) -> tuple[int, int]:
        """Получить центр."""

        x, y = 0, 0
        for p in self.position:
            x += p[0]
            y += p[1]

        return x // len(self.position), y // len(self.position)

    def get_goal_center(self, radius: int = None) -> tuple[int, int]:
        """Получить корзину для фигуры."""

        if self._goal_center:
            return self._goal_center

        x = rnd.randint(radius, self.x_sup - 2 * radius - 1)
        y = rnd.randint(radius, self.y_sup - 2 * radius - 1)

        self._goal_center = x, y

        return x, y

    def try_delete(
        self, figures: list["Figure"], radius: int, callback: Callable
    ) -> None:
        """Попытаться удалить, если центр фигуры задел центр мешени."""
        x1, y1 = self.get_center()
        x2, y2 = self.get_goal_center()

        distance = int(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)

        if distance <= radius:
            figures.remove(self)

            callback()

    @classmethod
    def get_needed_figure(
        cls, figures: list["Figure"], x: int, y: int
    ) -> Optional["Figure"]:
        """Получить фигуру для перемещения."""

        for figure in figures:
            if figure._can_move(x, y, cls._move_radius):
                return figure


COLORS = list(itertools.combinations((i for i in range(256)), 3))
THICKNESS = 2
DSQUARE = lambda s, dx, dy: [(dx, dy), (dx + s, dy), (dx, dy + s), (dx + s, dy + s)]
DRECTANGLEX = lambda s, dx, dy: [
    (dx, dy),
    (dx + 2 * s, dy),
    (dx, dy + s),
    (dx + 2 * s, dy + s),
]
DRECTANGLEY = lambda s, dx, dy: [
    (dx, dy),
    (dx + s, dy),
    (dx, dy + 2 * s),
    (dx + s, dy + 2 * s),
]
DAMOGUS = lambda s, dx, dy: [
    (dx, dy),
    (dx + 2 * s, dy),
    (dx, dy + s),
    (dx + 2 * s, dy + s),
    (int(0.25 * s) + dx, dy + 2 * s),
    (int(0.75 * s) + dx, dy + 2 * s),
]
DAMOGUSSS = lambda s, dx, dy: [
    (dx, dy),
    (dx + int(0.25 * s), dy + int(0.25 * s)),
    (dx + int(0.5 * s), dy + int(0.25 * s)),
    (dx + int(0.75 * s), dy + +int(0.5 * s)),
    (dx + s, dy + s),
]
FIGURES_TEMPLATES = [DSQUARE, DRECTANGLEX, DRECTANGLEY, DAMOGUS, DAMOGUSSS]
FIGURES = []
GOAL_RADIUS = 32
SCORE = 0
SCORE_FONT_SCALE = 1
HEALTH_POINTS = 3


@on_timer(interval=1)
def delete_by_liveness():
    """Обновляем счётчик жизни и удаляем если счётчик обнулился (=удаляем фигуру с поля)."""
    global HEALTH_POINTS

    for figure in FIGURES:
        figure.liveness_ticks -= 1

        if figure.liveness_ticks <= -1:
            FIGURES.remove(figure)
            HEALTH_POINTS -= 1


@on_timer(interval=3)
def create_new_figure(img_width, img_height):
    """Генерация новой фигуры."""

    if HEALTH_POINTS <= 0:
        return

    size = rnd.randint(60, 90)
    template = FIGURES_TEMPLATES[rnd.randint(0, len(FIGURES_TEMPLATES) - 1)]

    max_dx = abs(img_width - 2 * size)
    max_dy = abs(img_height - 2 * size)
    new_figure = Figure(
        position=template(
            size,
            rnd.randint(min(img_width, 64), max_dx),
            rnd.randint(min(img_height, 64), max_dy),
        ),
        x_sup=img_width,
        y_sup=img_height,
        color=COLORS[rnd.randint(0, len(COLORS) - 1)],
    )

    FIGURES.append(new_figure)


def update_score(dscore: int):
    """Обновить очки."""
    global SCORE

    SCORE += dscore


def prepare_img(img):
    """Подготовка изображения."""

    img_height, img_width, _ = img.shape

    img = cv2.flip(img, 1)

    if HEALTH_POINTS <= 0:
        cv2.putText(
            img,
            f"YOU LOSE!)0)), SCORE: {SCORE}",
            (int(0.1 * img_width), img_height // 2),
            cv2.FONT_ITALIC,
            SCORE_FONT_SCALE,
            (0, 0, 255),
        )

        return cv2.flip(img, 1)
    else:
        cv2.putText(
            img,
            f"SCORE: {SCORE}, HP: {HEALTH_POINTS}",
            (img_width // 2, int(0.1 * img_height)),
            cv2.FONT_ITALIC,
            SCORE_FONT_SCALE,
            (0, 255, 0),
        )

        img = cv2.flip(img, 1)

    delete_by_liveness()
    create_new_figure(img_width, img_height)

    for figure in FIGURES:
        for s_point, e_point in figure.get_position_for_lines():
            cv2.line(img, s_point, e_point, figure.color, THICKNESS)

        goal = figure.get_goal_center(GOAL_RADIUS)
        cv2.circle(img, goal, GOAL_RADIUS, figure.color, THICKNESS)

        figure.try_delete(FIGURES, GOAL_RADIUS, lambda: update_score(dscore=1))

    return img


def handle_img(image, hand_landmarks, delta_fingers=20):
    """Обработка изображения."""

    image_height, image_width, _ = image.shape

    distance = lambda x1, x2, y1, y2: int(((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5)

    index_finger_delta = distance(
        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width,
        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x * image_width,
        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
        * image_height,
        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y
        * image_height,
    )
    thumb_delta = distance(
        hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * image_width,
        hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x * image_width,
        hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * image_height,
        hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y * image_height,
    )
    middle_finger_delta = distance(
        hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x
        * image_width,
        hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x
        * image_width,
        hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
        * image_height,
        hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y
        * image_height,
    )
    ring_finger_delta = distance(
        hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x * image_width,
        hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x * image_width,
        hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y * image_height,
        hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y * image_height,
    )
    pinky_delta = distance(
        hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x * image_width,
        hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x * image_width,
        hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y * image_height,
        hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y * image_height,
    )

    # проверка на сжатие руки
    if not any(
        filter(
            lambda x: x < delta_fingers,
            [
                index_finger_delta,
                thumb_delta,
                middle_finger_delta,
                ring_finger_delta,
                pinky_delta,
            ],
        )
    ):
        return image

    #  выбор точки для направляющего вектора, сейчас это подушечка под средним пальцем
    player_x = int(
        hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].x * image_width
    )
    player_y = int(
        hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y
        * image_height
    )

    #  ищем фигуры для точки направляющего вектора
    if figure := Figure.get_needed_figure(FIGURES, player_x, player_y):
        center = figure.get_center()
        fade = player_x - center[0], player_y - center[1]

        figure.update_position(*fade)  # сдвигаем фигуру

    return image


def capture_video(window_name):
    """Обработать видео."""

    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(
        model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as hands:
        while cap.isOpened():
            success, image = cap.read()

            if not success:
                print("Ignoring empty camera frame.")

                continue

            # какая-то микрооптимизация от авторов
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            image = prepare_img(image)

            if results.multi_hand_landmarks:  # обработка каждой руки
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style(),
                    )

                    image = handle_img(image, hand_landmarks)

            cv2.imshow(window_name, cv2.flip(image, 1))

            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()


def main():
    capture_video("GAME 003")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

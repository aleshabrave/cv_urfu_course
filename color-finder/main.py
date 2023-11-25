import cv2
import copy


def capture_video(input, window_name):
    """Обработать видео."""

    vid = cv2.VideoCapture(input)

    while vid.isOpened:
        ret, frame = vid.read()

        draw_contours(img=frame, contours=get_contours(frame))

        cv2.imshow(window_name, frame)

        if cv2.waitKey(30) == 27:
            break

    vid.release()


def get_contours(frame):
    """Получить контуры."""

    hue_range = (65, 95)  # отрезок подходящих hue (green)

    frame = copy.deepcopy(frame)

    frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)     
    frame_threshold = cv2.inRange(frame_HSV, (hue_range[0], 0, 0), (hue_range[1], 255, 255))  # пороговая обработка

    cv2.imshow('test', frame_threshold)  # чтоб два окошка было

    contours, hierarchy = cv2.findContours(frame_threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)  # ищем все контуры, причес без сглаживания
    
    return contours

def draw_contours(img, contours):
    """Нарисовать контуры."""
    color = (0, 255, 0)  # цвет контура

    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)

        if not is_okay_contour(x, y, w, h):
            continue

        cv2.rectangle(img, (x,y), (x+w,y+h), color, 2)


def is_okay_contour(x, y, w, h):
    """Проверка на мусор(шум)."""

    return w * h > 1000


def main():
    capture_video(0, "webcam")
    #capture_video("example.mp4", "video")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
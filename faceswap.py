import cv2
import random
import numpy as np
import matplotlib.pyplot as plt


def imsize(img):
    height, width, channels = img.shape
    return height * width


def mean_colour(img):
    mean_row = np.mean(img, axis=0)
    return np.mean(mean_row, axis=0)


def random_face(img, face_boxes):
    x, y, w, h = random.choice(face_boxes)
    return img[y:y + h, x:x + w]


def adjust_brightness(img, col, prev_col):
    b, g, r = cv2.split(img)
    diff_b, diff_g, diff_r = prev_col - col
    b = cv2.add(b, diff_b)
    g = cv2.add(g, diff_g)
    r = cv2.add(r, diff_r)
    return cv2.merge((b, g, r))


def swap_face(face, newface, mask, invmask):
    nb, ng, nr = cv2.split(newface)  # Foreground
    b, g, r = cv2.split(face)  # Background
    nb = (nb.astype(np.float32) * mask).astype(np.uint8)
    ng = (ng.astype(np.float32) * mask).astype(np.uint8)
    nr = (nr.astype(np.float32) * mask).astype(np.uint8)
    b = (b.astype(np.float32) * invmask).astype(np.uint8)
    g = (g.astype(np.float32) * invmask).astype(np.uint8)
    r = (r.astype(np.float32) * invmask).astype(np.uint8)
    fg = cv2.merge((nb, ng, nr))
    bg = cv2.merge((b, g, r))
    return cv2.add(fg, bg)


def main():
    # Read stuff
    face_cascade = cv2.CascadeClassifier('./faceswap/haarcascade_frontalface_default.xml')
    img = cv2.imread('./faceswap/people.jpg')
    mask = cv2.imread('./faceswap/facemask.png', 0)
    mask = (mask / 255.0).astype(np.float32)

    if img is None:
        print 'Could not find image'
        exit(1)

    # Resize the image to a width of 600px
    r = 600.0 / img.shape[1]
    dim = dim = (600, int(img.shape[0] * r))
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    # Face detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_boxes = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(face_boxes) < 2:
        print 'Not enoght face_boxes found D:'
        exit(1)
    print 'Found', len(face_boxes), 'faces'

    # Pick random new face
    newface = random_face(img, face_boxes)
    newface_meancol = mean_colour(newface)

    # Replace each face for the new one
    for (x, y, w, h) in face_boxes:
        face = img[y:y + h, x:x + w]
        face_meancol = mean_colour(face)
        adjusted_new_face = adjust_brightness(newface, newface_meancol,
                                              face_meancol)
        scaled_new_face = cv2.resize(adjusted_new_face, (w, h))
        scaled_mask = cv2.resize(mask, (w, h))
        invmask = 1 - scaled_mask
        img[y:y + h, x:x + w] = swap_face(face, scaled_new_face, scaled_mask,
                                          invmask)

    # Show the result
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    main()

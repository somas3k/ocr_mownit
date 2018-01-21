import os

import numpy as np
from scipy import ndimage
import cv2

import patterns_creator


NUM_OF_PATTERNS = 66
PATH_TO_FONTS = os.path.realpath("./fonts")

eps = 19
SPACE_DELIMITER = 15
NEW_LINE_DELIMITER = 32


def get_original_text(path):
    return open(path, "r").read()


def count_diff_chars_in_text(text):
    counter_array = np.zeros(127)
    for char in text:
        counter_array[ord(char)] += 1
    return counter_array


def get_line_from_img(img):
    end_of_img = False
    row_up_bound = -1
    row_down_bound = -1

    for row_y1 in range(img.shape[0]):
        if (abs(img[row_y1]) > eps).any():
            if row_up_bound == -1:
                row_up_bound = row_y1
            else:
                break

    for row_y2 in range(row_y1, img.shape[0]):
        if (abs(img[row_y2]) < eps).all():
            if row_down_bound == -1:
                row_down_bound = row_y2
            else:
                break

    line = img[row_up_bound:row_down_bound + 1, :]
    img = img[row_down_bound + 1:, :]

    if row_up_bound == -1:
        end_of_img = True

    return img, line, end_of_img


def clean_char(char):
    char_up_bound = -1
    char_down_bound = -1

    for char_y1 in range(char.shape[0]):
        if any(abs(char[char_y1]) > eps):
            if char_up_bound == -1:
                char_up_bound = char_y1
            else:
                break
    for char_y2 in range(char.shape[0]-1, char_y1, -1):
        if any(abs(char[char_y2]) > eps):
            if char_down_bound == -1:
                char_down_bound = char_y2
            else:
                break

    return char[char_up_bound:char_down_bound+1, :]


def get_char_from_line(line):
    end_of_line = False
    is_space = False
    char_left_bound = -1
    char_right_bound = -1

    for char_x1 in range(line.shape[1]):
        if any(abs(line[:, char_x1]) > eps):
            if char_left_bound == -1:
                char_left_bound = char_x1
            else:
                break

    if SPACE_DELIMITER <= char_left_bound <= NEW_LINE_DELIMITER:
        is_space = True

    for char_x2 in range(char_x1, line.shape[1]):
        if all(abs(line[:, char_x2]) < eps):
            if char_right_bound == -1:
                char_right_bound = char_x2 - 1
            else:
                break

    if char_left_bound == -1:
        end_of_line = True

    char = line[:, char_left_bound:char_right_bound + 1]
    line = line[:, char_right_bound + 1:]

    return line, char, end_of_line, is_space


def get_correlation(img, char_path):
    pattern = ndimage.imread(char_path, flatten=True)
    pattern = 255 - pattern
    f_img = np.fft.fft2(img)
    f_pattern = np.fft.fft2(np.rot90(pattern, 2), f_img.shape)
    corr_matrix = np.multiply(f_img, f_pattern)
    correlation = np.fft.ifft2(corr_matrix)
    correlation = np.abs(correlation)
    correlation = correlation.astype(float)

    fit_value = np.amax(correlation)

    return fit_value


def get_decimal_value_of_char(char):
    if 1 <= char <= 26:
        return char + 96
    if 27 <= char <= 52:
        return char + 38
    if 53 <= char <= 62:
        return char - 5
    if char == 63:
        return 46
    if char == 64:
        return 44
    if char == 65:
        return 63
    if char == 66:
        return 33
    return char


def get_char(char, font_name):
    best_fit = 0
    best_char = 0
    for i in range(1, NUM_OF_PATTERNS + 1):
        pattern_path = PATH_TO_FONTS + "/" + font_name + "/" + str(i) + ".png"
        fit_value = get_correlation(char, pattern_path)
        if fit_value > best_fit:
            best_fit = fit_value
            best_char = i

    return chr(get_decimal_value_of_char(best_char))


def get_text_from_image(img, font_name, path):
    end_of_img = False
    fh = open(path, "w")
    fh.write("")
    text = ""
    while not end_of_img:
        end_of_line = False
        text_line = ""
        img, line, end_of_img = get_line_from_img(img)

        while not end_of_line:
            line, char, end_of_line, is_space = get_char_from_line(line)

            if is_space:
                text_line += " "
            if not end_of_line:
                char = clean_char(char)
                read_char = get_char(char, font_name)
                text_line += read_char
        text += text_line + "\n"
    text = text[:len(text)-2] + text[len(text)-2:].replace("\n", "")
    fh.write(text)
    fh.close()


def clean_text_image(image):
    up_border = -1
    down_border = -1
    left_border = -1
    right_border = -1

    for y1 in range(0, image.shape[0]):
        if (abs(image[y1]) > eps).any():
            if up_border == -1:
                up_border = y1
            else:
                break

    for y2 in range(image.shape[0] - 1, y1, -1):
        if (abs(image[y2]) > eps).any():
            if down_border == -1:
                down_border = y2
            else:
                break

    for x1 in range(0, image.shape[1]):
        if (abs(image[:, x1]) > eps).any():
            if left_border == -1:
                left_border = x1
            else:
                break

    for x2 in range(image.shape[1] - 1, x1, -1):
        if (abs(image[:, x2]) > eps).any():
            if right_border == -1:
                right_border = x2
            else:
                break

    return image[up_border - 10:down_border+10, left_border - 10:right_border+10]


def prepare_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    if not abs(angle) < 2:

        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h),
                                 flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        cv2.imwrite("tmp.png", rotated)

        image = ndimage.imread("tmp.png", flatten=True)
        os.remove("tmp.png")
    else:
        image = ndimage.imread(image_path, flatten=True)
    image = 255 - image
    image[image < eps] = 0
    image[image >= eps] = 255
    image = clean_text_image(image)

    return image


def count_diff_chars_in_file(path):
    counter = np.zeros(127)
    with open(path) as f:
        content = f.readlines()

    content = [x.strip() for x in content]
    for line in content:
        for char in line:
            counter[ord(char)] += 1
    return counter


def get_recognizing_stats(input_text_stats, processed_img_stats):
    errors = 0
    for i in range(input_text_stats.shape[0]):
        errors += abs(input_text_stats[i] - processed_img_stats[i])

    error_rate = errors / sum(input_text_stats)
    print("Error rate = ", error_rate)


def main():
    patterns_creator.main(eps)
    original_text = get_original_text("text.txt")
    input_text_stats = count_diff_chars_in_text(original_text)

    print("Working on ARIAL font")
    print("Straight image")
    image = prepare_image("./text_images/arial_text_2.png")
    get_text_from_image(image, "arial", "arial_text.txt")
    with open("arial_text.txt", "r") as processed:
        processed_img_stats = count_diff_chars_in_text(processed.read())
    get_recognizing_stats(input_text_stats, processed_img_stats)

    print("\n")

    print("Working on ARIAL font")
    print("Rotated image")
    image = prepare_image("./text_images/arial_text_with_rotation.png")
    get_text_from_image(image, "arial", "arial_text_r.txt")
    with open("arial_text_r.txt", "r") as processed:
        processed_img_stats = count_diff_chars_in_text(processed.read())
    get_recognizing_stats(input_text_stats, processed_img_stats)

    print("\n")

    print("Working on LIBERATION SERIF font")
    print("Straight image")
    image = prepare_image("./text_images/liberation_serif.png")
    get_text_from_image(image, "liberation", "liberation_text.txt")
    with open("liberation_text.txt", "r") as processed:
        processed_img_stats = count_diff_chars_in_text(processed.read())
    get_recognizing_stats(input_text_stats, processed_img_stats)

    print("\n")

    print("Working on LIBERATION SERIF font")
    print("Rotated image")
    image = prepare_image("./text_images/liberation_serif_with_rotation.png")
    get_text_from_image(image, "liberation", "liberation_text_r.txt")
    with open("liberation_text_r.txt", "r") as processed:
        processed_img_stats = count_diff_chars_in_text(processed.read())
    get_recognizing_stats(input_text_stats, processed_img_stats)


if __name__ == "__main__":
    main()

from scipy import ndimage
from scipy.misc import imsave


def clean_borders(image, eps):
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

    for y2 in range(image.shape[0]-1, y1, -1):
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

    for x2 in range(image.shape[1]-1, x1, -1):
        if (abs(image[:, x2]) > eps).any():
            if right_border == -1:
                right_border = x2
            else:
                break

    return image[up_border-1:down_border, left_border-2:right_border]


def clean_char(char, eps):
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


def get_char_from_line(line, eps):
    end_of_line = False
    char_left_bound = -1
    char_right_bound = -1

    for char_x1 in range(line.shape[1]):
        if any(abs(line[:, char_x1]) > eps):
            if char_left_bound == -1:
                char_left_bound = char_x1
            else:
                break

    if char_left_bound == -1:
        end_of_line = True
        char = None
    else:
        for char_x2 in range(char_x1, line.shape[1]):
            if all(abs(line[:, char_x2]) < eps):
                if char_right_bound == -1:
                    char_right_bound = char_x2 - 1
                else:
                    break
        if char_right_bound == -1:
            char_right_bound = line.shape[1] - 1
        char = line[:, char_left_bound:char_right_bound + 1]
        line = line[:, char_right_bound + 1:]

    return line, char, end_of_line


def get_line_from_img(img, eps):
    row_up_bound = -1
    row_down_bound = -1

    row_y1 = 0

    for row_y1 in range(img.shape[0]):
        if (abs(img[row_y1]) > eps).any():
            if row_up_bound == -1:
                row_up_bound = row_y1
            else:
                break

    row_y2 = img.shape[0]-1

    for row_y2 in range(img.shape[0]-1, row_y1, -1):
        if (abs(img[row_y2]) > eps).any():
            if row_down_bound == -1:
                row_down_bound = row_y2
            else:
                break

    if row_y2 == img.shape[0]-1:
        row_down_bound = row_y2

    line = img[row_up_bound:row_down_bound + 2, :]
    img = img[row_down_bound + 1:, :]

    return img, line


def crop_patterns(patterns, eps):
    pattern_list = []
    end_of_line = False

    patterns, line = get_line_from_img(patterns, eps)
    while not end_of_line:
        line, char, end_of_line = get_char_from_line(line, eps)
        if not end_of_line:
            char = clean_char(char, eps)
            pattern_list.append(char)

    return pattern_list


def save_patterns(patterns: list, patterns_type: str, font_name):
    start_index = 200
    if patterns_type.lower() in "lowercase":
        start_index = 1
    if patterns_type.lower() in "uppercase":
        start_index = 27
    if patterns_type.lower() in "numbers":
        start_index = 53
    if patterns_type.lower() in "special":
        start_index = 63

    for i in range(0, len(patterns)):
        pattern = patterns[i]
        pattern = 255 - pattern
        imsave("./fonts/"+font_name+"/"+str(start_index)+".png", pattern)
        start_index += 1


def create_patterns(image_path, patterns_type, font_name, eps):
    patterns = ndimage.imread(image_path, flatten=True)
    patterns = 255 - patterns
    patterns[patterns < eps] = 0
    patterns[patterns >= eps] = 255

    patterns = clean_borders(patterns, eps)
    patterns = crop_patterns(patterns, eps)
    save_patterns(patterns, patterns_type, font_name)


def main(eps=18):
    create_patterns("./patterns/lowercase_arial.png", "lowercase", "arial", eps)
    create_patterns("./patterns/uppercase_arial.png", "uppercase", "arial", eps)
    create_patterns("./patterns/numbers_arial.png", "numbers", "arial", eps)
    create_patterns("./patterns/special_arial.png", "special", "arial", eps)
    create_patterns("./patterns/lowercase_liberation.png", "lowercase", "liberation", eps)
    create_patterns("./patterns/uppercase_liberation.png", "uppercase", "liberation", eps)
    create_patterns("./patterns/numbers_liberation.png", "numbers", "liberation", eps)
    create_patterns("./patterns/special_liberation.png", "special", "liberation", eps)


if __name__ == "__main__":
    main()

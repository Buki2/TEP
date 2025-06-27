import base64
import re
import os
import cv2
from collections import Counter
from PIL import Image, ImageDraw, ImageFont

def read_prompt(path):
    with open(path, 'r') as f:
        prompt = f.read()
    return prompt

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def separate_code(text):
    try:
        match_text = re.search(r'```python(.*?)```', text, re.DOTALL)
        code = match_text.group(1)
        return code
    except Exception as e:
        print(e)
        return '', text

def store_cache(function_name, path, id, text):
    f_path = os.path.join(path, id + '_' + function_name + '.txt')
    with open(f_path, 'w') as f:
        f.write(text)

def load_cache(function_name, path, id):
    f_path = os.path.join(path, id + '_' + function_name + '.txt')
    if os.path.exists(f_path):
        with open(f_path, 'r') as f:
            text = f.read()
        return text
    else:
        return None

def extract_person_position(text):
    last_paragraph = text.split('\n\n')[-1]
    try:
        match_text = re.search(r'Therefore,(.*?)\.', last_paragraph)
        match_text = match_text.group(1).strip()
        marks = re.findall(r'\d+', match_text)
        if len(marks) == 0:
            marks = re.findall(r'\d+', last_paragraph)
        if len(marks) == 1:
            position_mark = marks[0]
        else:
            position_mark = Counter(marks).most_common(1)[0][0]
        return int(position_mark)
    except Exception as e:
        print(e)
        return 0

def draw_top_view_w_per(figure, mark):
    if mark < 10:
        new_mark = ' ' + str(mark) + ' '
        new_figure = figure.replace(new_mark, ' P ')
    else:
        new_figure = figure.replace(str(mark), 'P ')
    new_figure = re.sub(r'\d', ' ', new_figure)
    return new_figure

def extract_object_position(text):
    results = {}
    try:
        match_text = re.search(r'Therefore, the most possible marks(.*?)', text)
        match_text = text[match_text.start():] + '\n'
        match_object = re.findall(r'- Object (\d+)(.*?):(.*?)\n', match_text)
        for i in match_object:
            obj_num, _, obj_info = i

            column = re.findall(r'[^a-zA-Z]([A-Z])[^a-zA-Z]', obj_info + '\n')
            if len(column) > 1:
                column = Counter(column).most_common(1)[0][0]
            else:
                column = column[0]

            row = re.findall(r'\d', obj_info)
            if len(row) > 1 and len(row) < 4:
                row = Counter(row).most_common(1)[0][0]
            else:
                row = row[0]
            
            results[int(obj_num)] = [int(row), column]
    except Exception as e:
        print(e)
    return results

def ascii_to_image(ascii_art):
    font_size = 24
    font_path = "consola.ttf"
    font = ImageFont.truetype(font_path, font_size)
    line_spacing = 6

    image_height = len(ascii_art) * (font_size + line_spacing)
    image_width = image_height
    image_size = (image_width, image_height)

    # Create a blank white image
    image = Image.new("RGB", image_size, "white")
    draw = ImageDraw.Draw(image)

    x, y = 2, 2
    
    # Draw each line of ASCII art onto the image
    for line in ascii_art:
        draw.text((x, y), line, font=font, fill="black")
        y += font_size + line_spacing

    return image

def draw_top_view_w_obj(object_position, person_position, relationship_flag, id, cache_path):
    figure = [
        "                        ",
        "                        ",
        "                        ",
        "                        ",
        "                        ",
        "                        ",
        "                        ",
        "                        ",
        "                        ",
        "                        ",
        "                        ",
    ]
    
    # draw number marks using object_position
    for k, v in object_position.items():
        row, column = v
        figure_row = row + 2
        figure_column = ord(column) - 64 + 5  # 5 is the left margin to image edge
        number2letter = chr(k + 64)
        figure[figure_row] = figure[figure_row][:figure_column-1] + number2letter + figure[figure_row][figure_column:]

    image = ascii_to_image(figure)
    image_path = os.path.join(cache_path, 'ascii_image.jpg')
    image.save(image_path)

    # draw line using person_position
    img = cv2.imread(image_path)  # 330*330
    img_h, img_w = img.shape[:2]
    dict_person = {1: [(0, 0), 'LRBF'], 2: [(int(img_w / 4), 0), 'LRBF'], 3: [(int(img_w / 2), 0), 'LRBF'], 4: [(int(3 * img_w / 4), 0), 'LRFB'], 5: [(img_w, 0), 'LRFB'], 
                   6: [(img_w, int(img_h / 4)), 'LRFB'], 7: [(img_w, int(img_h / 2)), 'LRFB'], 8: [(img_w, int(3 * img_h / 4)), 'RLFB'], 9: [(img_w, img_h), 'RLFB'],
                   10: [(int(3 * img_w / 4), img_h), 'RLFB'], 11: [(int(img_w / 4), img_h), 'RLBF'], 12: [(0, img_h), 'RLBF'],
                   13: [(0, int(3 * img_h / 4)), 'RLBF'], 14: [(0, int(img_h / 2)), 'RLBF'], 15: [(0, int(img_h / 4)), 'LRBF']}
    start_point = dict_person[person_position][0]
    end_point = (int(img_w/2), int(img_h/2))

    if end_point[0] - start_point[0] != 0:
        slope_ori = (end_point[1] - start_point[1]) / (end_point[0] - start_point[0])
        slope_perpen = -1 / slope_ori if slope_ori != 0 else float('inf')
    else:
        slope_ori = float('inf')
        slope_perpen = 0
    
    if relationship_flag == 0:  # left and right
        slope = slope_perpen
        new_start_point = end_point
        if slope != float('inf'):
            if abs(slope) == 0:
                line_length = 110
            elif abs(slope) == 1:
                line_length = 104
            elif abs(round(slope, 1)) == 2:
                line_length = 60
            elif abs(round(slope, 1)) == 0.5:
                line_length = 108
            
            end_point_perpen = (new_start_point[0] + line_length, int(new_start_point[1] + slope * line_length))
            cv2.arrowedLine(img, new_start_point, end_point_perpen, (255, 0, 0), 2, line_type=cv2.LINE_AA)  # blue
            direction_name = 'Left' if dict_person[person_position][1][0] == 'L' else 'Right'
            text_margin_x = 5 if end_point_perpen[0] >= int(img_w/2) else -55
            text_margin_y = 10 if end_point_perpen[1] >= int(img_h/2) else -10
            cv2.putText(img, direction_name, (end_point_perpen[0] + text_margin_x, int(end_point_perpen[1] + slope * text_margin_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

            end_point_perpen = (new_start_point[0] - line_length, int(new_start_point[1] - slope * line_length))
            cv2.arrowedLine(img, new_start_point, end_point_perpen, (0, 0, 255), 2, line_type=cv2.LINE_AA)  # red
            direction_name = 'Left' if dict_person[person_position][1][1] == 'L' else 'Right'
            text_margin_x = 5 if end_point_perpen[0] >= int(img_w/2) else -55
            text_margin_y = 10 if end_point_perpen[1] >= int(img_h/2) else -10
            cv2.putText(img, direction_name, (end_point_perpen[0] + text_margin_x, int(end_point_perpen[1] + slope * text_margin_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        else:
            line_length = 110
            end_point_perpen = (new_start_point[0], int(new_start_point[1] + line_length))
            cv2.arrowedLine(img, new_start_point, end_point_perpen, (255, 0, 0), 2, line_type=cv2.LINE_AA)  # blue
            direction_name = 'Left' if dict_person[person_position][1][0] == 'L' else 'Right'
            text_margin_x = 5 if end_point_perpen[0] >= int(img_w/2) else -55
            text_margin_y = 10 if end_point_perpen[1] >= int(img_h/2) else -10
            cv2.putText(img, direction_name, (end_point_perpen[0] + text_margin_x, int(end_point_perpen[1] + text_margin_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

            end_point_perpen = (new_start_point[0], int(new_start_point[1] - line_length))
            cv2.arrowedLine(img, new_start_point, end_point_perpen, (0, 0, 255), 2, line_type=cv2.LINE_AA)  # red
            direction_name = 'Left' if dict_person[person_position][1][1] == 'L' else 'Right'
            text_margin_x = 5 if end_point_perpen[0] >= int(img_w/2) else -55
            text_margin_y = 10 if end_point_perpen[1] >= int(img_h/2) else -10
            cv2.putText(img, direction_name, (end_point_perpen[0] + text_margin_x, int(end_point_perpen[1] + text_margin_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        
    else:  # front and back
        slope = slope_ori
        new_start_point = end_point
        if slope != float('inf'):
            if abs(slope) == 0:
                line_length = 110
            elif abs(slope) == 1:
                line_length = 104
            elif abs(round(slope, 1)) == 2:
                line_length = 60
            elif abs(round(slope, 1)) == 0.5:
                line_length = 108
            
            end_point_perpen = (new_start_point[0] + line_length, int(new_start_point[1] + slope * line_length))
            cv2.arrowedLine(img, new_start_point, end_point_perpen, (0, 255, 0), 2, line_type=cv2.LINE_AA)  # green
            direction_name = 'Front' if dict_person[person_position][1][2] == 'F' else 'Back'
            text_margin_x = 5 if end_point_perpen[0] >= int(img_w/2) else -55
            text_margin_y = 10 if end_point_perpen[1] >= int(img_h/2) else -10
            cv2.putText(img, direction_name, (end_point_perpen[0] + text_margin_x, int(end_point_perpen[1] + slope * text_margin_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            end_point_perpen = (new_start_point[0] - line_length, int(new_start_point[1] - slope * line_length))
            cv2.arrowedLine(img, new_start_point, end_point_perpen, (0, 0, 255), 2, line_type=cv2.LINE_AA)  # red
            direction_name = 'Front' if dict_person[person_position][1][3] == 'F' else 'Back'
            text_margin_x = 5 if end_point_perpen[0] >= int(img_w/2) else -55
            text_margin_y = 10 if end_point_perpen[1] >= int(img_h/2) else -10
            cv2.putText(img, direction_name, (end_point_perpen[0] + text_margin_x, int(end_point_perpen[1] + slope * text_margin_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            line_length = 110
            end_point_perpen = (new_start_point[0], new_start_point[1] + line_length)
            cv2.arrowedLine(img, new_start_point, end_point_perpen, (0, 255, 0), 2, line_type=cv2.LINE_AA)
            direction_name = 'Front' if dict_person[person_position][1][2] == 'F' else 'Back'
            text_margin_x = 5 if end_point_perpen[0] >= int(img_w/2) else -55
            text_margin_y = 10 if end_point_perpen[1] >= int(img_h/2) else -10
            cv2.putText(img, direction_name, (end_point_perpen[0] + text_margin_x, int(end_point_perpen[1] + text_margin_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            end_point_perpen = (new_start_point[0], new_start_point[1] - line_length)
            cv2.arrowedLine(img, new_start_point, end_point_perpen, (0, 0, 255), 2, line_type=cv2.LINE_AA)
            direction_name = 'Front' if dict_person[person_position][1][3] == 'F' else 'Back'
            text_margin_x = 5 if end_point_perpen[0] >= int(img_w/2) else -55
            text_margin_y = 10 if end_point_perpen[1] >= int(img_h/2) else -10
            cv2.putText(img, direction_name, (end_point_perpen[0] + text_margin_x, int(end_point_perpen[1] + text_margin_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    new_image_path = image_path.replace('ascii_image', id + '_ascii_image_arrow')
    cv2.imwrite(new_image_path, img)
    
    return new_image_path


def draw_top_view_w_obj_composition(object_position, person_position, id, cache_path):
    figure = [
        "                        ",
        "                        ",
        "                        ",
        "                        ",
        "                        ",
        "                        ",
        "                        ",
        "                        ",
        "                        ",
        "                        ",
        "                        ",
    ]
    for k, v in object_position.items():
        row, column = v
        figure_row = row + 2
        figure_column = ord(column) - 64 + 5  # 5 is the left margin to image edge
        number2letter = chr(k + 64)
        figure[figure_row] = figure[figure_row][:figure_column-1] + number2letter + figure[figure_row][figure_column:]

    image = ascii_to_image(figure)
    image_path = os.path.join(cache_path, 'ascii_image.jpg')
    image.save(image_path)

    # draw line using person_position
    img = cv2.imread(image_path)  # 330*330
    img_h, img_w = img.shape[:2]
    dict_person = {1: [(0, 0), 'LRBF'], 2: [(int(img_w / 4), 0), 'LRBF'], 3: [(int(img_w / 2), 0), 'LRBF'], 4: [(int(3 * img_w / 4), 0), 'LRFB'], 5: [(img_w, 0), 'LRFB'], 
                   6: [(img_w, int(img_h / 4)), 'LRFB'], 7: [(img_w, int(img_h / 2)), 'LRFB'], 8: [(img_w, int(3 * img_h / 4)), 'RLFB'], 9: [(img_w, img_h), 'RLFB'],
                   10: [(int(3 * img_w / 4), img_h), 'RLFB'], 11: [(int(img_w / 4), img_h), 'RLBF'], 12: [(0, img_h), 'RLBF'],
                   13: [(0, int(3 * img_h / 4)), 'RLBF'], 14: [(0, int(img_h / 2)), 'RLBF'], 15: [(0, int(img_h / 4)), 'LRBF']}
    start_point = dict_person[person_position][0]
    end_point = (int(img_w/2), int(img_h/2))

    if end_point[0] - start_point[0] != 0:
        slope_ori = (end_point[1] - start_point[1]) / (end_point[0] - start_point[0])
        slope_perpen = -1 / slope_ori if slope_ori != 0 else float('inf')
    else:
        slope_ori = float('inf')
        slope_perpen = 0
    
    # left and right
    slope = slope_perpen
    new_start_point = end_point
    if slope != float('inf'):
        if abs(slope) == 0:
            line_length = 110
        elif abs(slope) == 1:
            line_length = 104
        elif abs(round(slope, 1)) == 2:
            line_length = 60
        elif abs(round(slope, 1)) == 0.5:
            line_length = 108
        
        end_point_perpen = (new_start_point[0] + line_length, int(new_start_point[1] + slope * line_length))
        cv2.arrowedLine(img, new_start_point, end_point_perpen, (255, 0, 0), 2, line_type=cv2.LINE_AA)  # blue
        direction_name = 'Left' if dict_person[person_position][1][0] == 'L' else 'Right'
        text_margin_x = 5 if end_point_perpen[0] >= int(img_w/2) else -55
        text_margin_y = 10 if end_point_perpen[1] >= int(img_h/2) else -10
        cv2.putText(img, direction_name, (end_point_perpen[0] + text_margin_x, int(end_point_perpen[1] + slope * text_margin_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

        end_point_perpen = (new_start_point[0] - line_length, int(new_start_point[1] - slope * line_length))
        cv2.arrowedLine(img, new_start_point, end_point_perpen, (0, 0, 255), 2, line_type=cv2.LINE_AA)  # red
        direction_name = 'Left' if dict_person[person_position][1][1] == 'L' else 'Right'
        text_margin_x = 5 if end_point_perpen[0] >= int(img_w/2) else -55
        text_margin_y = 10 if end_point_perpen[1] >= int(img_h/2) else -10
        cv2.putText(img, direction_name, (end_point_perpen[0] + text_margin_x, int(end_point_perpen[1] + slope * text_margin_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    else:
        line_length = 110
        end_point_perpen = (new_start_point[0], int(new_start_point[1] + line_length))
        cv2.arrowedLine(img, new_start_point, end_point_perpen, (255, 0, 0), 2, line_type=cv2.LINE_AA)  # blue
        direction_name = 'Left' if dict_person[person_position][1][0] == 'L' else 'Right'
        text_margin_x = 5 if end_point_perpen[0] >= int(img_w/2) else -55
        text_margin_y = 10 if end_point_perpen[1] >= int(img_h/2) else -10
        cv2.putText(img, direction_name, (end_point_perpen[0] + text_margin_x, int(end_point_perpen[1] + text_margin_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

        end_point_perpen = (new_start_point[0], int(new_start_point[1] - line_length))
        cv2.arrowedLine(img, new_start_point, end_point_perpen, (0, 0, 255), 2, line_type=cv2.LINE_AA)  # red
        direction_name = 'Left' if dict_person[person_position][1][1] == 'L' else 'Right'
        text_margin_x = 5 if end_point_perpen[0] >= int(img_w/2) else -55
        text_margin_y = 10 if end_point_perpen[1] >= int(img_h/2) else -10
        cv2.putText(img, direction_name, (end_point_perpen[0] + text_margin_x, int(end_point_perpen[1] + text_margin_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        
    # front and back
    slope = slope_ori
    new_start_point = end_point
    if slope != float('inf'):
        if abs(slope) == 0:
            line_length = 110
        elif abs(slope) == 1:
            line_length = 104
        elif abs(round(slope, 1)) == 2:
            line_length = 60
        elif abs(round(slope, 1)) == 0.5:
            line_length = 108
        
        end_point_perpen = (new_start_point[0] + line_length, int(new_start_point[1] + slope * line_length))
        cv2.arrowedLine(img, new_start_point, end_point_perpen, (0, 255, 0), 2, line_type=cv2.LINE_AA)  # green
        direction_name = 'Front' if dict_person[person_position][1][2] == 'F' else 'Back'
        text_margin_x = 5 if end_point_perpen[0] >= int(img_w/2) else -55
        text_margin_y = 10 if end_point_perpen[1] >= int(img_h/2) else -10
        cv2.putText(img, direction_name, (end_point_perpen[0] + text_margin_x, int(end_point_perpen[1] + slope * text_margin_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        end_point_perpen = (new_start_point[0] - line_length, int(new_start_point[1] - slope * line_length))
        cv2.arrowedLine(img, new_start_point, end_point_perpen, (0, 192, 255), 2, line_type=cv2.LINE_AA)  # yellow
        direction_name = 'Front' if dict_person[person_position][1][3] == 'F' else 'Back'
        text_margin_x = 5 if end_point_perpen[0] >= int(img_w/2) else -55
        text_margin_y = 10 if end_point_perpen[1] >= int(img_h/2) else -10
        cv2.putText(img, direction_name, (end_point_perpen[0] + text_margin_x, int(end_point_perpen[1] + slope * text_margin_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 192, 255), 2)
    else:
        line_length = 110
        end_point_perpen = (new_start_point[0], new_start_point[1] + line_length)
        cv2.arrowedLine(img, new_start_point, end_point_perpen, (0, 255, 0), 2, line_type=cv2.LINE_AA)
        direction_name = 'Front' if dict_person[person_position][1][2] == 'F' else 'Back'
        text_margin_x = 5 if end_point_perpen[0] >= int(img_w/2) else -55
        text_margin_y = 10 if end_point_perpen[1] >= int(img_h/2) else -10
        cv2.putText(img, direction_name, (end_point_perpen[0] + text_margin_x, int(end_point_perpen[1] + text_margin_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        end_point_perpen = (new_start_point[0], new_start_point[1] - line_length)
        cv2.arrowedLine(img, new_start_point, end_point_perpen, (0, 192, 255), 2, line_type=cv2.LINE_AA)
        direction_name = 'Front' if dict_person[person_position][1][3] == 'F' else 'Back'
        text_margin_x = 5 if end_point_perpen[0] >= int(img_w/2) else -55
        text_margin_y = 10 if end_point_perpen[1] >= int(img_h/2) else -10
        cv2.putText(img, direction_name, (end_point_perpen[0] + text_margin_x, int(end_point_perpen[1] + text_margin_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 192, 255), 2)
    

    new_image_path = image_path.replace('ascii_image', id + '_ascii_image_arrow')
    cv2.imwrite(new_image_path, img)
    return new_image_path

def extract_horizontal_result(text):
    results = []
    match_text = re.search(r'target_letter\s*=\s*', text)
    if match_text:
        match_letters = re.findall(r'[A-Z]', text[match_text.end():])
        results.extend(match_letters)
    return results

def compute_iou(box1, box2):  # xywh
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    x_intersection = max(x1, x2)
    y_intersection = max(y1, y2)
    w_intersection = max(0, min(x1 + w1, x2 + w2) - x_intersection)
    h_intersection = max(0, min(y1 + h1, y2 + h2) - y_intersection)

    area_intersection = w_intersection * h_intersection
    area_union = w1 * h1 + w2 * h2 - area_intersection

    iou = area_intersection / max(area_union, 1e-10)

    return iou

def area_intersection(box1, box2):  # xywh
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    x_intersection = max(x1, x2)
    y_intersection = max(y1, y2)
    w_intersection = max(0, min(x1 + w1, x2 + w2) - x_intersection)
    h_intersection = max(0, min(y1 + h1, y2 + h2) - y_intersection)

    area_intersection = w_intersection * h_intersection

    return area_intersection
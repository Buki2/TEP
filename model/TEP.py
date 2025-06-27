import time
from utils import *
from model.language_models import *
from model.verification_module import *
from model.glip_mebow.load_glip import visual_glip
from model.glip_mebow.load_mebow import visual_mebow

class Object:
    def __init__(self, name, bbox=None, depth_all=None):
        self.name = name

        self.bbox = bbox  # xywh
        self.x_min = self.bbox[0]
        self.x_max = self.bbox[0] + self.bbox[2]
        self.x_center = int(self.bbox[0] + self.bbox[2] / 2)
        self.y_min = self.bbox[1]
        self.y_max = self.bbox[1] + self.bbox[3]
        self.y_center = int(self.bbox[1] + self.bbox[3] / 2)
        self.w = self.bbox[2]
        self.h = self.bbox[3]
        self.area = self.w * self.h

        self.depth_all = depth_all  # min, max, median
        self.depth = round(self.depth_all[2], 4)
        if self.depth_all[0] < 0:
            self.depth_all[0] = 0.0
        if self.depth_all[1] > 1:
            self.depth_all[1] = 1.0
        self.depth_min = round(self.depth_all[0], 4)
        self.depth_max = round(self.depth_all[1], 4)

        self.number_mark = 0
        self.tmp = 0

def find_object(name, attribute=None):
    global CURRENT_LINE
    CURRENT_LINE += 1
    results = []
    for i in OBJ_RESULT:
        if i[0] == name or (attribute and i[0] == attribute + ' ' + name):
            results.append(Object(i[0], i[1], i[2]))
    return results

def filter_by_vertical_relationship(object_list, subsentence, reference_object_list=None):
    global CURRENT_LINE
    CURRENT_LINE += 1

    if len(object_list) == 1:
        results = object_list
        return results
    
    keywords_vertical_below = ['below', 'underneath', 'under', 'beneath', 'bottom', 'down', 'lower']
    results = []
    thres_cover_most = 0.8
    thres_cover_least = 0.2

    if reference_object_list:
        for obj in reference_object_list:
            if not isinstance(obj, Object):
                reference_object_list = []
                break

    # Step 1: if reference_object_list is not empty, find the objects that fulfill the conditions
    if reference_object_list:
        if set(subsentence.split()).intersection(set(keywords_vertical_below)):  # relationship = below
            for obj in object_list:
                for ref in reference_object_list:
                    if obj.w * obj.h > ref.w * ref.h:
                        if (compute_iou(obj.bbox, ref.bbox) > thres_cover_most or area_intersection(obj.bbox, ref.bbox) == ref.w * ref.h) and obj.depth_min <= ref.depth <= obj.depth_max:
                            results.append(obj)
                            ranking_criteria = 'd'
                    else:
                        if 0 < compute_iou(obj.bbox, ref.bbox) <= thres_cover_least and obj.y_center >= ref.y_center:
                            results.append(obj)
                            ranking_criteria = 'y'
        else:  # relationship = up
            for obj in object_list:
                for ref in reference_object_list:
                    if obj.w * obj.h < ref.w * ref.h:
                        if (compute_iou(obj.bbox, ref.bbox) > thres_cover_most or area_intersection(obj.bbox, ref.bbox) == obj.w * obj.h) and ref.depth_min <= obj.depth <= ref.depth_max:
                            results.append(obj)
                            ranking_criteria = 'd'
                    else:
                        if 0 < compute_iou(obj.bbox, ref.bbox) <= thres_cover_least and obj.y_center <= ref.y_center:
                            results.append(obj)
                            ranking_criteria = 'y'
    else:
        results = object_list
        ranking_criteria = 'y'
    
    # Step 2: ranking
    if len(results) == 0:
        # no result: degrade
        if set(subsentence.split()).intersection(set(keywords_vertical_below)):  # relationship = below
            for obj in object_list:
                for ref in reference_object_list:
                    if obj.w * obj.h > ref.w * ref.h:
                        obj.tmp = area_intersection(obj.bbox, ref.bbox) / (obj.w * obj.h)
                        ranking_criteria = 'd'
                    else:
                        ranking_criteria = 'y'
            results = object_list
            if ranking_criteria == 'd':
                results = sorted(results, key=lambda x: x.tmp)
            else:
                results = sorted(results, key=lambda x: x.y_center, reverse=True)
        else:
            # ranking by iou or area_intersection/w*h
            for obj in object_list:
                for ref in reference_object_list:
                    if obj.w * obj.h < ref.w * ref.h:
                        obj.tmp = area_intersection(obj.bbox, ref.bbox) / (obj.w * obj.h)
                        ranking_criteria = 'd'
                    else:
                        ranking_criteria = 'y'
            results = object_list
            if ranking_criteria == 'd':
                results = sorted(results, key=lambda x: x.tmp, reverse=True)
            else:
                results = sorted(results, key=lambda x: x.y_center)
    elif len(results) > 1:
        if set(subsentence.split()).intersection(set(keywords_vertical_below)):  # relationship = below
            if ranking_criteria == 'd':
                results = sorted(results, key=lambda x: x.depth, reverse=True)  # the greater the depth value, the greater the likelihood of the object being located below
            else:
                results = sorted(results, key=lambda x: x.y_center, reverse=True)  # the greater the y_center value, the greater the likelihood of the object being located below
        else:  # relationship = up
            if ranking_criteria == 'd':
                results = sorted(results, key=lambda x: x.depth)
            else:
                results = sorted(results, key=lambda x: x.y_center)
    
    if len(results) == 0:
        results = object_list
    
    # ordinal
    if set(subsentence.split()).intersection(set(['first', 'second', 'third', 'fourth'])):
        if len(results) > 0 and 'first' in subsentence.split():
            results = results[0]
        elif len(results) > 1 and 'second' in subsentence.split():
            results = results[1]
        elif len(results) > 2 and 'third' in subsentence.split():
            results = results[2]
        elif len(results) > 3 and 'fourth' in subsentence.split():
            results = results[3]
    
    if not reference_object_list and len(results) > 1:
        results = results[0]

    if not isinstance(results, list):
        results = [results]

    return results

def filter_by_horizontal_relationship(object_list, subsentence, reference_object_list=None):
    global CURRENT_LINE
    CURRENT_LINE += 1
    results = []

    if len(object_list) == 1:
        results = object_list
        return results

    obj_person_list = []
    obj_table_list = []
    for i in OBJ_RESULT:
        if i[0] == 'person':
            obj_person_list.append(Object('person', i[1], i[2]))
        elif i[0] == 'table':
            obj_table_list.append(Object('table', i[1], i[2]))
    if len(obj_person_list) > 1:
        obj_person_list = sorted(obj_person_list, key=lambda x: x.area, reverse=True)
    obj_person = obj_person_list[0]
    if len(obj_table_list) > 1:
        obj_table_list = sorted(obj_table_list, key=lambda x: x.area, reverse=True)
    obj_table = obj_table_list[0]

    # Step1: person's position in top view

    # use body orientation estimation model to obtain person positions
    person_position = visual_mebow(IMAGE_PATH, obj_person.bbox)
    store_cache('mebow_results', ARGS.cache_path, ID, str(person_position))

    # Step1.5: verify the generated top view
    flag_verification_person = True
    ver_per_result = 0
    if flag_verification_person and person_position:
        ver_per_result = verification_person(obj_person, obj_table, person_position)
    if not ver_per_result or not person_position:
        prompt = read_prompt(ARGS.prompt_path + '2_top_person.txt')
        obj_info_str = ''
        if obj_person:
            obj_info_str += 'person, ' + str([obj_person.x_min, obj_person.y_min, obj_person.x_max, obj_person.y_max]) + ', ' + '%.4f' % obj_person.depth + '\n'
        if obj_table:
            obj_info_str += 'table, ' + str([obj_table.x_min, obj_table.y_min, obj_table.x_max, obj_table.y_max]) + ', ' + '%.4f' % obj_table.depth + '\n'
        prompt = prompt.replace('<OBJECT_INFORMATION>', obj_info_str)
        prompt = prompt.replace('<INT_X_CENTER_PERSON>', str(obj_person.x_center))
        
        response = load_cache('top_person', ARGS.cache_path, ID)
        if not response:
            response = llm_vision(ARGS.llm_top_person, prompt, IMAGE_PATH)
            if ARGS.flag_write_cache:
                store_cache('top_person', ARGS.cache_path, ID, response)
        person_position_new = extract_person_position(response)

        if not person_position_new:
            if response.startswith('I\'m sorry') or response.startswith('Sorry'):
                # repeat
                print('sleeping...')
                time.sleep(3)
                response = llm_vision(ARGS.llm_top_person, prompt, IMAGE_PATH)
                if ARGS.flag_write_cache:
                    store_cache('top_person', ARGS.cache_path, ID, response)
                person_position_new = extract_person_position(response)
        if not person_position_new:
            if person_position:
                person_position_new = person_position
            else:
                person_position_new = '3'
        
        person_position = person_position_new
    
    person_position = int(person_position)


    # Step2: objects' positions in top view
    prompt = read_prompt(ARGS.prompt_path + '3_top_object.txt')
    default_person = read_prompt(ARGS.prompt_path + 'default_person.txt')
    top_view_w_per = draw_top_view_w_per(default_person, person_position)
    prompt = prompt.replace('<TOP_VIEW_MAP>', top_view_w_per)
    
    obj_info_str = ''
    obj_number = 1
    for obj in object_list:
        obj_info_str += str(obj_number) + '. ' + obj.name + ', x_center=' + str(obj.x_center) + ', depth=' + '%.4f' % obj.depth + '\n'
        obj.number_mark = obj_number
        obj_number += 1
    if reference_object_list:
        for obj in reference_object_list:
            if not isinstance(obj, Object):
                reference_object_list = []
                break
            obj_info_str += str(obj_number) + '. ' + obj.name + ', x_center=' + str(obj.x_center) + ', depth=' + '%.4f' % obj.depth + '\n'
            obj.number_mark = obj_number
            obj_number += 1
    prompt = prompt.replace('<OBJECT_INFORMATION>', obj_info_str)
    prompt = prompt.replace('<TABLE_X>', 'from ' + str(obj_table.x_min) + ' to ' + str(obj_table.x_max))
    prompt = prompt.replace('<TABLE_X_LEFT>', 'from ' + str(obj_table.x_min) + ' to ' + str(obj_table.x_center))
    prompt = prompt.replace('<TABLE_X_RIGHT>', 'from ' + str(obj_table.x_center + 1) + ' to ' + str(obj_table.x_max))
    
    cache_name = '_' + '_'.join(object_list[0].name.split()) + '_' + '_'.join(subsentence.split())
    if reference_object_list:
        cache_name += ('_' + '_'.join(reference_object_list[0].name.split()))
    response = load_cache('top_object' + cache_name, ARGS.cache_path, ID)
    if not response:
        response = llm_vision(ARGS.llm_top_object, prompt, IMAGE_PATH)
        if ARGS.flag_write_cache:
            store_cache('top_object' + cache_name, ARGS.cache_path, ID, response)
    object_position = extract_object_position(response)

    if not object_position:
        if response.startswith('I\'m sorry') or response.startswith('Sorry'):
            # repeat
            print('sleeping...')
            time.sleep(3)
            response = llm_vision(ARGS.llm_top_object, prompt, IMAGE_PATH)
            if ARGS.flag_write_cache:
                store_cache('top_object' + cache_name, ARGS.cache_path, ID, response)
            object_position = extract_object_position(response)
    if not object_position:
        if object_list:
            return [object_list[0]]
        else:
            return None

    # Step2.5: verify the generated top view
    flag_verification_object = True
    if flag_verification_object:
        if object_position:
            if reference_object_list:
                ver_obj_result = verification_object(object_list + reference_object_list, object_position)
            else:
                ver_obj_result = verification_object(object_list, object_position)
            
            if ver_obj_result == 1:
                pass
            else:
                prompt_r2 = "Your answer may be incorrect, because it does not match the following object relationships:\n" + ver_obj_result + "\n\nPlease reanalyze the position of the object after considering the above factors and output the final answer."
                response_r2 = None
                try:
                    response_r2 = llm_vision_round2(ARGS.llm_top_object, prompt, response, prompt_r2, IMAGE_PATH)
                except Exception as e:
                    print(ID)
                    print('Error in LLMs ...')
                if response_r2:
                    if ARGS.flag_write_cache:
                        store_cache('top_object_r2' + cache_name, ARGS.cache_path, ID, response_r2)
                    object_position_r2 = extract_object_position(response_r2)
                    if object_position_r2:
                        object_position = object_position_r2
    
    # Step3: determine horizontal relationships
    response = load_cache('top_reason' + cache_name, ARGS.cache_path, ID)
    if not response:

        keywords_left_right = ['left', 'right', 'leftmost', 'rightmost', 'left-handed', 'right-handed', 'left-hand', 'right-hand', 'Rightmost', 'Leftmost']
        keywords_middle = ['between', 'among', 'center', 'middle', 'central', 'medial']
        keywords_beside = ['beside', 'adjacent', 'next']
        keywords_left_right.extend(keywords_middle)
        keywords_left_right.extend(keywords_beside)

        keywords_front_back = ['front', 'close', 'closer', 'closest', 'nearest', 'near', 'back', 'further', 'farther', 'furthest', 'farthest', 'behind', 'rear']

        top_view_w_obj_img_path = ''
        if set(subsentence.split()).intersection(set(keywords_left_right)) and set(subsentence.split()).intersection(set(keywords_front_back)):
            top_view_w_obj_img_path = draw_top_view_w_obj_composition(object_position, person_position, ID, ARGS.cache_path)
            flag_relation = 2
        else:
            if set(subsentence.split()).intersection(set(keywords_left_right)):
                flag_relation = 0
            else:
                flag_relation = 1
            top_view_w_obj_img_path = draw_top_view_w_obj(object_position, person_position, flag_relation, ID, ARGS.cache_path)

        prompt = read_prompt(ARGS.prompt_path + '4_top_reason.txt')
        if flag_relation == 0:
            prompt = prompt.replace('<TARGET_DIRECTION>', 'left and right')
        elif flag_relation == 1:
            prompt = prompt.replace('<TARGET_DIRECTION>', 'front and back')
        elif flag_relation == 2:
            prompt = prompt.replace('<TARGET_DIRECTION>', 'left, right, front and back')

        letter_or_letters = ''
        if reference_object_list:
            if set(subsentence.split()).intersection(set(keywords_middle + keywords_beside)):
                letter_or_letters = 'letter'
            else:
                letter_or_letters = 'letters'
        else:
            letter_or_letters = 'letter'
        if CURRENT_LINE == COUNT_LINES:
            letter_or_letters = 'letter'
        prompt = prompt.replace('<LETTER_OR_LETTERS>', letter_or_letters)

        subsentence = subsentence.replace('my', 'the')
        subsentence = subsentence.replace('to me', '')
        subsentence = subsentence.replace('from me', '')
        spatial_phrase = ''
        if reference_object_list:
            if len(reference_object_list) == 1 and not isinstance(reference_object_list[0], list):
                spatial_phrase = subsentence + ' the letter ' + chr(reference_object_list[0].number_mark + 64)
            else:
                if subsentence in ['between', 'among']:
                    spatial_phrase = subsentence + ' the letters '
                    flag_and = 0
                    for i in reference_object_list:
                        if flag_and:
                            spatial_phrase += ' and '
                        tmp_letter_list = []
                        if isinstance(i, list):
                            for j in i:
                                tmp_letter_list += chr(j.number_mark + 64)
                            if len(tmp_letter_list) > 1:
                                tmp_letter_str = ','.join(tmp_letter_list)
                            else:
                                tmp_letter_str = tmp_letter_list[0]
                        else:
                            pass
                        spatial_phrase += tmp_letter_str
                        flag_and = 1
                else:
                    tmp_letter_list = []
                    for i in reference_object_list:
                        if isinstance(i, list):
                            for j in i:
                                tmp_letter_list += chr(j.number_mark + 64)
                        else:
                            tmp_letter_list += chr(i.number_mark + 64)
                    if len(tmp_letter_list) > 1:
                        tmp_letter_str = ','.join(tmp_letter_list)
                        spatial_phrase = subsentence + ' the letters '
                    else:
                        tmp_letter_str = tmp_letter_list[0]
                        spatial_phrase = subsentence + ' the letter '
                    spatial_phrase += tmp_letter_str
        else:
            spatial_phrase = subsentence
            if len(subsentence.split()) == 1:
                if subsentence in ['right', 'left']:
                    spatial_phrase = 'on the ' + subsentence
                elif subsentence in ['front', 'back']:
                    spatial_phrase = 'in ' + subsentence
        prompt = prompt.replace('<SPATIAL_RELATIONSHIP>', spatial_phrase)

        response = llm_vision(ARGS.llm_top_reason, prompt, top_view_w_obj_img_path)
        if ARGS.flag_write_cache:
            store_cache('top_reason' + cache_name, ARGS.cache_path, ID, response)

    results_letter = extract_horizontal_result(response)
    if not results_letter:
        if object_list:
            return [object_list[0]]
        else:
            return None
    results_letter2obj = []
    for i in results_letter:
        results_letter2num = ord(i) - 64
        for j in object_list:
            if j.number_mark == results_letter2num:
                results_letter2obj.append(j)
                break
    results = results_letter2obj
    
    return results

def execute_code(code):
    target_object = None
    g = {'target_object': target_object,
         'find_object': find_object, 
         'filter_by_vertical_relationship': filter_by_vertical_relationship, 
         'filter_by_horizontal_relationship': filter_by_horizontal_relationship}
    l = {}
    code_compile = compile(code, '', 'exec')
    exec(code_compile, g, l)
    target_object = l['target_object']
    return target_object

def grounding_steps(args, expression, image):

    global IMAGE_PATH
    IMAGE_PATH = image

    global ARGS
    ARGS = args

    global ID
    ID = '0'

    final_answer = [0, 0, 0, 0]
    
    response = load_cache('grounding', args.cache_path, ID)
    if not response:
        main_steps_prompt = read_prompt(args.prompt_path + '1_main.txt')
        prompt = main_steps_prompt.replace('<REFERRING_EXPRESSION>', expression)
        
        response = llm(args.llm_grounding, prompt)
        if args.flag_write_cache:
            store_cache('grounding', args.cache_path, ID, response)

    code = separate_code(response)

    global OBJ_RESULT
    OBJ_RESULT = visual_glip(code, image)
    store_cache('glip_results', args.cache_path, ID, str(OBJ_RESULT))

    global CURRENT_LINE
    CURRENT_LINE = 0
    global COUNT_LINES
    code_lines = [i for i in code.strip().split('\n') if not i == '' and not i.startswith('#')]
    COUNT_LINES = len(code_lines)

    final_answer = execute_code(code)

    if isinstance(final_answer, list):
        if len(final_answer) > 0:
            final_answer = final_answer[0].bbox
        else:
            pass
    else:
        final_answer = final_answer.bbox

    return final_answer
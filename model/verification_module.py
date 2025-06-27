def verification_person(obj_person, obj_table, person_position):
    flag_correct = 1
    if person_position in ['4', '5', '6', '7', '8', '9', '10']:
        if obj_person.x_center < obj_table.x_center:
            flag_correct = 0
    elif person_position in ['1', '2', '11', '12', '13', '14', '15']:
        if obj_person.x_center > obj_table.x_center:
            flag_correct = 0
    return flag_correct

def verification_object(object_list, object_position):
    object_dict = {x.number_mark: x for x in object_list}
    error_reason = []
    for k1, v1 in object_position.items():
        row1, column1 = v1
        for k2, v2 in object_position.items():
            if k2 == k1:
                continue
            row2, column2 = v2

            if ord(column1) < ord(column2):
                if object_dict[k1].x_center > object_dict[k2].x_center:
                    error_reason.append("Because the x_center of Object " + str(k1) + " is greater than the x_center of Object " + str(k2) + ", Object " + str(k1) + " may on the right side of Object " + str(k2) + ".")
            elif ord(column1) > ord(column2):
                if object_dict[k1].x_center < object_dict[k2].x_center:
                    error_reason.append("Because the x_center of Object " + str(k1) + " is less than the x_center of Object " + str(k2) + ", Object " + str(k1) + " may on the left side of Object " + str(k2) + ".")
            
            if row1 < row2:
                if object_dict[k1].depth < object_dict[k2].depth:
                    error_reason.append("Because the depth value of Object " + str(k1) + " is less than the depth value of Object " + str(k2) + ", Object " + str(k1) + " may below Object " + str(k2) + " in the top view.")
            elif row1 > row2:
                if object_dict[k1].depth > object_dict[k2].depth:
                    error_reason.append("Because the depth value of Object " + str(k1) + " is greater than the depth value of Object " + str(k2) + ", Object " + str(k1) + " may above Object " + str(k2) + " in the top view.")
    
    if len(error_reason) == 0:
        return 1
    else:
        error_reason_all = '\n'.join(error_reason)
        return error_reason_all
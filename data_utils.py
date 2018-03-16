import os
import numpy as np
def get_file_list(root_dir):
    """
        get all files in root_dir directory
    """
    path_list = []
    file_list = []
    join_list = []
    for path, _, files in os.walk(root_dir):
        for name in files:
            path_list.append(path)
            file_list.append(name)
            join_list.append(os.path.join(path, name))

    return path_list, file_list, join_list

def expand(number, width):
    s = np.zeros(width)
    s[number] = 1
    return s

def log_print(log_obj, info_str):
    print info_str
    if info_str[0] == '\r':
        log_obj.info(info_str[1:])
    else:
        log_obj.info(info_str)
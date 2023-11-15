import os


def up_direction(path):
    return os.path.dirname(path)


def get_full_path(relative_path):
    current_file_path = os.path.abspath(__file__)

    for _ in range(9):
        # current path goes the the top of the project
        current_file_path = up_direction(current_file_path)
    
    if relative_path[0] == "/":
        return current_file_path + relative_path
    else:
        return current_file_path + "/" + relative_path
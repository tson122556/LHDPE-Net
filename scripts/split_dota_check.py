import os
import numpy as np


def is_valid_label_line(line):
    try:
        parts = line.split()
        if len(parts) == 10:  # 根据DOTA的标签格式，调整此值
            [float(part) for part in parts]
            return True
    except ValueError:
        return False
    return False


def check_label_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        if not is_valid_label_line(line):
            print(f"Invalid line in file {file_path}: {line.strip()}")
            return False
    return True


def process_directory(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            if not check_label_file(file_path):
                print(f"File {file_path} contains invalid lines.")
            else:
                print(f"File {file_path} is valid.")


# 调用函数来检查标签文件所在的目录
label_directory = '/home/sdb/pk/datasets/DOTA/labels/train/'
process_directory(label_directory)

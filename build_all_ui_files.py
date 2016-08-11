from PyQt4 import uic
import os
script_path = os.path.realpath(__file__)
script_folder = os.path.dirname(script_path)


def remove_extension(filename):
    """return a the filename without the extension"""
    dot_occurrences = [n for n in xrange(len(filename)) if filename.find('.', n) == n]
    return filename[:dot_occurrences[-1]]


def is_ui_file(item):
    return item.endswith('.ui')


def get_output_file_name_item(filename):
    output_file_name = os.path.basename(filename)
    output_file_name = remove_extension(output_file_name)
    output_file_name += '_ui.py'
    return output_file_name


def build_item(item,verbose=True):
    dir = os.path.dirname(item)
    basename = os.path.basename(item)
    output_file_name = get_output_file_name_item(basename)
    output_file_path = os.path.join(dir,output_file_name)
    with open(output_file_path, 'w') as fout:
        uic.compileUi(item, fout)

    if verbose:
        print item, "==>", output_file_path


def build_all_ui_files(basepath):
    for item in os.listdir(basepath):
        if os.path.isdir(os.path.join(basepath,item)):
            build_all_ui_files(os.path.join(basepath,item))
        else:
            if is_ui_file(item):
                build_item(os.path.join(basepath,item))

if __name__ == '__main__':
    build_all_ui_files(script_folder)

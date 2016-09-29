from PyQt4 import uic
import os
script_path = os.path.realpath(__file__) # assumes script is in project root directory
script_folder = os.path.dirname(script_path)


def remove_extension(filename):
    """return the filename without the extension"""
    dot_occurrences = [n for n in xrange(len(filename)) if filename.find('.', n) == n]
    return filename[:dot_occurrences[-1]]


def is_ui_file(item):
    return item.endswith('.ui')


def get_output_file_name(filename):
    """Naming scheme foo.ui ==> foo_ui.py"""
    output_file_name = os.path.basename(filename)
    output_file_name = remove_extension(output_file_name)
    output_file_name += '_ui.py'
    return output_file_name


def build_item(item, verbose=True):
    """Build the ui file at the path in "item" to python file in the same directory.
     The output file name is specified by the get_output_file_name function"""
    dir = os.path.dirname(item)
    basename = os.path.basename(item)
    output_file_name = get_output_file_name(basename)
    output_file_path = os.path.join(dir,output_file_name)
    with open(output_file_path, 'w') as fout:
        uic.compileUi(item, fout)

    if verbose:
        print item, "==>", output_file_path


def build_all_ui_files(basepath, verbose=True):
    """Build all ui files found in folder basepath"""
    for item in os.listdir(basepath):
        if os.path.isdir(os.path.join(basepath,item)):
            build_all_ui_files(os.path.join(basepath,item), verbose=verbose)
        else:
            if is_ui_file(item):
                build_item(os.path.join(basepath,item), verbose=verbose)

if __name__ == '__main__':
    # assumes script is in project root directory
    build_all_ui_files(script_folder)

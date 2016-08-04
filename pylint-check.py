PYLINT_PYTHON = r'C:\Users\OMi32458\Workspace\mantid\external\src\ThirdParty\lib\python2.7\python.exe'
WORKING_DIR = r'C:\MantidInstall\bin'

import os,subprocess,sys
script_path = os.path.realpath(__file__)
script_folder = os.path.dirname(script_path)
os.chdir(WORKING_DIR)


def remove_extension(filename):
    """Remove the file extension"""
    dot_occurences = [n for n in xrange(len(filename)) if filename.find('.', n) == n]
    return filename[:dot_occurences[-1]]


def is_code(item):
    return item.endswith('.py') and not item.endswith('_ui.py')


def get_output_file_name_item(filename):
    output_file_name = os.path.basename(filename)
    output_file_name = remove_extension(output_file_name)
    output_file_name += '_ui.py'
    return output_file_name


def pylint_item(item):
    try:
        out = subprocess.call(['python','-m','pylint','-E',item])
        return
        for line in out.split('\n'):
            if 'PyQt4' in line:
                pass #continue
            if 'No config file found, using default configuration' in line:
                continue
            print line
    except Exception as e:
        print e
        pass


def pylint_all_files(basepath):
    for item in os.listdir(basepath):
        if os.path.isdir(os.path.join(basepath,item)):
            pylint_all_files(os.path.join(basepath, item))
        else:
            if is_code(item):
                pylint_item(os.path.join(basepath,item))

if __name__ == '__main__':
    pylint_all_files(script_folder)
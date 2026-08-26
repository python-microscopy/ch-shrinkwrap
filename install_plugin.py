from PYME import config
import os
import sys
import shutil

def main():
    this_dir = os.path.dirname(os.path.abspath(__file__))

    try:
        if sys.argv[1] == 'dist':
            shutil.copytree(os.path.join(this_dir, '_etc', 'PYME'), config.dist_config_directory, dirs_exist_ok=True)
    except IndexError:  # no argument provided, default to user config directory
        shutil.copytree(os.path.join(this_dir, '_etc', 'PYME'), config.user_config_dir, dirs_exist_ok=True)

if __name__ == '__main__':
    main()
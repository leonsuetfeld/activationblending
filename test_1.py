
import subprocess

command = 'python3 test_2.py -l 1234 2345 3456 4567 -m 5.1 3.4 -n lalala -b True -c False -i 5 -f 1.3'
subprocess.run(command, shell=True)

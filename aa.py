import subprocess
import sys

def main():
    args = sys.argv[1:]
    cmd = 'mpiexec -n 4 ps '+' '.join(args)
    print(cmd)
    print(subprocess.call(cmd, shell=True))

if __name__ == '__main__':
    main()
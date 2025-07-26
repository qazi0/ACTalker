import os
import sys
import signal
import pdb
def kill_process(*pids):
  for pid in pids:
    a = os.kill(pid, signal.SIGKILL)
    print('Killed process with pid %s, return value: %s' % (pid, a))

def get_pid(*ports):
	# The \" is an escape character for "
    pids = []
    print(ports)
    for port in ports:
        msg = os.popen('lsof -i:{}'.format(port)).read()
        msg = msg.split('\n')[1:-1]
        for m in msg:
            try:
                m = m.replace('  ', ' ')
                m = m.replace('  ', ' ')
                tokens = m.split(' ')
                pids.append(int(tokens[1]))
            except Exception as e:
                print(e)
    return pids

if __name__ == "__main__":
    # Kill processes occupying the port numbers
    ports = sys.argv[1:]
    kill_process(*get_pid(*ports))
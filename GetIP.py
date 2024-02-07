import subprocess
import re

def find_device(ip_address, mac, mac2):
    command = ["nmap", "-sn", ip_address]
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    ris, err = process.communicate()
    output=ris.decode('utf-8')
    print("Ip del primo drone: ")
    getIP(output, mac)
    print("Ip del secondo drone: ")
    getIP(output, mac2)


def getIP(text, pattern):
    lines = text.split('\n')
    match_index = None
    for i, line in enumerate(lines):
        if pattern in line:
            match_index = i
            break
    if match_index is not None and match_index >= 2:
        print(lines[match_index - 2])




if __name__ == "__main__":
    ip_address = "192.168.1.0/24"
    mac1 = "9C:50:D1:3B:5B:94"
    mac2="9C:50:D1:3B:54:08"
    out=find_device(ip_address,mac1, mac2)
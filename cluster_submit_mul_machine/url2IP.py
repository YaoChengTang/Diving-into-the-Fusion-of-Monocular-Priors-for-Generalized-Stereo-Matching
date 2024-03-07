"""
Tools to convert default hostfile on aidi cluster.
"""
import socket


def URL2IP():
    for oneurl in urllist.readlines():
        if not oneurl.strip():
            continue
        url = str(oneurl.strip())
        try:
            ip = socket.gethostbyname(url)
            print(ip)
            iplist.writelines(str(ip) + "\n")
        except Exception:
            print("this URL 2 IP ERROR ")


try:
    urllist = open("/job_data/hosts", "r")
    iplist = open("/job_data/mpi_hosts", "w")
    URL2IP()
    urllist.close()
    iplist.close()
    print("complete !")
except Exception:
    print("ERROR !")
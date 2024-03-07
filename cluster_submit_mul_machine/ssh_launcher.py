"""
Tools to launch multi-machine multi-gpus training on aidi cluster.
"""
import argparse
import logging
import os
import random
import socket
import subprocess
from functools import partial
from threading import Thread

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def parse_hostfile(hostfile, convert_ip=False):
    hosts = []
    with open(hostfile) as f:
        for h in f.readlines():
            h = h.strip()
            if len(h) > 0:
                if not convert_ip:
                    hosts.append(h.strip())
                else:
                    try:
                        ip = socket.gethostbyname(h)
                        hosts.append(ip)
                    except Exception as e:
                        print("error host ", h, " error: ", e)
    return hosts


def find_available_ports(ip, port_num=1, port=9091, port_end=9999):
    max_retry = 100

    port_list = []
    for i in range(0, max_retry):
        if i + 1 == max_retry:
            raise Exception("faild to bind a port")
        port = random.randint(port, port_end)
        print("local port ", port)
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(("", port))
            port_list.append(port)
            print("success port ", port)
            sock.close()
            if len(port_list) == port_num:
                break
        except socket.error:
            print("error port ", port)
            continue
    return port_list


def get_env(pass_envs):
    envs = []
    for k, v in list(pass_envs.items()):
        if "()" in k or "()" in v:
            continue
        envs.append("export " + str(k) + "=" + str(v) + ";")
    return " ".join(envs)


def run(cmd, node, exit_if_error: bool = True):
    logger.info(f"launch on node {node}: {cmd}")
    try:
        subprocess.check_call(
            cmd,
            shell=True,
        )
    except subprocess.CalledProcessError as e:
        logger.warning(
            f"subprocess({e.cmd}) failed({e.returncode})! {e.output}.\n"
        )
        if exit_if_error:
            os._exit(-1)


def apply_on_multi_node_by_ssh(
    prog_func,
    nworker,
    hosts,
    cmd,
    pass_envs,
    working_dir,
):
    thread_list = []
    for i in range(nworker):
        node = hosts[i % len(hosts)]
        pass_envs["NODE_RANK"] = str(i % len(hosts))

        # ssh_port_arg = " -p " + str(sshport) + " "
        prog = get_env(pass_envs) + " cd " + working_dir + "; " + cmd
        prog = (
            "ssh -o StrictHostKeyChecking=no "
            # + ssh_port_arg   # no port available in aidi
            + node
            + " '"
            + prog
            + "'"
        )
        thread = Thread(target=prog_func, args=(prog, node))
        thread.setDaemon(True)
        thread.start()
        thread_list.append(thread)

    for t in thread_list:
        t.join()
        print("thread join success")


def parse_hosts_and_ports(nworker, hostfile, port):
    ip_ports = []
    if hostfile is not None:
        hosts = parse_hostfile(hostfile)
        ip_ports = [h + ":" + str(port) for h in hosts]
    else:
        local_host = "127.0.0.1"
        hosts = [local_host] * nworker
        ports = find_available_ports(local_host, nworker)
        for p in ports:
            ip_ports.append(local_host + ":" + str(p))

    return hosts, ip_ports


def submit(
    nworker,
    ngpus,
    hostfile,
    port,
    sshport,
    cmd,
    check=False,
    monitor=False,
):
    if nworker <= 0:
        return

    hosts, ip_ports = parse_hosts_and_ports(nworker, hostfile, port)

    local_dir = os.getcwd() + "/"

    pass_envs = os.environ.copy()
    if "MASTER_ADDR" not in os.environ:
        pass_envs["MASTER_ADDR"] = hosts[0]
    if "MASTER_PORT" not in os.environ:
        pass_envs["MASTER_PORT"] = str(port)

    if check:
        logger.info("=" * 50 + "SSH LAUNCHER PRE CHECK" + "=" * 50)
        check_cmd = f"python3 nccl_check.py --driver --ngpus {ngpus} --nccl"
        apply_on_multi_node_by_ssh(
            prog_func=partial(run, exit_if_error=False),
            nworker=nworker,
            hosts=hosts,
            cmd=check_cmd,
            pass_envs=pass_envs,
            working_dir=local_dir,
        )
        logger.info("=" * 50 + "SSH LAUNCHER END PRE CHECK" + "=" * 50)

    # ssh for multi-node launch
    apply_on_multi_node_by_ssh(
        prog_func=run,
        nworker=nworker,
        hosts=hosts,
        cmd=cmd,
        pass_envs=pass_envs,
        working_dir=local_dir,
    )

    print("process end success")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-num_machines",
        "--nworker",
        type=int,
        required=True,
        help="number of worker process to be launched",
    )
    parser.add_argument(
        "-num_gpus",
        "--ngpus",
        type=int,
        required=True,
        help="number of gpus on per node",
    )
    parser.add_argument(
        "-host",
        "--hostfile",
        type=str,
        help="the hostfile of workers",
    )
    parser.add_argument(
        "-ports",
        "--port",
        type=int,
        default=9876,
        help="the port used for every worker, used when distribute-training",
    )
    parser.add_argument(
        "-sp",
        "--sshport",
        type=int,
        default=443,
        help="the port used for ssh connect, used when distribute-training",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        default=False,
        help="whether to do gpu driver and nccl-test check",
    )
    parser.add_argument(
        "--monitor",
        action="store_true",
        default=False,
        help="whether to use watchdog to monitor",
    )
    parser.add_argument(
        "command", nargs="+", help="command for plugin program"
    )
    args = parser.parse_args()
    cmd = " ".join(args.command)
    submit(
        nworker=args.nworker,
        ngpus=args.ngpus,
        hostfile=args.hostfile,
        port=args.port,
        sshport=args.sshport,
        cmd=cmd,
        check=args.check,
        monitor=args.monitor,
    )
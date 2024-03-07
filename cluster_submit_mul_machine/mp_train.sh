# 1. setup environments
echo "======================= Set Environment Variables ======================"
# export HDFS_URL="hdfs://hobot-bigdata-ucloud"
export OUTDIR="/job_data"
export LOG_ROOT="/job_log"
export TB_ROOT="/job_tboard"
export CKPOINT_ROOT="/job_data"

# 2. copy code
echo "============================== save code to /job_data ===================================="
cd ${WORKING_PATH}
echo "Current IP: $(hostname -I)"

python3 cluster_submit_mul_machine/url2IP.py

cat /job_data/mpi_hosts
dis_url=$(head -n +1 /job_data/mpi_hosts)

python3 cluster_submit_mul_machine/ssh_launcher.py -num_machines 2 -num_gpus 4 \
    -host /job_data/mpi_hosts -ports 8000 bash cluster_submit_mul_machine/train_stereo_cloud_MulIP.sh
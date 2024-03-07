# /usr/bin/bash

cd $WORKING_PATH

echo ${MPI_SUBMIT}
which ${MPI_SUBMIT}

echo "FIRST"
hostname -I
echo "FIRST COMPLETE"

python3 ./cluster_submit_mul_machine/url2IP.py

${MPI_SUBMIT} sh -x ${WORKING_PATH}/train_stereo_cloud_MulIP.sh


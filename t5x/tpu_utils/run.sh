# When working with pods, one has to send command to all tpus workers
TPU_NAME=$1
OMMAND=$2
ZONE=$3

echo $COMMAND

# TODO: wrap this in tmux in order for command not to be killed upon lost of ssh connection.
gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} --zone ${ZONE} --worker=all --command="$COMMAND" -- -t

# Example to run t5_c4_span_corruption
# sh bigscience/scripts/run_on_all_vms.sh thomas "cd code/t5x; git pull; sh bigscience/scripts/launch_command_in_tmux.sh \"\$(cat bigscience/scripts/pretrain_t5_c4_span_corruption.sh)\""

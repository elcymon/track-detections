
# Use current working directory and current modules
#$ -cwd -V

#$ -e /nobackup/scsoo/logs/errors
#$ -o /nobackup/scsoo/logs/outputs

# Request a full node (24 cores and 128 GB or 768 GB on ARC3)
# -l nodes=0.1
# -l node_type=24core-128G

# To request for x cores on a single machine, with around y memory per core
# -pe smp x -l h_vmem=yG

#memory?
#$ -l h_vmem=30G

#no of cores
#$ -pe smp 2

# Request Wallclock time of hh:mm:ss
#$ -l h_rt=10:0:0

#Iterations
#$ -t 1-1

#Iterations in batch of
#$ -tc 1

#e-mail
#$ -m ae
#$ -M scsoo@leeds.ac.uk


task=$1
hpc=$2
if ((hpc)); then
    module load singularity
fi
singularity exec --bind ../:/mnt ../opencv-pandas-ffmpeg.simg bash -c "cd /mnt/track-detections && python3 $task"
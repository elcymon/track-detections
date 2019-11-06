task=$1
network=$2
hpc=$3
if ((hpc)); then
    Command=qsub
else
    Command=bash
fi
for vid in $(cd ../data/mp4; ls *.MP4); do
    #prepare path to save results
    echo $vid
    vidname=${vid%.*}
    resultPath=../data/model_data/$vidname/$network
    videoPath=../data/mp4
    mkdir -p $resultPath

    #do a qsub with task and network

    $Command exec_file.sh "$task $network $vidname False" $hpc
done
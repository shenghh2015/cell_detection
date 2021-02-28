cd /home/shenghuahe/wb_cells/cell_detection/EfficientDet/
python2 job_parser.py 'multi_GPU.sh'
for i in $(seq 0 3)
do
   sh job_folder/job_$i.sh&
   sleep 30s &
done
wait

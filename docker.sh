chcon -Rt svirt_sandbox_file_t /shared2/Data_FDA_Breast/Detection
docker run --gpus 0 -v /shared2/Data_FDA_Breast/Detection:/data -w /data/cell_detection -it --user $(id -u):$(id -g) shenghh2020/tf_gpu_py3.5:2.0 bash

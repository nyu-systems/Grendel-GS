

expe_folder_base=$1 # specify the folder to save these experiments log and checkpoints
dataset_folder=$2 # specify the dataset folder

# Train all scenes on single GPU with batch size 1
bash examples/mip360_4k/1g_1b.sh $expe_folder_base $dataset_folder

# Train all scenes on 4 GPU distributed with batch size 1
bash examples/mip360_4k/4g_1b.sh $expe_folder_base $dataset_folder

# analyze the results from logs and generate the result table
python examples/mip360_4k/analyze_results.py $expe_folder_base
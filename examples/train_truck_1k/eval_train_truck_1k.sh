

expe_folder_base=$1 # specify the folder to save these experiments log and checkpoints
dataset_folder=$2 # specify the dataset folder

# Train all scenes on single GPU with batch size 1
bash examples/train_truck_1k/train_truck_1k.sh $expe_folder_base ${dataset_folder}/tandt

# analyze the results from logs and generate the result table
python examples/train_truck_1k/analyze_results.py $expe_folder_base
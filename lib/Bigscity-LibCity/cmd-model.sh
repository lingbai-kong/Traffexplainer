echo "Run Model Training Batch..."

for dataset in PEMSD4 PEMSD8
do
    echo "Dataset: ${dataset}"
    for model in GTS
    do
        echo "Training ${model}..."
        python run_model.py --task=traffic_state_pred --model=${model} --base_model=${model} --dataset=${dataset} --batch_size=128 --max_epoch=50 --learning_rate=0.001 --gpu_id=3 --saved_model=True > cmd_log/${dataset}-${model}.log 2>&1
    done
done

echo "Finish"
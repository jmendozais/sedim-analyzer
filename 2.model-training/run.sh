#python preprocess_data.py -i /home/juliomb/d5_data/images/ -o dataset_test/
#python kfold.py -i dataset -o skfold_5_v2 -e "png" -y /home/juliomb/d5_data/Etiquetas/etiquetado_entrenamiento_2.csv

epoch_budget=1
model=densenet121 #densenet201

# mean ACC 0.45 aprox
#python train.py -d dataset -s skfold_5 -o results --n_epochs $epoch_budget --train_mode fixed-feats --model_name $model --learning_rate=0.0001 >"${model}_lr0.0001_fixed_feats_ep${epoch_budget}_dataaug.txt"

#python train.py -d dataset -s skfold_5 -o results --n_epochs $epoch_budget --train_mode fixed-feats --model_name $model --learning_rate=0.0001 --data_augmentation advanced >"${model}_lr0.0001_fixed_feats_ep${epoch_budget}_dataaug.txt"

# mean ACC 0.57 
#python train.py -d dataset -s skfold_5 -o results --n_epochs $epoch_budget --train_mode transfer-learning --model_name $model --learning_rate=0.0001 >"${model}_lr0.0001_unfreeze_allparams_ep$epoch_budget.txt"

#python train.py -d dataset -s skfold_5 -o results --n_epochs $epoch_budget --train_mode downscale_lr --model_name $model --learning_rate=0.0001 >"${model}_lr0.0001_dslr_ep$epoch_budget.txt"

#python train.py -d dataset -s skfold_5 -o results --n_epochs $epoch_budget --train_mode fixed-feats --model_name $model --learning_rate=0.001 >"${model}_lr0.001_fixed_feats_ep$epoch_budget.txt"

#python train.py -d dataset -s skfold_5 -o results --n_epochs $epoch_budget --train_mode fixed-feats --model_name $model --learning_rate=0.001 --data_augmentation advanced >"${model}_lr0.001_fixed_feats_ep${epoch_budget}_dataaug.txt"

# mean ACC 0.579
#model_tag="${model}_lr0.0001_unfreeze_all_ep${epoch_budget}_dataaug"
#python train.py -d dataset -s skfold_5 -o $model_tag --n_epochs $epoch_budget --train_mode transfer-learning --model_name $model --learning_rate=0.0001 --data_augmentation advanced >${model_tag}.txt

#python train.py -d dataset -s skfold_5 -o results --n_epochs $epoch_budget --train_mode transfer-learning --model_name $model --learning_rate=0.001 --data_augmentation advanced >"${model}_lr0.001_unfreeze_all_ep${epoch_budget}_dataaug.txt"
#python train.py -d dataset -s skfold_5 -o results --n_epochs $epoch_budget --train_mode transfer-learning --model_name $model --learning_rate=0.01 --data_augmentation advanced >"${model}_lr0.01_unfreeze_all_ep${epoch_budget}_dataaug.txt"

# mean acc 0.55
#model=resnet152 #densenet201
#python train.py -d dataset -s skfold_5 -o results --n_epochs $epoch_budget --train_mode transfer-learning --model_name $model --learning_rate=0.0001 >"${model}_lr0.0001_unfreeze_allparams_ep$epoch_budget.txt"

# Train model for evaluation
#model_tag="${model}_lr0.0001_unfreeze_all_ep${epoch_budget}_dataaug_complete"
#python train.py --model_evaluation -d dataset -s skfold_5 -o $model_tag --n_epochs $epoch_budget --train_mode transfer-learning --model_name $model --learning_rate=0.0001 --data_augmentation advanced >${model_tag}.txt

# INFERENCE
model='densenet121_lr0.0001_unfreeze_all_ep1_dataaug_complete/model_convnet_complete.model'
python inference.py --input_dir 'example_test_set' --model_file $model --output_file 'example_output.csv'

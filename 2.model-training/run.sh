#python preprocess_data.py -i /home/juliomb/d5_data/images/ -o dataset_test/
#python kfold.py -i dataset -o skfold_5_v2 -e "png" -y /home/juliomb/d5_data/Etiquetas/etiquetado_entrenamiento_2.csv


#python train.py -d dataset -s skfold_5 -o results --n_epochs 20 --train_mode fixed-feats --model_name densenet201 --learning_rate=0.0001 

epoch_budget=30
model=densenet121 #densenet201

# mean ACC 0.45 aprox
#python train.py -d dataset -s skfold_5 -o results --n_epochs $epoch_budget --train_mode fixed-feats --model_name $model --learning_rate=0.0001 >"${model}_lr0.0001_fixed_feats_ep${epoch_budget}_dataaug.txt"

#python train.py -d dataset -s skfold_5 -o results --n_epochs $epoch_budget --train_mode fixed-feats --model_name $model --learning_rate=0.0001 --data_augmentation advanced >"${model}_lr0.0001_fixed_feats_ep${epoch_budget}_dataaug.txt"

# mean ACC 0.57 
#python train.py -d dataset -s skfold_5 -o results --n_epochs $epoch_budget --train_mode transfer-learning --model_name $model --learning_rate=0.0001 >"${model}_lr0.0001_unfreeze_allparams_ep$epoch_budget.txt"

#python train.py -d dataset -s skfold_5 -o results --n_epochs $epoch_budget --train_mode downscale_lr --model_name $model --learning_rate=0.0001 >"${model}_lr0.0001_dslr_ep$epoch_budget.txt"

#python train.py -d dataset -s skfold_5 -o results --n_epochs $epoch_budget --train_mode fixed-feats --model_name $model --learning_rate=0.001 >"${model}_lr0.001_fixed_feats_ep$epoch_budget.txt"

#python train.py -d dataset -s skfold_5 -o results --n_epochs $epoch_budget --train_mode fixed-feats --model_name $model --learning_rate=0.001 --data_augmentation advanced >"${model}_lr0.001_fixed_feats_ep${epoch_budget}_dataaug.txt"

#python train.py -d dataset -s skfold_5 -o results --n_epochs $epoch_budget --train_mode transfer-learning --model_name $model --learning_rate=0.001 --data_augmentation advanced >"${model}_lr0.001_unfreeze_all_ep${epoch_budget}_dataaug.txt"

#python train.py -d dataset -s skfold_5 -o results --n_epochs $epoch_budget --train_mode transfer-learning --model_name $model --learning_rate=0.0001 --data_augmentation advanced >"${model}_lr0.0001_unfreeze_all_ep${epoch_budget}_dataaug.txt"

#python train.py -d dataset -s skfold_5 -o results --n_epochs $epoch_budget --train_mode transfer-learning --model_name $model --learning_rate=0.01 --data_augmentation advanced >"${model}_lr0.01_unfreeze_all_ep${epoch_budget}_dataaug.txt"


model=resnet152 #densenet201
python train.py -d dataset -s skfold_5 -o results --n_epochs $epoch_budget --train_mode transfer-learning --model_name $model --learning_rate=0.0001 >"${model}_lr0.0001_unfreeze_allparams_ep$epoch_budget.txt"

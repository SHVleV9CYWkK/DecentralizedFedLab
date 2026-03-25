#!/bin/bash
#Cifar10
#echo cifar10 dfedcad
#python main.py --fl_method dfedcad --dataset_name cifar10 --alpha 0.4 --model lenet --local_epochs 1 --lr 0.0005 --batch_size 32 --n_rounds 50 --minimum_join_rounds 25 --optimizer_name adam --seed 42 --device cuda --split_method dirichlet --set_single_delay_client 81 --lambda_alignment 10 --base_decay_rate 0.1
#echo cifar10 dfedavg
#python main.py --fl_method dfedavg --dataset_name cifar10 --alpha 0.4 --model lenet --local_epochs 1 --lr 0.001 --batch_size 32 --n_rounds 50 --minimum_join_rounds 25 --optimizer_name adam --seed 42 --device cuda --split_method dirichlet --set_single_delay_client 81
#echo cifar10 dfedmtkdrl
#python main.py --fl_method dfedmtkdrl --dataset_name cifar10 --alpha 0.4 --model lenet --local_epochs 1 --lr 0.001 --batch_size 32 --n_rounds 50 --minimum_join_rounds 25 --optimizer_name adam --seed 42 --device cpu --split_method dirichlet --set_single_delay_client 81 --lambda_alignment 0.1

#EMNIST
#echo emnist dfedcad
#python main.py --fl_method dfedcad --dataset_name emnist --alpha 0.4 --model leafcnn1 --local_epochs 1 --lr 0.0005 --batch_size 32 --n_rounds 50 --minimum_join_rounds 25 --optimizer_name adam --seed 42 --device cuda --split_method dirichlet --set_single_delay_client 81 --lambda_alignment 10 --base_decay_rate 0.1
#echo emnist dfedavg
#python main.py --fl_method dfedavg --dataset_name emnist --alpha 0.4 --model leafcnn1 --local_epochs 1 --lr 0.001 --batch_size 32 --n_rounds 50 --minimum_join_rounds 25 --optimizer_name adam --seed 42 --device cuda --split_method dirichlet --set_single_delay_client 81
#echo emnist dfedmtkdrl
#python main.py --fl_method dfedmtkdrl --dataset_name emnist --alpha 0.4 --model lenet --local_epochs 1 --lr 0.001 --batch_size 32 --n_rounds 50 --minimum_join_rounds 25 --optimizer_name adam --seed 42 --device cpu --split_method dirichlet --set_single_delay_client 81 --lambda_kd 0.1


#Cifar100
#echo cifar100 dfedcad
#python main.py --fl_method dfedcad --dataset_name cifar100 --alpha 0.4 --model resnet18 --local_epochs 1 --lr 0.0005 --batch_size 32 --n_rounds 50 --minimum_join_rounds 25 --optimizer_name adam --seed 42 --device cuda --split_method dirichlet --set_single_delay_client 40 --lambda_alignment 10 --base_decay_rate 0.1
#echo cifar100 dfedavg
#python main.py --fl_method dfedavg --dataset_name cifar100 --alpha 0.4 --model resnet18 --local_epochs 1 --lr 0.001 --batch_size 32 --n_rounds 50 --minimum_join_rounds 25 --optimizer_name adam --seed 42 --device cuda --split_method dirichlet --set_single_delay_client 40
#echo cifar100 dfedmtkdrl
#python main.py --fl_method dfedmtkdrl --dataset_name cifar100 --alpha 0.4 --model lenet --local_epochs 1 --lr 0.001 --batch_size 32 --n_rounds 50 --minimum_join_rounds 25 --optimizer_name adam --seed 42 --device cpu --split_method dirichlet --set_single_delay_client 40 --lambda_kd 0.1

#tiny_imagenet
#echo tiny_imagenet dfedcad
#python main.py --fl_method dfedcad --dataset_name tiny_imagenet --alpha 0.4 --model resnet18 --local_epochs 1 --lr 0.0005 --batch_size 32 --n_rounds 50 --minimum_join_rounds 25 --optimizer_name adam --seed 42 --device cuda --split_method dirichlet --set_single_delay_client 40 --lambda_alignment 10 --base_decay_rate 0.1
#echo tiny_imagenet dfedavg
#python main.py --fl_method dfedavg --dataset_name tiny_imagenet --alpha 0.4 --model resnet18 --local_epochs 1 --lr 0.001 --batch_size 32 --n_rounds 50 --minimum_join_rounds 25 --optimizer_name adam --seed 42 --device cuda --split_method dirichlet --set_single_delay_client 40
#echo tiny_imagener dfedkdrl
#python main.py --fl_method dfedmtkdrl --dataset_name tiny_imagenet --alpha 0.4 --model lenet --local_epochs 1 --lr 0.001 --batch_size 32 --n_rounds 50 --minimum_join_rounds 25 --optimizer_name adam --seed 42 --device cpu --split_method dirichlet --set_single_delay_client 40 --lambda_kd 0.1


# Ablation studies
#echo cifar100 dfedcad lambda_alignment 0
#python main.py --fl_method dfedcad --dataset_name cifar100 --alpha 0.4 --model resnet18 --local_epochs 1 --lr 0.0005 --batch_size 32 --n_rounds 50 --minimum_join_rounds 25 --optimizer_name adam --seed 42 --device cuda --split_method dirichlet --set_single_delay_client 40 --base_decay_rate 0.1 --lambda_alignment 0

#echo tiny_imagener dfedkdrl lambda_alignment 0
#python main.py --fl_method dfedcad --dataset_name tiny_imagenet --alpha 0.4 --model resnet18 --local_epochs 1 --lr 0.0005 --batch_size 32 --n_rounds 50 --minimum_join_rounds 25 --optimizer_name adam --seed 42 --device cuda --split_method dirichlet --set_single_delay_client 40 --lambda_alignment 0 --base_decay_rate 0.1

new_client_ids=(69 53 26 87 7 98 39 1 74 81)
#alphas=(0.4 1.0)
alpha=1.0

for client_id in "${new_client_ids[@]}"
do
#  echo "Running with client_id $client_id in Cifar10 alpha $alpha"
#  echo "dfedcad"
#  python main.py --fl_method dfedcad --dataset_name cifar10 --alpha $alpha --model lenet --local_epochs 1 --lr 0.0005 --batch_size 32 --n_rounds 50 --minimum_join_rounds 25 --optimizer_name adam --seed 42 --device cuda --split_method dirichlet --lambda_alignment 10 --base_decay_rate 0.1 --set_single_delay_client $client_id
#  echo "dfedavg"
#  python main.py --fl_method dfedavg --dataset_name cifar10 --alpha $alpha --model lenet --local_epochs 1 --lr 0.001 --batch_size 32 --n_rounds 50 --minimum_join_rounds 25 --optimizer_name adam --seed 42 --device cuda --split_method dirichlet --set_single_delay_client $client_id
#  echo "dfedmtkdrl"
#  python main.py --fl_method dfedmtkdrl --dataset_name cifar10 --alpha $alpha --model lenet --local_epochs 1 --lr 0.001 --batch_size 32 --n_rounds 50 --minimum_join_rounds 25 --optimizer_name adam --seed 42 --device cuda --split_method dirichlet  --lambda_alignment 0.1 --set_single_delay_client $client_id
#  echo "dfedpgp"
#  python main.py --fl_method dfedpgp --dataset_name cifar10 --alpha $alpha --model lenet --local_epochs 1 --lr 0.0005 --batch_size 32 --n_rounds 50 --minimum_join_rounds 25 --optimizer_name adam --seed 42 --device cuda --split_method dirichlet  --set_single_delay_client $client_id
#  echo "dfedsam "
#  python main.py --fl_method dfedsam --dataset_name cifar10 --alpha $alpha --model lenet --local_epochs 1 --lr 0.0005 --batch_size 32 --n_rounds 50 --minimum_join_rounds 25 --optimizer_name adam --seed 42 --device cuda --split_method dirichlet  --set_single_delay_client $client_id
#  echo  "dfedgo"
#  python main.py --fl_method fedgo --dataset_name cifar10 --alpha $alpha --model lenet --local_epochs 1 --lr 0.0005 --batch_size 32 --n_rounds 50 --minimum_join_rounds 25 --optimizer_name adam --seed 42 --device cuda --split_method dirichlet --set_single_delay_client $client_id
#  echo  "qfedcg"
#  python main.py --fl_method qfedcg --dataset_name cifar10 --alpha $alpha --model lenet --local_epochs 1 --lr 0.0005 --batch_size 32 --n_rounds 50 --minimum_join_rounds 25 --optimizer_name adam --seed 42 --device cuda --split_method dirichlet --set_single_delay_client $client_id
#  echo "dfedcad_without_alignment"
#  python main.py --fl_method dfedcad --dataset_name cifar10 --alpha $alpha --model lenet --local_epochs 1 --lr 0.0005 --batch_size 32 --n_rounds 50 --minimum_join_rounds 25 --optimizer_name adam --seed 42 --device cuda --split_method dirichlet --lambda_alignment 0 --base_decay_rate 0.1 --set_single_delay_client $client_id
#  for alpha in "${alphas[@]}"
#  do
    echo retfhd
    python main.py --fl_method retfhd --dataset_name cifar10 --alpha $alpha --model lenet --local_epochs 1 --lr 0.0001 --batch_size 32 --n_rounds 50 --minimum_join_rounds 25 --optimizer_name adam --seed 42 --device cuda --split_method dirichlet --set_single_delay_client $client_id
  done
done


#echo "dfedpgp"
#python main.py --fl_method dfedpgp --dataset_name emnist --alpha 0.4 --model leafcnn1 --local_epochs 1 --lr 0.0005 --batch_size 32 --n_rounds 50 --minimum_join_rounds 25 --optimizer_name adam --seed 42 --device cuda --split_method dirichlet  --set_single_delay_client 69

#new_client_ids=(74 81)
#for client_id in "${new_client_ids[@]}"
#do
#  echo "Running with client_id $client_id in EMNIST"
#  echo "dfedcad"
#  python main.py --fl_method dfedcad --dataset_name emnist --alpha 0.4 --model leafcnn1 --local_epochs 1 --lr 0.0005 --batch_size 32 --n_rounds 50 --minimum_join_rounds 25 --optimizer_name adam --seed 42 --device cuda --split_method dirichlet --lambda_alignment 10 --base_decay_rate 0.1 --set_single_delay_client $client_id
#  echo "dfedavg"
#  python main.py --fl_method dfedavg --dataset_name emnist --alpha 0.4 --model leafcnn1 --local_epochs 1 --lr 0.001 --batch_size 32 --n_rounds 50 --minimum_join_rounds 25 --optimizer_name adam --seed 42 --device cuda --split_method dirichlet --set_single_delay_client $client_id
#  echo "dfedmtkdrl"
#  python main.py --fl_method dfedmtkdrl --dataset_name emnist --alpha 0.4 --model leafcnn1 --local_epochs 1 --lr 0.001 --batch_size 32 --n_rounds 50 --minimum_join_rounds 25 --optimizer_name adam --seed 42 --device cuda --split_method dirichlet --lambda_kd 0.1 --set_single_delay_client $client_id
#  echo "dfedpgp"
#  python main.py --fl_method dfedpgp --dataset_name emnist --alpha 0.4 --model leafcnn1 --local_epochs 1 --lr 0.0005 --batch_size 32 --n_rounds 50 --minimum_join_rounds 25 --optimizer_name adam --seed 42 --device cuda --split_method dirichlet  --set_single_delay_client $client_id
#  echo "dfedsam "
#  python main.py --fl_method dfedsam --dataset_name emnist --alpha 0.4 --model leafcnn1 --local_epochs 1 --lr 0.0005 --batch_size 32 --n_rounds 50 --minimum_join_rounds 25 --optimizer_name adam --seed 42 --device cuda --split_method dirichlet  --set_single_delay_client $client_id
#  echo "fedgo"
#  python main.py --fl_method fedgo --dataset_name emnist --alpha 0.4 --model leafcnn1 --local_epochs 1 --lr 0.0005 --batch_size 32 --n_rounds 50 --minimum_join_rounds 25 --optimizer_name adam --seed 42 --device cuda --split_method dirichlet  --set_single_delay_client $client_id
#  echo "qfedcg"
#  python main.py --fl_method qfedcg --dataset_name emnist --alpha 0.4 --model leafcnn1 --local_epochs 1 --lr 0.0005 --batch_size 32 --n_rounds 50 --minimum_join_rounds 25 --optimizer_name adam --seed 42 --device cuda --split_method dirichlet  --set_single_delay_client $client_id
#  echo retfhd
#  python main.py --fl_method retfhd --dataset_name emnist --alpha 0.4 --model leafcnn1 --local_epochs 1 --lr 0.001 --batch_size 32 --n_rounds 50 --minimum_join_rounds 25 --optimizer_name adam --seed 42 --device cuda --split_method dirichlet --set_single_delay_client $client_id
#done



#new_client_ids=(12 42 39 34 40)
#for client_id in "${new_client_ids[@]}"
#do
#  echo "Running with client_id $client_id in Cifar100"
#  echo dfedcad
#  python main.py --fl_method dfedcad --dataset_name cifar100 --alpha $alpha --model resnet18 --local_epochs 1 --lr 0.0005 --batch_size 32 --n_rounds 50 --minimum_join_rounds 25 --optimizer_name adam --seed 42 --device cuda --split_method dirichlet --set_single_delay_client $client_id --lambda_alignment 10 --base_decay_rate 0.1
#  echo dfedavg
#  python main.py --fl_method dfedavg --dataset_name cifar100 --alpha $alpha --model resnet18 --local_epochs 1 --lr 0.0001 --batch_size 32 --n_rounds 50 --minimum_join_rounds 25 --optimizer_name adam --seed 42 --device cuda --split_method dirichlet --set_single_delay_client $client_id
#  echo dfedmtkdrl
#  python main.py --fl_method dfedmtkdrl --dataset_name cifar100 --alpha $alpha --model resnet18 --local_epochs 1 --lr 0.001 --batch_size 32 --n_rounds 50 --minimum_join_rounds 25 --optimizer_name adam --seed 42 --device cuda --split_method dirichlet --set_single_delay_client $client_id --lambda_kd 0.1
#  echo dfedpgp
#  python main.py --fl_method dfedpgp --dataset_name cifar100 --alpha $alpha --model resnet18 --local_epochs 1 --lr 0.0001 --batch_size 32 --n_rounds 50 --minimum_join_rounds 25 --optimizer_name adam --seed 42 --device cuda --split_method dirichlet --set_single_delay_client $client_id --lambda_kd 0.1
#  echo dfedsam
#  python main.py --fl_method dfedsam --dataset_name cifar100 --alpha $alpha --model resnet18 --local_epochs 1 --lr 0.0001 --batch_size 32 --n_rounds 50 --minimum_join_rounds 25 --optimizer_name adam --seed 42 --device cuda --split_method dirichlet --set_single_delay_client $client_id --lambda_kd 0.1
#  echo fedgo
#  python main.py --fl_method fedgo --dataset_name cifar100 --alpha 0.4 --model resnet18 --local_epochs 1 --lr 0.0001 --batch_size 32 --n_rounds 50 --minimum_join_rounds 25 --optimizer_name adam --seed 42 --device cuda --split_method dirichlet --set_single_delay_client $client_id --lambda_kd 0.1
#  echo qfedcg
#  python main.py --fl_method qfedcg --dataset_name cifar100 --alpha $alpha --model resnet18 --local_epochs 1 --lr 0.0001 --batch_size 32 --n_rounds 50 --minimum_join_rounds 25 --optimizer_name adam --seed 42 --device cuda --split_method dirichlet --set_single_delay_client $client_id --lambda_kd 0.1
#  for alpha in "${alphas}"
#  do
#    echo dfedcad_without_alignment
#    python main.py --fl_method dfedcad --dataset_name cifar100 --alpha $alpha --model resnet18 --local_epochs 1 --lr 0.0005 --batch_size 32 --n_rounds 50 --minimum_join_rounds 25 --optimizer_name adam --seed 42 --device cuda --split_method dirichlet --set_single_delay_client $client_id --lambda_alignment 0 --base_decay_rate 0.1
#  done
#  do
#  echo retfhd
#  python main.py --fl_method retfhd --dataset_name cifar100 --alpha $alpha --model resnet18 --local_epochs 1 --lr 0.0001 --batch_size 32 --n_rounds 50 --minimum_join_rounds 25 --optimizer_name adam --seed 42 --device cuda --split_method dirichlet --set_single_delay_client $client_id
#  done
#done

#new_client_ids=(12)
#for client_id in "${new_client_ids[@]}"
#do
#  echo "Running with client_id $client_id in tiny_imagenet"
#  echo dfedcad
#  python main.py --fl_method dfedcad --dataset_name tiny_imagenet --alpha $alpha --model resnet18 --local_epochs 1 --lr 0.0005 --batch_size 32 --n_rounds 50 --minimum_join_rounds 25 --optimizer_name adam --seed 42 --device cuda --split_method dirichlet --set_single_delay_client $client_id --lambda_alignment 10 --base_decay_rate 0.1
#  echo dfedavg
#  python main.py --fl_method dfedavg --dataset_name tiny_imagenet --alpha $alpha --model resnet18 --local_epochs 1 --lr 0.0001 --batch_size 32 --n_rounds 50 --minimum_join_rounds 25 --optimizer_name adam --seed 42 --device cuda --split_method dirichlet --set_single_delay_client $client_id
#  echo dfedmtkdrl
#  python main.py --fl_method dfedmtkdrl --dataset_name tiny_imagenet --alpha $alpha --model resnet18 --local_epochs 1 --lr 0.001 --batch_size 32 --n_rounds 50 --minimum_join_rounds 25 --optimizer_name adam --seed 42 --device cuda --split_method dirichlet --set_single_delay_client $client_id --lambda_kd 0.1
#  echo dfedpgp
#  python main.py --fl_method dfedpgp --dataset_name tiny_imagenet --alpha $alpha --model resnet18 --local_epochs 1 --lr 0.0001 --batch_size 32 --n_rounds 50 --minimum_join_rounds 25 --optimizer_name adam --seed 42 --device cuda --split_method dirichlet --set_single_delay_client $client_id --lambda_kd 0.1
#  echo dfedsam 
#  python main.py --fl_method dfedsam --dataset_name tiny_imagenet --alpha $alpha --model resnet18 --local_epochs 1 --lr 0.0001 --batch_size 32 --n_rounds 50 --minimum_join_rounds 25 --optimizer_name adam --seed 42 --device cuda --split_method dirichlet --set_single_delay_client $client_id --lambda_kd 0.1
#  echo fedgo
#  python main.py --fl_method fedgo --dataset_name tiny_imagenet --alpha $alpha --model resnet18 --local_epochs 1 --lr 0.0001 --batch_size 32 --n_rounds 50 --minimum_join_rounds 25 --optimizer_name adam --seed 42 --device cuda --split_method dirichlet --set_single_delay_client $client_id --lambda_kd 0.1
#  echo qfedcg
#  python main.py --fl_method qfedcg --dataset_name tiny_imagenet --alpha $alpha --model resnet18 --local_epochs 1 --lr 0.0001 --batch_size 32 --n_rounds 50 --minimum_join_rounds 25 --optimizer_name adam --seed 42 --device cuda --split_method dirichlet --set_single_delay_client $client_id --lambda_kd 0.1
#  for alpha in "${alphas}"
#  do
#    echo dfedcad_without_alignment
#    python main.py --fl_method dfedcad --dataset_name tiny_imagenet --alpha $alpha --model resnet18 --local_epochs 1 --lr 0.0005 --batch_size 32 --n_rounds 50 --minimum_join_rounds 25 --optimizer_name adam --seed 42 --device cuda --split_method dirichlet --set_single_delay_client $client_id --lambda_alignment 0 --base_decay_rate 0.1
#  done
#  do
#    echo retfhd
#    python main.py --fl_method retfhd --dataset_name tiny_imagenet --alpha $alpha --model resnet18 --local_epochs 1 --lr 0.0001 --batch_size 32 --n_rounds 50 --minimum_join_rounds 25 --optimizer_name adam --seed 42 --device cuda --split_method dirichlet --set_single_delay_client $client_id
#  done
#done

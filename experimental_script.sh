#Cifar10
echo cifar10 dfedcad
python main.py --fl_method dfedcad --dataset_name cifar10 --alpha 0.4 --model lenet --local_epochs 1 --lr 0.0005 --batch_size 32 --n_rounds 50 --minimum_join_rounds 25 --optimizer_name adam --seed 42 --device cuda --split_method dirichlet --set_single_delay_client 81 --lambda_alignment 10 --base_decay_rate 0.1
echo cifar10 dfedavg
python main.py --fl_method dfedavg --dataset_name cifar10 --alpha 0.4 --model lenet --local_epochs 1 --lr 0.001 --batch_size 32 --n_rounds 50 --minimum_join_rounds 25 --optimizer_name adam --seed 42 --device cuda --split_method dirichlet --set_single_delay_client 81
echo cifar10 dfedmtkdrl
python main.py --fl_method dfedmtkdrl --dataset_name cifar10 --alpha 0.4 --model lenet --local_epochs 1 --lr 0.001 --batch_size 32 --n_rounds 50 --minimum_join_rounds 25 --optimizer_name adam --seed 42 --device cpu --split_method dirichlet --set_single_delay_client 81 --lambda_alignment 0.1

#EMNIST
echo emnist dfedcad
python main.py --fl_method dfedcad --dataset_name emnist --alpha 0.4 --model leafcnn1 --local_epochs 1 --lr 0.0005 --batch_size 32 --n_rounds 50 --minimum_join_rounds 25 --optimizer_name adam --seed 42 --device cuda --split_method dirichlet --set_single_delay_client 81 --lambda_alignment 10 --base_decay_rate 0.1
echo emnist dfedavg
python main.py --fl_method dfedavg --dataset_name emnist --alpha 0.4 --model leafcnn1 --local_epochs 1 --lr 0.001 --batch_size 32 --n_rounds 50 --minimum_join_rounds 25 --optimizer_name adam --seed 42 --device cuda --split_method dirichlet --set_single_delay_client 81
echo emnist dfedmtkdrl
python main.py --fl_method dfedmtkdrl --dataset_name emnist --alpha 0.4 --model lenet --local_epochs 1 --lr 0.001 --batch_size 32 --n_rounds 50 --minimum_join_rounds 25 --optimizer_name adam --seed 42 --device cpu --split_method dirichlet --set_single_delay_client 81 --lambda_kd 0.1


#Cifar100
echo cifar100 dfedcad
python main.py --fl_method dfedcad --dataset_name cifar100 --alpha 0.4 --model resnet18 --local_epochs 1 --lr 0.0005 --batch_size 32 --n_rounds 50 --minimum_join_rounds 25 --optimizer_name adam --seed 42 --device cuda --split_method dirichlet --set_single_delay_client 40 --lambda_alignment 10 --base_decay_rate 0.1
echo cifar100 dfedavg
python main.py --fl_method dfedavg --dataset_name cifar100 --alpha 0.4 --model resnet18 --local_epochs 1 --lr 0.001 --batch_size 32 --n_rounds 50 --minimum_join_rounds 25 --optimizer_name adam --seed 42 --device cuda --split_method dirichlet --set_single_delay_client 40
echo cifar100 dfedmtkdrl
python main.py --fl_method dfedmtkdrl --dataset_name cifar100 --alpha 0.4 --model lenet --local_epochs 1 --lr 0.001 --batch_size 32 --n_rounds 50 --minimum_join_rounds 25 --optimizer_name adam --seed 42 --device cpu --split_method dirichlet --set_single_delay_client 40 --lambda_kd 0.1

#tiny_imagenet
echo tiny_imagenet dfedcad
python main.py --fl_method dfedcad --dataset_name tiny_imagenet --alpha 0.4 --model resnet18 --local_epochs 1 --lr 0.0005 --batch_size 32 --n_rounds 50 --minimum_join_rounds 25 --optimizer_name adam --seed 42 --device cuda --split_method dirichlet --set_single_delay_client 40 --lambda_alignment 10 --base_decay_rate 0.1
echo tiny_imagenet dfedavg
python main.py --fl_method dfedavg --dataset_name tiny_imagenet --alpha 0.4 --model resnet18 --local_epochs 1 --lr 0.001 --batch_size 32 --n_rounds 50 --minimum_join_rounds 25 --optimizer_name adam --seed 42 --device cuda --split_method dirichlet --set_single_delay_client 40
echo tiny_imagener dfedkdrl
python main.py --fl_method dfedmtkdrl --dataset_name tiny_imagenet --alpha 0.4 --model lenet --local_epochs 1 --lr 0.001 --batch_size 32 --n_rounds 50 --minimum_join_rounds 25 --optimizer_name adam --seed 42 --device cpu --split_method dirichlet --set_single_delay_client 40 --lambda_kd 0.1
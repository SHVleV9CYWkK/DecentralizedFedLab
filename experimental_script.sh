#Cifar10
echo cifar10 dfedcad
python main.py --fl_method dfedcad --dataset_name cifar10 --alpha 0.4 --model lenet --local_epochs 1 --lr 0.001 --batch_size 32 --n_rounds 50 --minimum_join_rounds 25 --optimizer_name adam --seed 42 --device cuda --split_method dirichlet --set_single_delay_client 81 --lambda_alignment 10
echo cifar10 fedavg
python main.py --fl_method fedavg --dataset_name cifar10 --alpha 0.4 --model lenet --local_epochs 1 --lr 0.001 --batch_size 32 --n_rounds 50 --minimum_join_rounds 25 --optimizer_name adam --seed 42 --device cuda --split_method dirichlet --set_single_delay_client 81 --lambda_alignment 10

#EMNIST
echo emnist dfedcad
python main.py --fl_method dfedcad --dataset_name emnist --alpha 0.4 --model leafcnn1 --local_epochs 1 --lr 0.001 --batch_size 32 --n_rounds 50 --minimum_join_rounds 25 --optimizer_name adam --seed 42 --device cuda --split_method dirichlet --set_single_delay_client 81 --lambda_alignment 10
echo emnist fedavg
python main.py --fl_method fedavg --dataset_name emnist --alpha 0.4 --model leafcnn1 --local_epochs 1 --lr 0.001 --batch_size 32 --n_rounds 50 --minimum_join_rounds 25 --optimizer_name adam --seed 42 --device cuda --split_method dirichlet --set_single_delay_client 81 --lambda_alignment 10

#Cifar100
echo cifar100 dfedcad
python main.py --fl_method dfedcad --dataset_name cifar100 --alpha 0.4 --model resnet18 --local_epochs 1 --lr 0.001 --batch_size 32 --n_rounds 50 --minimum_join_rounds 25 --optimizer_name adam --seed 42 --device cuda --split_method dirichlet --set_single_delay_client 81 --lambda_alignment 10
echo cifar100 fedavg
python main.py --fl_method fedavg --dataset_name cifar100 --alpha 0.4 --model resnet18 --local_epochs 1 --lr 0.001 --batch_size 32 --n_rounds 50 --minimum_join_rounds 25 --optimizer_name adam --seed 42 --device cuda --split_method dirichlet --set_single_delay_client 81 --lambda_alignment 10

#tiny_imagenet
echo tiny_imagenet dfedcad
python main.py --fl_method dfedcad --dataset_name tiny_imagenet --alpha 0.4 --model resnet18 --local_epochs 1 --lr 0.001 --batch_size 32 --n_rounds 50 --minimum_join_rounds 25 --optimizer_name adam --seed 42 --device cuda --split_method dirichlet --set_single_delay_client 81 --lambda_alignment 10
echo tiny_imagenet fedavg
python main.py --fl_method fedavg --dataset_name tiny_imagenet --alpha 0.4 --model resnet18 --local_epochs 1 --lr 0.001 --batch_size 32 --n_rounds 50 --minimum_join_rounds 25 --optimizer_name adam --seed 42 --device cuda --split_method dirichlet --set_single_delay_client 81 --lambda_alignment 10


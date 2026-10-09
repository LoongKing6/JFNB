# JFNB
JFNB
Until the paper is accepted, we are providing a dataset of participants' data：
https://drive.google.com/file/d/1K4uXp_DMbXEGev9JDHdo1e83E0Xjt7S2/view?usp=drive_link


Details about the dataset can be found under “github-pages” in the lower-right corner.



How to Run:
Select the task you want to run in `config`, then click “Run” in `main.py`.

python main.py --dataset EMO --model 'JFNB'  --num-class 2  --data-path /home/RESB --label-type A --train_method n_fold  --graph-type 'fro'  --input-shape' "1,32,2000"

python main.py --dataset EMO --model 'JFNB'  --num-class 2  --data-path /home/RESB --label-type A --train_method loso    --graph-type 'fro'  --input-shape' "1,32,2000"

python main.py --dataset EMO --model 'JFNB'  --num-class 6  --data-path /home/RESB --label-type S --train_method n_fold  --graph-type 'fro'  --input-shape' "1,32,2000"

python main.py --dataset EMO --model 'JFNB'  --num-class 6  --data-path /home/RESB --label-type S --train_method loso    --graph-type 'fro'  --input-shape' "1,32,2000"

python main.py --dataset EMO --model 'JFNB'  --num-class 7  --data-path /home/RESB --label-type T --train_method n_fold  --graph-type 'fro'  --input-shape' "1,32,2000"

python main.py --dataset EMO --model 'JFNB'  --num-class 7  --data-path /home/RESB --label-type T --train_method loso    --graph-type 'fro'  --input-shape' "1,32,2000"

python main.py --dataset DEAP --model 'JFNB'  --num-class 2  --data-path /home/DEAP --label-type A --train_method n_fold  --graph-type 'fro'  --input-shape' "1,32,800"

python main.py --dataset DEAP --model 'JFNB'  --num-class 2  --data-path /home/DEAP --label-type A --train_method loso    --graph-type 'fro'  --input-shape' "1,32,800"


python main.py --dataset EEGMAT --model 'JFNB'  --num-class 2  --data-path /home/EEGMAT --label-type A --train_method loso    --graph-type 'fro'  --input-shape' "1,20,800"

python main.py --dataset ISRUC --model 'JFNB'  --num-class 2  --data-path /home/ISRUC --label-type A --train_method n_fold  --graph-type 'fro'  --input-shape' "1,6,3000"

python main.py --dataset ISRUC --model 'JFNB'  --num-class 2  --data-path /home/ISRUC --label-type A --train_method loso    --graph-type 'fro'  --input-shape' "1,32,3000"

python main.py --dataset MIP --model 'JFNB'  --num-class 2  --data-path /home/multisensory --label-type A --train_method n_fold  --graph-type 'fro'

python main.py --dataset MIP --model 'JFNB'  --num-class 2  --data-path /home/multisensory --label-type A --train_method loso    --graph-type 'fro'

REM python3 main.py --resdir ./resule --use_pretrain SBERT --bert distilbert --datapath ./data --dataname searchsnippets --num_classes 8 --text text --label label --objective SCCL --augtype virtual --temperature 0.5 --eta 10 --lr 1e-05 --lr_scale 100 --max_length 32 --batch_size 400 --max_iter 1000 --print_freq 100 --gpuid 1 &
python3 main.py ^
        --resdir ,/results ^
        --use_pretrain SBERT ^
        --bert distilbert ^
        --datapath ./data ^
        --dataname agnew ^
        --num_classes 8 ^
        --text text ^
        --label label ^
        --objective SCCL ^
        --augtype virtual ^
        --temperature 0.5 ^
        --eta 10 ^
        --lr 1e-05 ^
        --lr_scale 100 ^
        --max_length 32 ^
        --batch_size 400 ^
        --max_iter 1000 ^
        --print_freq 100 ^
        --gpuid 1 &

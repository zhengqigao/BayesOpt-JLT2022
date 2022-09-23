
#python main_bo.py       --N_initial 30 --N_total 70 --acq 0  >> ./logs/log_lcb1.txt
#python main_bo.py       --N_initial 30 --N_total 70 --acq 1  >> ./logs/log_ei.txt
#python main_bo.py       --N_initial 30 --N_total 70 --acq 2  >> ./logs/log_pi.txt
#
#python main_gradient.py                --N_total 68          >> ./logs/log_gradient.txt
#python main_ga.py                      --N_iter  1           >> ./logs/log_ga.txt
#
#python main_bo.py       --N_initial 30 --N_total 70 --acq 3 >> ./logs/log_lcb2.txt
#python main_bo.py       --N_initial 30 --N_total 70 --acq 4 >> ./logs/log_lcb3.txt


python main_bo.py       --N_initial 30 --N_total 500 --acq 0  --seed 0 >> ./logs/log_lcb1_longrun_seed0.txt

python main_bo.py       --N_initial 30 --N_total 70 --acq 0  --seed 0 >> ./logs/log_lcb1_seed0.txt
python main_bo.py       --N_initial 30 --N_total 70 --acq 0  --seed 1 >> ./logs/log_lcb1_seed1.txt
python main_bo.py       --N_initial 30 --N_total 70 --acq 0  --seed 2 >> ./logs/log_lcb1_seed2.txt
python main_bo.py       --N_initial 30 --N_total 70 --acq 0  --seed 3 >> ./logs/log_lcb1_seed3.txt
python main_bo.py       --N_initial 30 --N_total 70 --acq 0  --seed 4 >> ./logs/log_lcb1_seed4.txt
python main_bo.py       --N_initial 30 --N_total 70 --acq 0  --seed 5 >> ./logs/log_lcb1_seed5.txt
python main_bo.py       --N_initial 30 --N_total 70 --acq 0  --seed 6 >> ./logs/log_lcb1_seed6.txt
python main_bo.py       --N_initial 30 --N_total 70 --acq 0  --seed 7 >> ./logs/log_lcb1_seed7.txt
python main_bo.py       --N_initial 30 --N_total 70 --acq 0  --seed 8 >> ./logs/log_lcb1_seed8.txt
python main_bo.py       --N_initial 30 --N_total 70 --acq 0  --seed 9 >> ./logs/log_lcb1_seed9.txt
python main_bo.py       --N_initial 30 --N_total 70 --acq 0  --seed 10 >> ./logs/log_lcb1_seed10.txt
python main_bo.py       --N_initial 30 --N_total 70 --acq 0  --seed 11 >> ./logs/log_lcb1_seed11.txt
python main_bo.py       --N_initial 30 --N_total 70 --acq 0  --seed 12 >> ./logs/log_lcb1_seed12.txt
python main_bo.py       --N_initial 30 --N_total 70 --acq 0  --seed 13 >> ./logs/log_lcb1_seed13.txt
python main_bo.py       --N_initial 30 --N_total 70 --acq 0  --seed 14 >> ./logs/log_lcb1_seed14.txt
python main_bo.py       --N_initial 30 --N_total 70 --acq 0  --seed 15 >> ./logs/log_lcb1_seed15.txt
python main_bo.py       --N_initial 30 --N_total 70 --acq 0  --seed 16 >> ./logs/log_lcb1_seed16.txt
python main_bo.py       --N_initial 30 --N_total 70 --acq 0  --seed 17 >> ./logs/log_lcb1_seed17.txt
python main_bo.py       --N_initial 30 --N_total 70 --acq 0  --seed 18 >> ./logs/log_lcb1_seed18.txt
python main_bo.py       --N_initial 30 --N_total 70 --acq 0  --seed 19 >> ./logs/log_lcb1_seed19.txt

python main_ga.py                  --N_iter  1    --seed 0        >> ./logs/log_ga_seed0.txt
python main_ga.py                  --N_iter  1    --seed 1        >> ./logs/log_ga_seed1.txt
python main_ga.py                  --N_iter  1    --seed 2        >> ./logs/log_ga_seed2.txt
python main_ga.py                  --N_iter  1    --seed 3        >> ./logs/log_ga_seed3.txt
python main_ga.py                  --N_iter  1    --seed 4        >> ./logs/log_ga_seed4.txt
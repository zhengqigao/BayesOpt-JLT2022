
#python main_bo.py       --N_initial 5 --N_total 60 --acq 0  >> ./logs/log_lcb1.txt
#python main_bo.py       --N_initial 5 --N_total 60 --acq 1  >> ./logs/log_ei.txt
#python main_bo.py       --N_initial 5 --N_total 60 --acq 2  >> ./logs/log_pi.txt

#python main_gradient.py                --N_total 50          >> ./logs/log_gradient.txt
python main_ga.py                      --N_iter  1           >> ./logs/log_ga2.txt


#python main_bo.py       --N_initial 5 --N_total 60 --acq 3 >> ./logs/log_lcb2.txt
#python main_bo.py       --N_initial 5 --N_total 60 --acq 4 >> ./logs/log_lcb3.txt

#python main_bo.py       --N_initial 5 --N_total 60 --acq 5 >> ./logs/log_lcb4.txt
#python main_bo.py       --N_initial 5 --N_total 60 --acq 6 >> ./logs/log_lcb5.txt

#python main_bo.py       --N_initial 5 --N_total 60 --acq 7 >> ./logs/log_lcb6.txt
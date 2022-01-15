for offset in {10..140} ; do
    #python replay_comma_more_attack_drp_wb_metric.py scnn $SLURM_ARRAY_TASK_ID ${offset}
    python opt_4ndplace.py ${offset}
    #python replay_comma_more_attack_drp_wb_metric.py polylanenet $SLURM_ARRAY_TASK_ID ${offset}
done

from multiprocessing import Process

import signal
import os
import sys
import random
import json
import argparse
import datetime

start_time = datetime.datetime.now()
  
configs = {
    'script_path': os.path.abspath(os.getcwd()),
    'top_dir': os.path.abspath('../experiments/'),
    'build_dir': os.path.abspath('../klee/build/')
}



def load_pgm_config(config_file):
    with open(config_file, 'r') as f:
        parsed = json.load(f)
    return parsed


def gen_run_cmd(pgm, stgy, mem, small_time, iters, tool, ith_trial, result_dir):
    
    base_command=" ".join([configs['build_dir']+"/bin/klee", "-trial="+str(iters), "--max-memory="+mem, "--watchdog -max-time="+small_time, 
                           "-dirname="+configs['top_dir']+"/"+result_dir, "-write-kqueries", "-only-output-states-covering-new", 
                           "--simplify-sym-indices", "--output-module=false", "--output-source=false", "--output-stats=false", 
                           "--disable-inlining", "--use-forked-solver", "--use-cex-cache", "--libc=uclibc", "--posix-runtime", 
                           "-env-file="+configs['build_dir']+"/../test.env", 
                           "--max-sym-array-size=4096", "--max-instruction-time=30", "--switch-type=internal", 
                           "--use-batching-search", "--batch-instructions=10000", "-ignore-solver-failures"])

    opt_flag=1
    no_opt_pgms=["gawk", "trueprint"]    
    if pgm in no_opt_pgms:
        opt_flag=0
    
    if stgy=="roundrobin": 
        stgy="random-path --search=nurs:covnew"

    if opt_flag==1:
        base_command=" ".join([base_command, "--optimize"])

    if (tool=="homi") and (iters!=0):
        base_command=" ".join([base_command, "-homi", "-parallel="+str(ith_trial)])
    
    # Follow the symbolic arguments in KLEE paper. (https://klee.github.io/docs/coreutils-experiments/)
    if pgm=="dd":
        argv = "--sym-args 0 3 10 --sym-files 1 8 --sym-stdin 8 --sym-stdout"
    else:
        argv = "--sym-args 0 1 10 --sym-args 0 2 2 --sym-files 1 8 --sym-stdin 8 --sym-stdout"
    
    run_cmd = " ".join([base_command, "--search="+stgy, pgm+".bc", argv])
    
    return run_cmd


def run_all(l_config, pgm, stgy, mem, small_time, ith_trial, iters, tool, d_name):
    top_dir = "/".join([configs['top_dir'], tool+"__"+stgy+str(iters), pgm])
    if not os.path.exists(top_dir):
        os.makedirs(top_dir)
    
    group_dir = top_dir + "/" + str(ith_trial)
    os.system(" ".join(["cp -r", l_config['pgm_dir'], group_dir]))
    os.chdir(group_dir+l_config['exec_dir'])
    
    result_dir="result_"+d_name 
    top_tc_dir="/".join([configs['top_dir'], result_dir])
    print top_tc_dir
    if not os.path.exists(top_tc_dir):
        os.mkdir(top_tc_dir)
    
    if tool=="homi":
        tc_dir="/".join([configs['top_dir'], result_dir, str(ith_trial)+"homi_"+pgm+"_"+stgy+"_tc_dir"]) 
    else:
        tc_dir="/".join([configs['top_dir'], result_dir, str(ith_trial)+"pureklee_"+pgm+"_"+stgy+"_tc_dir"]) 
    
    if not os.path.exists(tc_dir):
        os.mkdir(tc_dir)
    
    os.chdir(group_dir+l_config['exec_dir'])
    run_cmd = gen_run_cmd(pgm, stgy, mem, small_time, iters, tool, ith_trial, result_dir)
    
    with open(os.devnull, 'wb') as devnull:
        os.system(run_cmd)
    
    klee_dir = "klee-out-0"
    rm_cmd=" ".join(["rm", klee_dir+"/assembly.ll", klee_dir+"/run.istats"])
    os.system(rm_cmd) 

    cp_cmd = " ".join(["cp", "-r", klee_dir, tc_dir+"/"+str(iters)+"__tc_dirs"]) 
    print cp_cmd
    os.system(cp_cmd)

    cp2_cmd = " ".join(["cp", "time_result state_data", tc_dir+"/"+str(iters)+"__tc_dirs/"]) 
    os.system(cp2_cmd)
    
    rm_cmd=" ".join(["rm -rf", group_dir])
    os.system(rm_cmd)  

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("pgm_config")
    parser.add_argument("pgm")
    parser.add_argument("search_heuristic",help='[nurs:covnew, random-path, ..]')
    parser.add_argument("memory")
    parser.add_argument("small_time",help='[200(s),800(s)]')
    parser.add_argument("ith_trial",help='[1,2,3,..]')
    parser.add_argument("iters")
    parser.add_argument("tool",help='[homi, pureklee]')
    parser.add_argument("d_name", help='0314')
    
    args = parser.parse_args()
    pgm_config = args.pgm_config
    load_config = load_pgm_config(args.pgm_config)
    pgm = args.pgm
    stgy = args.search_heuristic
    mem = args.memory
    small_time = args.small_time
    ith_trial = int(args.ith_trial)
    iters = int(args.iters)
    tool = args.tool
    d_name=args.d_name
    run_all(load_config, pgm, stgy, mem, small_time, ith_trial, iters, tool, d_name)

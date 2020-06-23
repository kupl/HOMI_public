#!/usr/bin/python3
from __future__ import with_statement
import os, re, sys, random, csv
import pandas as pd
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import time, datetime, argparse, glob
from datetime import datetime

import signal
import subprocess
from subprocess import Popen, PIPE
import json
from copy import deepcopy
from threading import Timer

configs = {
    's_dir': os.path.abspath(os.getcwd()), # script dir
    'e_dir': os.path.abspath('../experiments/'), # experiment dir
    'b_dir': os.path.abspath('../klee/build/') # build dir
}

lines = ['-', '--', '-.', ':']
marker = ['^', '.', 'D', '*']
line_markers = {
    'HOMI': ('-', 'D', 'm'),
    'InstrCount': (':', '^', 'g'),
    'RandomPath': ('--', '.', 'b'),
    'QueryCost': ('-', None, 'y'),
    'Divide': ('-.', None, 'g'),
    'Depth': ('--', '*', 'c'),
    'MinDistance': ('--', 'x', 'indigo'),
    'CovNew': ('-', "p", 'lime'),
    'CPICount': (':', "h", 'tomato'),
    'RandomState': ('--', "d", 'cadetblue'),
    'RoundRobin': ('-', 's', 'orange'),
}

def Load_Pgm_Config(config_file):
    with open(config_file, 'r') as f:
        parsed = json.load(f)
    return parsed


def Total_Coverage(cov_file):
    coverage=0
    with open(cov_file, 'r') as f:
        lines= f.readlines()
        for line in lines:
            if "Taken at least" in line:
                data=line.split(':')[1]
                total=int((data.split('% of ')[1]).strip())
                coverage=coverage+total
    return coverage


def Cal_Coverage(cov_file):
    coverage=0
    with open(cov_file, 'r') as f:
        lines= f.readlines()
        for line in lines:
            if "Taken at least" in line:
                data=line.split(':')[1]
                percent=float(data.split('% of ')[0])
                total=int((data.split('% of ')[1]).strip())
                cov=int(percent*total/100)
                coverage=coverage+cov
    return coverage


def Kill_Process(process, testcase):
    with open(configs['s_dir']+"/killed_history", 'a') as f:
        f.write(testcase+"\n")
    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    print("timeout test-case!")


def Graph_Data(pgm, stgy, knowledge, iters, dir_name, target_dir):
    # init_time 
    os.chdir(configs['e_dir']+"/"+target_dir+"/"+dir_name+"/0__tc_dirs")
    init_time=0
    with open("time_result",'r') as f:
        lines = f.readlines()
        init_info=lines[0].split(': ')[1] #2019-11-02.20:49:41
        day=init_info.split('.')[0].split('-')[2] 
        date = init_info.split('.')[1].split(':') #[20, 49, 41]
        init_time =int(day)*3600*24+int(date[0])*3600+int(date[1])*60+int(date[2])

    time_cov_dic={}
    cov = 0
    for num in range(0,iters+1):
        time_inputs_dic={}
        sub_dir=str(num)+"__tc_dirs"
        os.chdir(configs['e_dir']+"/"+target_dir+"/"+dir_name+"/"+sub_dir)
        with open("time_result") as f1:
            lines = f1.readlines()
            for line in lines[1:]:
                info = line.split('/')
                day = int(info[-1].split('.')[1].split('-')[2])
                hour = int(info[-1].split('.')[2].split(':')[0])
                minute = int(info[-1].split('.')[2].split(':')[1])
                sec = int(info[-1].split('.')[2].split(':')[2])
                testcase_name = info[-1].split(':')[0] 
            
                current_time = day*3600*24+hour*3600+minute*60+sec-init_time+1
                if current_time in time_inputs_dic:
                    time_inputs_dic[current_time].append(testcase_name)
                else:
                    time_inputs_dic[current_time]=[testcase_name]
     
            time_inputs_list = sorted(time_inputs_dic.items(), key=lambda kv: kv[0], reverse = False)
        
        for element in time_inputs_list:
            time=element[0]
            for tc in element[1]:
                if sub_dir+"/"+tc in knowledge.keys():
                    cov = max(cov, knowledge[sub_dir+"/"+tc])
            time_cov_dic[time]=cov
    
    time_cov_list = sorted(time_cov_dic.items(), key=lambda kv: kv[0], reverse = False)
    return time_cov_list 


def Write_Data(time_cov_list, pgm, stgy, dir_name, tool, target_dir):
    
    os.chdir(configs['e_dir']+"/"+target_dir)
    
    if stgy.find('nurs:covnew') >=0:
        stgy = "CovNew"
    elif stgy.find('nurs:icnt')>=0:
        stgy = "InstrCount"
    elif stgy.find('nurs:cpicnt')>=0:
        stgy = "CPICount"
    elif stgy.find('nurs:depth')>=0:
        stgy = "Depth"
    elif stgy.find('nurs:qc')>=0:
        stgy = "QueryCost"
    elif stgy.find('.w')>=0:
        stgy = "OURS"
    elif stgy.find('random-path')>=0:
        stgy = "RandomPath"
    elif stgy.find('random-state')>=0:
        stgy = "RandomState"
    elif stgy.find('nurs:md2u')>=0:
        stgy = "MinDistance"
    elif stgy.find('roundrobin')>=0:
        stgy = "RoundRobin"
   
    if "homi" in tool:
        ith_trial=tool.split('homi')[0]
    elif "pureklee" in tool:
        ith_trial=tool.split('pureklee')[0]
    else:
        print ("error")
        ith_trial="0"

    if "homi" in tool:
        result_name = "".join([pgm,"__",stgy+"+Homi","__result", ith_trial])
    else:
        result_name = "".join([pgm,"__",stgy,"__result", ith_trial])
    with open(result_name, 'w') as rf:
        total_time = int(time_cov_list[-1][0])
        total_cov = int(time_cov_list[-1][1])
        size = len(time_cov_list)
        
        idx = 0 
        for t in range(0,total_time+1):
            if t < time_cov_list[0][0]:
                cov = "0"
            elif t > time_cov_list[idx][0]:
                idx = idx +1 
                cov = time_cov_list[idx][1]
            else:
                cov = time_cov_list[idx][1]
            
            rf.write(str(t)+" "+str(cov)+"\n")
        

def Run_Gcov(load_config, pgm, stgy, tool, target_dir):
    knowledge={}
    dir_name="/".join([tool+"_"+pgm+"_"+stgy+"_tc_dir"])
    os.chdir(configs['e_dir']+"/"+target_dir+"/"+dir_name)
    dir_numbers = glob.glob("*__tc_dirs")
    max_list=[]
    for dir_num in dir_numbers:
        max_list.append(int(dir_num.split("__")[0]))
    iters=max(max_list)

    os.chdir(configs['s_dir']+"/"+load_config['gcov_path'])
    rm_cmd = " ".join(["rm", load_config['gcov_file'], load_config['gcda_file']])
    os.system(rm_cmd)
    
    testcase_results ={}
    for num in range(0,iters+1):

        os.chdir(configs['e_dir']+"/"+target_dir+"/"+dir_name+"/"+str(num)+"__tc_dirs")
        testcases= glob.glob("*.ktest")
        testcases.sort(key=lambda x:float((x.split('.ktest')[0]).split('test')[1]))       
        os.chdir(configs['s_dir']+"/"+load_config['gcov_path'])
         
        result_set=set()
        for tc in testcases:
            tc= str(num)+"__tc_dirs/"+tc
            
            run_cmd=[configs['b_dir']+"/bin/klee-replay", "./"+pgm, configs['e_dir']+"/"+target_dir+"/"+dir_name+"/"+tc] 
            proc = subprocess.Popen(run_cmd, preexec_fn=os.setsid, stdout=PIPE, stderr=PIPE)  
            my_timer = Timer(1, Kill_Process, [proc, configs['e_dir']+"/"+target_dir+"/"+dir_name+"/"+tc])
            try:
                my_timer.start()
                stdout, stderr = proc.communicate()
                lines = stderr.splitlines()
                for line in lines:
                    if "klee-replay: EXIT STATUS: " in str(line):
                        result = (str(line).split("klee-replay: EXIT STATUS: ")[1]).split(' (')[0]
                        result_set.add(result)
                        if "CRASHED" in result:
                            key=dir_name+"/"+tc
                            testcase_results[key]=result
            finally:
                my_timer.cancel()
            
            gcov_file="cov_result"
            gcov_cmd=" ".join(["gcov", "-b", load_config['gcda_file']+" >", gcov_file])
            os.system(gcov_cmd)
            
            coverage= Cal_Coverage(gcov_file)
            knowledge[tc]=coverage   
        total_coverage = Total_Coverage(gcov_file)
        print ("--gcov ongoing--")  
    os.chdir(configs['e_dir']+"/"+target_dir+"/"+dir_name)
    if testcase_results:
        with open('bug_result', 'w') as f:
            for key in testcase_results.keys():
                f.write("tc "+key+": "+testcase_results[key]+"\n")
    else:
        with open('bug_result', 'w') as f:
            f.write("sadly fail to find any bugs!\n")


    return dir_name, knowledge, iters

def Target_Find(target_dir, target_pgm):
    os.chdir(configs['e_dir']+"/"+target_dir) 
    dirs=glob.glob("*_dir")
    stgys=set()
    tools=set()
    for d_name in dirs:
        pgm=d_name.split('_')[1]
        if pgm == target_pgm:
            tool=d_name.split('_')[0]
            stgy=d_name.split('_')[2]
            stgys.add(stgy)
            tools.add(tool)
    return dirs, stgys, tools

def Target_Find2(target_dir, target_pgm):
    os.chdir(configs['e_dir']+"/"+target_dir) 
    dirs=glob.glob("*_result*")
    tools=set()
    iters=set()
    for d_name in dirs:
        pgm=d_name.split('__')[0]
        if pgm == target_pgm:
            tool=d_name.split('__')[1]
            ith=d_name.split('__')[2].split('result')[1]
            tools.add(tool)
            iters.add(ith)
    return tools, str(len(iters))


def average_file(pgm, stgy, iters): 
    # combine__RandomPath+Homi__result1
    # combine__RandomPath__result1    
    
    max_linenum = 0
    result_dic = {} 
    for i in range(1, int(iters)+1):
        try:
            with open(pgm+"__"+stgy+"__result"+str(i), 'r') as logf:
                result_dic[i] = logf.readlines()
                c_linenum = len(result_dic[i])
                if c_linenum > max_linenum:
                    max_linenum = c_linenum
        except:
            print ('nofile')

    for key in result_dic.keys():
        last_cov = result_dic[key][-1].split()[1]
        last_time = int(result_dic[key][-1].split()[0])
        
        diff_num = max_linenum - len(result_dic[key])
        for i in range(1, diff_num+1):
            result_dic[key].append(str(last_time+i)+" "+last_cov)

    
    aver_cov= {}
    for key in result_dic.keys():
        lines = result_dic[key]
        for l in lines:
            time = int(l.split()[0])
            cov = float(l.split()[1])
            if time in aver_cov.keys():
                aver_cov[time].append(cov)
            else: 
                aver_cov[time]=[cov]

    
    return aver_cov

def draw_graph(pgm, folder, stgys, iters):  
    dir_name = configs['s_dir']+"/"+folder+"/" 
    os.chdir(dir_name)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.grid(True, linestyle=':', linewidth=0.5, color='gray')
    ax.set_title(pgm)
    ax.set_xlabel('time(s)')
    ax.set_ylabel('# of Covered Branches')
    ax.autoscale_view()

    # sort top-6 stgys
    topk_stgys = {}
    for stgy in stgys:
        aver_cov = average_file(pgm, stgy, iters)
        average_list = sorted(aver_cov.items(), key=lambda kv: kv[0], reverse = False)
        topk_stgys[stgy] = sum(average_list[-1][1])/float(len(average_list[-1][1]))
    topk_stgy_list = (sorted(topk_stgys.items(), key=lambda kv: kv[1], reverse = True))[:6]
    print_list=[]
    for l in topk_stgy_list:
        print_list.append(l[0])
    top_stgys= [] 
    for l in topk_stgy_list:    
        stgy = l[0]
        top_stgys.append(stgy)
        markers_on= []
        times = []
        covs = []
        aver_cov = average_file(pgm, stgy, iters)
        average_list = sorted(aver_cov.items(), key=lambda kv: kv[0], reverse = False)
        for l in average_list:
            time= l[0]
            """
            if time<799:
                cov = 0
            else:
                cov = sum(l[1])/float(len(l[1]))
            """
            cov = sum(l[1])/float(len(l[1]))
            times.append(time)
            covs.append(cov)
            if time%300==0:
                markers_on.append(int(time))
       
        if "Homi" in stgy:
            stgy = "HOMI"
        
        l_time = int(average_list[-1][0])
        markers_on.append(l_time)
        ax.plot(times, covs, label=stgy, linestyle=line_markers[stgy][0], marker=line_markers[stgy][1],
        color=line_markers[stgy][2], markevery=markers_on, markeredgewidth=0.55, markeredgecolor='black')
        
    # save plotted image    
    filename = pgm +'_cov.pdf' 
    #plt.legend(loc='upper left',prop={'size': 10})
    plt.legend(loc=4, ncol=2, prop={'size': 10}, frameon=False)
    plt.savefig (configs['s_dir']+"/"+filename)
    print ("#############################################")
    print(filename + ' produced')
    return top_stgys

def rename_stgy(stgy):
    if stgy=="random-path":
        stgy="RandomPath"
    elif stgy == "nurs:covnew":
        stgy= 'CovNew'
    elif stgy == "nurs:icnt":
        stgy='InstrCount'
    elif stgy == "roundrobin":
        stgy = 'RoundRobin'
    elif stgy == "bfs":
        stgy="BFS"
    elif stgy == "nurs:cpicnt":
        stgy = "CPICount"
    elif stgy == "nurs:md2u":
        stgy = "MinDistance"
    elif stgy == "random-state":
        stgy = "RandomState"
    elif stgy == "nurs:depth":
        stgy = 'Depth'
    elif stgy == "nurs:qc":
        stgy = 'QueryCost'                       
    return stgy



def draw_state_graph(pgm, folder, used_tools, stgys, iters):
    dir_name = configs['s_dir']+"/"+folder+"/" 
    os.chdir(dir_name)
    fig, ax = plt.subplots(1, 1)
    ax.grid(True, linestyle=':', linewidth=0.5, color='gray')
    ax.set_title(pgm)
    ax.set_xlabel('time(s)')
    ax.set_ylabel('# of States')
    
    topk_stgys = {}
    
    used_set=set()
    for tool in used_tools:
        if "homi" in tool:
            used_set.add("homi")
        elif "pureklee" in tool:
            used_set.add("pureklee")

    tools= list(used_set) 
    for tool in tools:
        for stgy in stgys:
            markers_on= []
            times = []
            covs = []
            aver_cov = average_statenum(pgm, folder, tool, stgy, iters)
       
            average_list = sorted(aver_cov.items(), key=lambda kv: kv[0], reverse = False)
            mark_iter=0 
            for l in average_list:
                time= l[0]
                cov = sum(l[1])/float(len(l[1]))
                times.append(time)
                covs.append(cov)
        
            if tool=="homi":
                label_name = rename_stgy(stgy)+"+Homi"
                stgy2="HOMI"
            else:
                label_name=rename_stgy(stgy)
                stgy2=label_name
            avs=round(np.mean(covs),1)

            ax.plot(times, covs, label=label_name, linewidth=1.0, linestyle=line_markers[stgy2][0], marker=line_markers[stgy2][1],
        color=line_markers[stgy2][2], markevery=800, markeredgewidth=0.65, markeredgecolor='black')
        
   
    filename = pgm +'_statenum.pdf' 
    plt.legend(loc='upper left',prop={'size': 10})
    plt.legend(loc='best', fancybox=True, shadow=True,  ncol=3, prop={'size': 7}, frameon=False)
    plt.savefig (configs['s_dir']+"/"+filename)
    print(filename + ' produced')
    print ("#############################################")


def average_statenum(pgm, folder, tool, stgy, iters):
    # 2pureklee_gawk_nurs:md2u_tc_dir
    aver_statenum= {}
    
    dir_name = configs['s_dir']+"/"+folder+"/" 
    for i in range(1, int(iters)+1):
        time=0
        spdir_name=str(i)+tool+"_"+pgm+"_"+stgy+"_tc_dir"
        os.chdir(dir_name+"/"+spdir_name) 
        
        dir_numbers = glob.glob("*__tc_dirs")
        max_list=[]
        for dir_num in dir_numbers:
            max_list.append(int(dir_num.split("__")[0]))
        max_iters=max(max_list)

        os.chdir("0__tc_dirs")
        init_time=0 
        with open("time_result",'r') as f:
            lines = f.readlines()
            init_info=lines[0].split(': ')[1] #2019-11-02.20:49:41
            day=init_info.split('.')[0].split('-')[2] 
            date = init_info.split('.')[1].split(':') #[20, 49, 41]
            init_time =int(day)*3600*24+int(date[0])*3600+int(date[1])*60+int(date[2])
        os.chdir(dir_name+"/"+spdir_name) 
        
        c_time=0            
        for j in range(0, max_iters+1):
            os.chdir(str(j)+"__tc_dirs")
            with open("time_result") as f1:
                lines = f1.readlines()
                init_info=lines[0].split(': ')[1] #2019-11-02.20:49:41
                day=int(init_info.split('.')[0].split('-')[2]) 
                date=init_info.split('.')[1].split(':') #[20, 49, 41]
                hour=int(date[0])
                minute=int(date[1])
                sec=int(date[2])
                c_time = day*3600*24+hour*3600+minute*60+sec-init_time+1
            
            np=0
            try:
                with open("state_data", 'r') as f:
                    data = f.readlines()[0]
                    statenum_data= data.split(": ")[1].split()
                    for snum in statenum_data:
                        if c_time in aver_statenum.keys():
                            aver_statenum[c_time].append(int(snum))
                        else:
                            aver_statenum[c_time]=[int(snum)]
                        c_time = c_time +1
            except:
                np=np+1
            os.chdir(dir_name+"/"+spdir_name)
    
    return aver_statenum



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pgm_config")
    parser.add_argument("target_dir")

    args = parser.parse_args()
    pgm_config = args.pgm_config
    load_config = Load_Pgm_Config(args.pgm_config)
    target_dir = args.target_dir

    pgm=load_config['pgm_name'] 
    dirs, stgys, used_tools = Target_Find(target_dir,pgm)
    for tool in used_tools:
        for stgy in stgys:
            dir_name, knowledge, iters = Run_Gcov(load_config, pgm, stgy, tool, target_dir) 
            time_cov_list = Graph_Data(pgm, stgy, knowledge, iters, dir_name, target_dir)
            Write_Data(time_cov_list, pgm, stgy, dir_name, tool, target_dir) 
    
    tools, iters= Target_Find2(target_dir, pgm)
    draw_graph(pgm, target_dir, tools, iters)
    draw_state_graph(pgm, target_dir, used_tools, stgys, iters)

if __name__ == '__main__':
    main()

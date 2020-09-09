import scipy.stats as stats
import glob, os, sys, argparse, signal
import numpy as np
from copy import deepcopy
import json, random
import time, datetime
from threading import Timer
import subprocess
from subprocess import Popen, PIPE

configs = {
    's_dir': os.path.abspath(os.getcwd()), # script dir
    'e_dir': os.path.abspath('../experiments/'), # experiment dir
    'b_dir': os.path.abspath('../klee/build/') # build dir
}
start_time = datetime.datetime.now()

mem_budget=2000 #default memeory budget in KLEE

S_time=[200,800,4] # sample space for the small time budget.
S_ratio=[20,60,3] # sample space for the pruning ratio.
lower, upper = -1.0, 1.0 #feature weight range

tried_wv={}
d_tried_budget={}
d_tc_data={}


def Discrete_Space(sample_space):
    # Sample Space S = [min_val, max_val, interval].
    # (e.g., [200,800,4] -> [200,400,600,800])
    space=[]
    min_val=sample_space[0]
    max_val=sample_space[1]
    interval=sample_space[2]
    space.append(min_val)
    for i in range(1,interval-1):
        val=min_val+int((max_val-min_val)/(interval-1))*i
        space.append(val)
    space.append(max_val)
    return space


def Load_Pgm_Config(config_file):
    with open(config_file, 'r') as f:
        parsed = json.load(f)
    return parsed


def Kill_Process(process, testcase):
    with open(configs['s_dir']+"/killed_history", 'a') as f:
        f.write(testcase+"\n")
    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    print("timeover!")


def Timeout_Checker(total_time, tool):
    current_time = datetime.datetime.now()
    elapsed_time = (current_time-start_time).total_seconds()
    if total_time < elapsed_time:
        os.chdir(configs['s_dir'])
        print ("#############################################")
        print ("################Time Out!!!!!################")
        print ("#############################################")
        sys.exit()

    return elapsed_time


def Run_KLEE(pgm_config, pgm, stgy, total_time, small_time, ith_trial, iters, tool, d_name, Space_time):
    # Check whether the total time budget expires.
    elapsed_time = Timeout_Checker(total_time, tool)
    
    # Maintain the number of each tried small budget.
    if iters !=0 and tool=="homi":
        if small_time in d_tried_budget.keys():
            d_tried_budget[small_time]=d_tried_budget[small_time]+1
        else:
            d_tried_budget[small_time]=1

    os.chdir(configs['s_dir'])
    if tool=="homi":
        remain_time = int(total_time-elapsed_time)
        if remain_time < int(min(Space_time)):
            small_time=str(remain_time)
        cmd=" ".join(["python", "Run_KLEE.py", pgm_config, pgm, stgy, str(mem_budget), small_time, ith_trial, str(iters), tool, d_name])
        os.system(cmd)
    else:
        if (iters!=0):
            small_time = str(int(total_time - elapsed_time))
        cmd=" ".join(["python", "Run_KLEE.py", pgm_config, pgm, stgy, str(mem_budget), small_time, ith_trial, str(iters), tool, d_name])
        os.system(cmd)


def Total_Coverage(pgm, load_config):
    gcov_files= glob.glob(load_config['gcov_file'])
    bid=1
    total_set=set()
    for fname in gcov_files: 
        if os.path.exists(fname):
            with open(fname, 'r') as f:
                lines = f.readlines()
                for l in lines:
                    if ("branch" in l):
                        total_set.add(bid)
                    bid=bid+1    
    return total_set


def Cal_Coverage(pgm, load_config):
    gcov_files= glob.glob(load_config['gcov_file'])
    bid=1
    cov_set=set()
    for fname in gcov_files: 
        if os.path.exists(fname):
            with open(fname, 'r') as f:
                lines = f.readlines()
                for l in lines:
                    if ("branch" in l) and ("never" not in l) and ("taken 0%" not in l):
                        cov_set.add(bid)
                    bid=bid+1    
    return cov_set


def Run_gcov(load_config, pgm, stgy, iters, tool, ith_trial, Data, d_name):
    result_dir="result_"+d_name 
    
    dir_name="/".join([result_dir, ith_trial+tool+"_"+pgm+"_"+stgy+"_tc_dir"])

    os.chdir(configs['e_dir']+"/"+dir_name+"/"+str(iters)+"__tc_dirs")
    testcases= glob.glob("*.ktest")
    testcases.sort(key=lambda x:float((x.split('.ktest')[0]).split('test')[1]))       
        
    early_testcases= glob.glob("*.early")
    early_testcases.sort(key=lambda x:float((x.split('.early')[0]).split('test')[1]))       
    
    # Maintains a quadruple of information used to generate each test-case tc.
    # tc -> [time_budget, pruning_ratio, rfeat_list, weight_vector]
    if iters != 0:
        with open('info', 'r') as f:
            args=f.readlines()[0].split()
            for arg in args:
                if "-max-time" in arg:
                    budget = int(arg.split('=')[1])

        for e_tc in early_testcases:
            with open(e_tc, 'r') as f:
                lines=f.readlines()
                rfeat_data=lines[0]
                if "rfeats: " in rfeat_data:
                    l_rfeats= (rfeat_data.split('rfeats: ')[1]).split()
                    l_rfeats= list(map(lambda s: int(s.strip()), l_rfeats))

                    pratio_data=lines[1]
                    pratio = str(int(pratio_data.split('pratio: ')[1]))
                        
                    wvector_data=lines[2]
                    wvector = (wvector_data.split('wvector: ')[1].split('\n'))[0]
                        
                    tc = str(iters)+"__tc_dirs/"+e_tc.split('.early')[0]+".ktest"

                    d_tc_data[tc]=[budget, pratio, l_rfeats, wvector]
        
        flag = 0
        for tc in testcases:
            tc= str(iters)+"__tc_dirs/"+tc
            if tc in d_tc_data.keys():
                recent_data=d_tc_data[tc]
                flag=1
            elif tc not in d_tc_data.keys() and flag==1:
                d_tc_data[tc]=recent_data
            else:
                continue

    os.chdir(configs['s_dir']+"/"+load_config['gcov_path'])
    rm_cmd = " ".join(["rm", load_config['gcov_file'], load_config['gcda_file']])
        
    # Calculate a set of covered branches corresponding to each test-case by running gcov.
    for tc in testcases:
        tc= str(iters)+"__tc_dirs/"+tc
        os.system(rm_cmd)
            
        run_cmd=[configs['b_dir']+"/bin/klee-replay", "./"+pgm, configs['e_dir']+"/"+dir_name+"/"+tc] 
        proc = subprocess.Popen(run_cmd, preexec_fn=os.setsid) 
        my_timer = Timer(1, Kill_Process, [proc, configs['e_dir']+"/"+dir_name+"/"+tc])
        try:
            my_timer.start()
            stdout, stderr = proc.communicate()
        finally:
            my_timer.cancel()
            
        gcov_cmd=" ".join(["gcov", "-b", load_config['gcda_file']])
        os.system(gcov_cmd)
            
        cov_set = Cal_Coverage(pgm,load_config)
        Data[tc]=cov_set   
    total_set = Total_Coverage(pgm,load_config)
    
    # ERASE ##############################################################         
    coverage= set()
    for tc in Data.keys():
        coverage = coverage | Data[tc]
    os.chdir(configs['e_dir']+"/"+dir_name)
    with open('learning_result', 'a') as l:
        l.write(pgm+","+stgy+"("+str(iters)+") : "+ str(len(coverage)) + "/"+str(len(total_set))+"\n")
    # ##### ##############################################################         
    
    return dir_name, Data


def SetCoverProblem(Data, iters):
    temp_Data = deepcopy(Data)
    topk_testcases = []
    intersect_set = set()
    
    total_size = len(temp_Data)
    
    # greedy algorithm for solving the set cover problem. 
    for i in range(1, total_size+1):
        sorted_list = sorted(temp_Data.items(), key=lambda kv:(len(kv[1])), reverse = True)
        topk_tc = sorted_list[0][0]
        topk_covset = sorted_list[0][1]

        if len(topk_covset) > 0: 
            topk_testcases.append(topk_tc)
            intersect_set = intersect_set | topk_covset 
            for tc in temp_Data.keys():
                temp_Data[tc] = temp_Data[tc] - intersect_set
        else:
            break
    # ERASE ##############################################################         
    result = set()
    for tc in topk_testcases:
        result= result | Data[tc]
    with open('learning_result', 'a') as l:
        l.write("# of effective test-cases: "+str(len(topk_testcases))+"\n")
    # ##### ##############################################################         
    
    return topk_testcases


def Feature_Extractor(pgm, stgy, dir_name, topk_testcases, ith_trial, iters):
    os.chdir(configs['e_dir']+"/"+dir_name)
    feat_set=set()
    
    Symbolic_arg="arg"
    Nonsymbolic_arg="const_arr"
    Eq_expr="Eq"
    Neq_expr="false"
    
    for tc in topk_testcases:
        tc_dir=tc.split('/')[0]
        tc=tc.split('/')[1]
        kquery=tc.split('.')[0]+".kquery"
        
        if os.path.exists(tc_dir+"/"+kquery):
            with open(tc_dir+"/"+kquery, 'r') as f:
                query_command_flag=0
                queries = f.readlines()
                for query in queries:    
                    if "query" in query:
                        query_command_flag=1
                    
                    if query_command_flag==1:
                        if ((Eq_expr in query) and (Symbolic_arg in query)
                        and (Neq_expr not in query) and (Nonsymbolic_arg not in query)):
                            feature=query.split('\n')[0]
                            if (len(feat_set)<200):
                                feat_set.add(feature)
                            else:
                                break
    with open(ith_trial+"homi_"+pgm+"_"+stgy+"_feature_data", 'w') as f:
        for feat in feat_set:
            f.write(feat+"\n")  
    return feat_set 


def PruningStgy_Generator(load_config, pgm, stgy, ith_trial, features, dir_name, topk_testcases, iters, Space_time):
    os.chdir(configs['e_dir']+"/"+dir_name)
    
    Space_ratio=Discrete_Space(S_ratio)
    
    # "wv_dir" is a set of pruning strategies (= weight vectors).
    wv_dir = "weights/" 
    if not os.path.exists(wv_dir):
        os.mkdir(wv_dir)
    
    wv_t_dir = "weights/"+str(iters+1)+"trials/" 
    if not os.path.exists(wv_t_dir):
        os.mkdir(wv_t_dir)
     
    with open(wv_dir+"/"+str(iters+1)+"feature_data", 'w') as f:
        for feat in features:
            f.write(feat+"\n")  
    
    if iters !=0:
        for wnum in range(1,51):
            key=str(iters)+"trials"+"/"+str(wnum)+".w"
            with open(wv_dir+"/"+key, 'r') as f:
                lines = f.readlines()
                current_wv = []
                for line in lines:
                    current_wv.append(line.split('\n')[0])
                tried_wv[key]=current_wv

    exploit_decisions=["exploit", "reverse_exploit", "explore"]
    Prob_exploit=[1,1,1] # set the same probablity for the three sampling methods
    policy= (random.choices(exploit_decisions, Prob_exploit))[0]
    
    d_prune_ratio={}
    d_prune_time={}
    d_budget={}
    
    l_budget_weight=[]
    
    list_size=len(Space_time)-1
    for idx in range(0,len(Space_time)):
        tw= float(Space_time[list_size] / Space_time[idx])
        l_budget_weight.append(tw)
    
    
    file_name=ith_trial+"homi_"+pgm+"_"+stgy+"_"
    pruning_ratio=[]
    budget_probability=[]
    
    # Sample the weight vector, time budget, and purning ratio via Exploration.
    # Use only the exploration method 10 times to collect the enough data.
    if iters<10 or policy=="explore": 
        policy="explore"
        # Randomly generate a set of pruning-strategies.
        for wv_id in range(1,51):
            fname = wv_t_dir  + str(wv_id) + ".w"
            weights = [str(random.uniform(lower, upper)) for _ in range(len(features))] 
            with open(fname, 'w') as f:
                for w in weights:
                    f.write(str(w) + "\n")
        
        # Randomly generate the time budget and pruning_ratio.
        small_time=str(random.choice(Space_time))

        with open (file_name+"pruning_ratio", 'w') as f:
            for i in range(0,51):
                ratio=random.choice(Space_ratio)
                f.write(str(ratio)+"\n") 


       
    # Sample the weight vector, time budget, and purning ratio via Exploitation or Reverse Exploitation.
    else:
        # Collect the learning data. 
        # Learning data: (1). each feature and the weight value, (2) time budget, (3). pruning-ratio)
        d_feat_wvs={}
        for tc in topk_testcases:
            trial_num = int(tc.split('__tc_dirs')[0])

            if (tc in d_tc_data.keys()) and (trial_num !=0):
                budget= d_tc_data[tc][0]
                pratio= d_tc_data[tc][1]
                l_rfeats= d_tc_data[tc][2] 
                wvector= d_tc_data[tc][3] 

                feats=wv_dir+str(trial_num)+"feature_data"
                feats_list=[]
                with open(feats, 'r') as ft:
                    feats_list=ft.readlines()
                    feats_list = list(map(lambda s: s.strip(), feats_list))
                
                wv_list = tried_wv[wvector]
                
                for idx in range(0, len(feats_list)):
                    if idx not in l_rfeats:
                        feature = feats_list[idx]
                        weight = float(wv_list[idx])
                        if feature in d_feat_wvs.keys():
                            d_feat_wvs[feature].append(weight)
                        else:
                            d_feat_wvs[feature]=[weight]
 
        # Sample the weight vector via Exploitation.
        if (policy=="exploit"):
            for wnum in range(1,51):
                fname = wv_t_dir  + str(wnum) + ".w"
                weights=[]
                for feature in features:
                    if feature in d_feat_wvs.keys():
                        mu = np.mean(d_feat_wvs[feature])
                        sigma = np.std(d_feat_wvs[feature])
                        set_size= len(set(d_feat_wvs[feature]))

                        if sigma==0 or set_size==1:
                            sigma=1
                            x = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
                        else:
                            x = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
                        w =x.rvs(1)[0]
                    else:
                        w=random.uniform(lower, upper)
                    weights.append(w)

                with open(fname, 'w') as f:
                    for w in weights:
                        f.write(str(w) + "\n")
        
        # Sample the weight vector via Reverse_Exploitation.
        else:
            for wnum in range(1,51):
                fname = wv_t_dir  + str(wnum) + ".w"
                weights=[]
                for feature in features:
                    if feature in d_feat_wvs.keys():
                        mu = np.mean(d_feat_wvs[feature])
                        sigma = np.std(d_feat_wvs[feature])
                        set_size= len(set(d_feat_wvs[feature]))

                        if sigma==0 or set_size==1:
                            sigma=1
                            x = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
                        else:
                            x = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
                        w =x.rvs(1)[0]
                    
                        cand_w_list= list(np.random.uniform(lower, upper, 20))
                        contrary_w=0
                        diff=0
                    
                        for cand_w in cand_w_list:
                            if abs(cand_w - w) > diff:
                                diff=abs(cand_w -w)
                                contrary_w=cand_w
                        w = contrary_w
                    
                    else:
                        w=random.uniform(lower, upper)
                    weights.append(w)

                with open(fname, 'w') as f:
                    for w in weights:
                        f.write(str(w) + "\n")

        # Sample the time budget.
        budget_probability=[]
        for time in Space_time:
            idx= Space_time.index(time)
            tw= l_budget_weight[idx]
            tried_tb=1
            if time in d_budget.keys():
                if time in d_tried_budget.keys():
                    tried_tb=d_tried_budget[time]
                else:
                    tried_tb=1
                num = int(d_budget[time]*tw/tried_tb)
                budget_probability.append(num)
            else:
                budget_probability.append(1)
        small_time = str((random.choices(Space_time, budget_probability))[0]) 
        
        # Sample the pruning ratio.
        pratio_probability=[]
        for r in Space_ratio:
            r= str(r)
            if r in d_prune_ratio.keys():
                pratio_probability.append(d_prune_ratio[r])
            else:
                pratio_probability.append(1)
        with open (file_name+"pruning_ratio", 'w') as f:
            for i in range(0,51):
                ratio = (random.choices(Space_ratio, pratio_probability))[0]
                f.write(str(ratio)+"\n") 
        

    with open('topk_tcs_data', 'a') as f:
        f.write(str(iters+1)+"-> topk-tcs: "+str(len(topk_testcases))+")\n")
        f.write("policy: "+policy+"\n")
        f.write("ratio: "+str(d_prune_ratio)+"\n")
        f.write("budget: "+str(d_budget)+"\n")
        f.write("tried_budget_counter: "+str(d_tried_budget)+"\n")
        f.write("budget_prob: "+str(budget_probability)+"\n")
    

    return small_time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pgm_config")
    parser.add_argument("total_time",help='[3600(s),18000(s)]')
    parser.add_argument("tool",help='[homi, pureklee]')
    parser.add_argument("search_heuristic",help='[nurs:covnew, random-path, ..]')
    parser.add_argument("ith_trial",help='[1,2,3,..]')
    
    args = parser.parse_args()
    pgm_config = args.pgm_config
    load_config = Load_Pgm_Config(args.pgm_config)
    stgy = args.search_heuristic
    total_time = int(args.total_time)
    tool = args.tool    
    ith_trial = args.ith_trial    
    
    iters=0
    Data = {} # Data denotes the accumulated data.
    Space_time=Discrete_Space(S_time) # Initialize the sample space for the time budget.
    small_time=str(max(Space_time))

    pgm=load_config['pgm_name'] 
    d_name="All" 

    if tool=="homi":         
        # Homi performs the general symbolic execution without state-pruning on the first iteration.
        Run_KLEE(pgm_config, pgm, stgy, total_time, small_time, ith_trial, iters, tool, d_name, Space_time)  
        
        while iters<100:
            dir_name, Data = Run_gcov(load_config, pgm, stgy, iters, tool, ith_trial, Data, d_name)
            topk_testcases = SetCoverProblem(Data, iters)
            features = Feature_Extractor(pgm, stgy, dir_name, topk_testcases, ith_trial, iters)
            small_time= PruningStgy_Generator(load_config, pgm, stgy, ith_trial, features, dir_name, topk_testcases, iters, Space_time)
            
            iters=iters+1
            Run_KLEE(pgm_config, pgm, stgy, total_time, small_time, ith_trial, iters, tool, d_name, Space_time)
    else:
        for num in range(1,100):
            Run_KLEE(pgm_config, pgm, stgy, total_time, small_time, ith_trial, iters, tool, d_name, Space_time) 
            iters=iters+1

if __name__ == '__main__':
    main()

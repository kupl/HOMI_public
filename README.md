# HOMI

HOMI is a tool that performs symbolic execution while maintaining only a small number of **promising states**  via online learning. This tool is implemented on the top of [KLEE][klee], a publicly available symbolic execution tool for testing C programs. For more technical details, please read our [paper](./FSE20.pdf).

#### Table of Contents

* [Installation](#Installation)
* [Artifact](#Artifact)
  * [Running Homi](#Running-Homi)  
  * [Running pure KLEE](#Running-pure-KLEE)  
  * [Visualizing the experimental results](#Visualizing-the-experimental-results)  
* [How to run Homi on arbitary C open-source programs](#How-to-run-Homi-on-arbitary-C-open-source-programs) 
* [How to extend the standard KLEE](#How-to-extend-the-standard-KLEE) 
## Installation

We provide a simple way to install HOMI:
* A VirtualBox image containing all resources to reproduce the main experimental results in our paper: [FSE20_HOMI_artifacts.tar.gz](https://drive.google.com/file/d/1xD31tZk9bitZSkkbvB3wbgw2nsBMprGd/view?usp=sharing)
   * Ubuntu ID/PW: homi/homi

Please see [INSTALL.md](./INSTALL.md) for full installation instructions.

## Artifact

We design a shorter experiment that performs **Homi** and **pure KLEE**, respectively, on our benchmark **trueprint-5.4** for an hour once. This is because all the experiments in our paper take at least hundreds of hours; for instance, to reproduce the experimental result for a benchmark in Figure 1 of our paper with a single core, it takes a total of 150 hours (5 hours * 6 testing techniques * 5 iterations). In this section, we will show the commands to run the short experiment. 

### Running Homi.

The following command will run our tool, **Homi** with the given search heuristic, to test one of our benchmarks, **trueprint-5.4**, for an hour once.

```bash
$  cd ~/Homi/script
$  python3.7 Homi.py pgm_config/1trueprint.json 3600 homi nurs:md2u 1
```

Each argument of the last command means as follows:
-	**pgm_config/1trueprint.json** : A json file to describe the benchmark
-	**3600**:  Total time budget (sec)
-	**homi** : Tool to use. We offer 2 options. (homi, pureklee)
-	**nurs:md2u**: Search heuristic. We can use 9 search heuristics offered in KLEE (e.g., nurs:cpicnt, nurs:md2u, nurs:covnew, random-path, ...).  For more details, see this [klee document][sh_link].
-	**1** : i-th trial. 

If the Homi scipt successfully ends, you can see the following command:

```sh
#############################################
################Time Out!!!!!################
#############################################
```

### Running pure KLEE.
if you want to run **pure KLEE** with the given search heuristic on  **trueprint-5.4**, the command is as:

```bash
$  cd ~/Homi/script
$  python3.7 Homi.py pgm_config/1trueprint.json 3600 pureklee nurs:md2u 1
```

### Visualizing the experimental results. 

To visualize the experimental results, we first use **gcov**, one of the most popular tools for measuring code coverage and then generate two graphs in terms of the number of covered branches and candidate states. To do so, the command is as:

```bash
$  cd ~/Homi/script
$  python3.7 graph_generate.py pgm_config/1trueprint.json ../experiments/result_All/
```

Each argument of the last command means as follows:

- **pgm_config/1trueprint.json** : A json file to describe the benchmark

  - Specifically, a json file consists of 6 information as follows: 

   {   
    "pgm_name": "trueprint",   
    "pgm_dir": "../benchmarks/trueprint-5.4/obj-llvm/",   
    "exec_dir": "/src",   
    "gcov_path": "../benchmarks/trueprint-5.4/1obj-gcov/src/",   
    "gcov_file": "../\*/\*.gcov",   
    "gcda_file": "../\*/\*.gcda"      
   }

    First, "pgm_name" represents the program name, and "pgm_dir" and "exec_dir" are used to denote the directory where  the LLVM bitcode is located. "gcov_path" denotes the directory where the gcov files are generated when running gcov.  Lastly, "gcov_file" and "gcda_file" are used to run gcov and calculate the set of branches covered by each test-case.   

    Of course, depending on the benchmark, this information such as "exec_dir" and "gcov_file" can be different. For example, the json file for gawk is:   

    {   
     "pgm_name": "gawk",   
     "pgm_dir": "../benchmarks/gawk-3.1.4/obj-llvm/",   
     "exec_dir": "/",   
     "gcov_path": "../benchmarks/gawk-3.1.4/1obj-gcov/",   
     "gcov_file": "./\*.gcov",   
     "gcda_file": "./\*.gcda"   
    }

-	**../experiments/result_All/**:  a directory containing test-cases generated by each tool. 

if the script  successfully ends, you can see the following command:

```bash
#############################################
trueprint_cov.pdf produced
trueprint_statenum.pdf produced
#############################################
```
Now, you can find ```trueprint_cov.pdf``` and ```trueprint_statenum.pdf``` in ```~/Homi/script/``` folder. 

## How to run Homi on arbitary C open-source programs.

We also provide the detailed instructions to run our tool, Homi, for arbitary open-source C programs. 

**1)** The first step is to **download** the program and **build** it with gcov and LLVM. 

```bash
# Download an open-source C program you want.
$  cd ~/Homi/benchmarks/
$  wget https://ftp.gnu.org/gnu/enscript/enscript-1.6.6.tar.gz
$  tar -zxvf enscript-1.6.6.tar.gz

# Build the program with gcov
$  cd ~/Homi/benchmarks/enscript-1.6.6
$  mkdir 1obj-gcov
$  cd 1obj-gcov
$  ../configure --disable-nls CFLAGS="-g -fprofile-arcs -ftest-coverage"
$  make

# Build the program with LLVM
$  cd ~/Homi/benchmarks/enscript-1.6.6
$  mkdir obj-llvm
$  cd obj-llvm
$  CC=wllvm ../configure --disable-nls CFLAGS="-g -O1 -Xclang -disable-llvm-passes -D__NO_STRING_INLINES  -D_FORTIFY_SOURCE=0 -U__OPTIMIZE__"
$  make
$  cd src
# This command extracts the LLVM bitcode from a build project.
$  find . -executable -type f | xargs -I '{}' extract-bc '{}'
```

**2)** The second step is to **generate the json file** to describe the program as follows:

 ```bash
$  cd ~/Homi/script
$  python3.7 json_generator.py enscript-1.6.6 1
$  # "Successfully generate 1enscript.json in the `pgm_config' directory"
 ```

**3)** Lastly, **run Homi** with the search heuristic (nurs:cpicnt) on **enscript-1.6.6** for **30 minutes** as:

 ```bash
$  cd ~/Homi/script
$  python3.7 Homi.py pgm_config/1enscript.json 1800 homi nurs:cpicnt 1
 ```

[klee]: https://klee.github.io/releases/docs/v2.0/
[sh_link]: https://klee.github.io/releases/docs/v2.0/docs/options/#search-heuristics

## How to extend the standard KLEE.

We extended the standard KLEE in order to prune the states based on the given n features and n-dimensional weight vectors. We mainly implemented it in the files 'Homi/klee/lib/Core/Executor.cpp' and 'Homi/klee/lib/Core/Executor.h'. That is, our pruning technique is not available in the original KLEE. 
 
Our new implementation is able to prune states through three steps; the first step is to transform each state into a n-dimensional boolean vector when given n features. Second, it scores each state with a given weight vector. Third, it prunes the states based on the scores and the pruning ratio. 

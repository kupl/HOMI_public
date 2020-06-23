import json, os, glob, argparse


def json_maker(pgm, dirs, iters):
    pgm_fname=iters+pgm.split('-')[0]
    pgm_name=pgm.split('-')[0]
    with open("pgm_config/"+pgm_fname+".json", 'w') as f:
        f.write('{\n')
        f.write("\"pgm_name\": \""+pgm_name+"\",\n")
        f.write("\"pgm_dir\": \"../benchmarks/"+pgm+"/obj-llvm/\",\n")
        f.write("\"exec_dir\": \"/"+dirs+"\",\n")
        if "gawk" not in pgm:
            f.write("\"gcov_path\": \"../benchmarks/"+pgm+"/"+iters+"obj-gcov/"+dirs+"/\",\n")
            f.write("\"gcov_file\": \"../*/*.gcov\",\n")
            f.write("\"gcda_file\": \"../*/*.gcda\"\n")
        else:
            f.write("\"gcov_path\": \"../benchmarks/"+pgm+"/"+iters+"obj-gcov/\",\n")
            f.write("\"gcov_file\": \"./*.gcov\",\n")
            f.write("\"gcda_file\": \"./*.gcda\"\n")
        f.write('}\n')
    print ("\"Successfully generate "+pgm_fname+".json in the `pgm_config' directory\"")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pgm")
    parser.add_argument("iters")
    
    args = parser.parse_args()
    pgm = args.pgm
    iters= int(args.iters)

    for i in range(1, iters+1):
        json_maker(pgm, "src", str(i))
 

if __name__ == '__main__':
    main()


import argparse
#Command Line Arguments

ap = argparse.ArgumentParser(
    description='predict-file')

ap.add_argument('--num',default = 5,  action = "store", type=int)

pa = ap.parse_args()

if __name__ == '__main__':
    print ("Hello from Jupyter. Number entered: ", pa.num)

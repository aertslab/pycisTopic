import pickle
import sys
import argparse
import os
import json 

from pycisTopic.cistopic_class import *
from pycisTopic.lda_models import *

def make_argument_parser():
    """
    Creates an ArgumentParser to read the options for this script from
    sys.argv
    """
    parser = argparse.ArgumentParser(
        description="Run topic models.",)
    parser.add_argument('--inputcisTopic_obj', '-i', type=str, required=True,
                        help='Path to cisTopic object pickle file.')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Path to save final model list.')
    parser.add_argument('--n_topics', '-nt', type=str, required=True, nargs='+',
                        help='Txt file containing selected topic id.')
    parser.add_argument('--n_cpu', '-c', type=int, required=True,
                        help = 'Number of cores')
    parser.add_argument('--n_iter', '-it', type=int, required=False, default=150,
                        help = 'Number of iterations')
    parser.add_argument('--alpha', '-a', type=int, required=False,  default=50,
                        help='Alpha value')
    parser.add_argument('--alpha_by_topic', '-abt', type=str, required=False, default=True,
                        help = 'Whether the alpha value should by divided by the number of topics')
    parser.add_argument('--eta', '-e', type=float, required=False, default=0.1,
                        help='Eta value.')
    parser.add_argument('--eta_by_topic', '-ebt', type=str, required=False, default=False,
                        help = 'Whether the eta value should by divided by the number of topics')
    parser.add_argument('--save_path', '-sp', type=str, required=False,
                        default=None, help='Whether intermediate models should be saved')
    parser.add_argument('--seed', '-s', type=int, required=False,
                        default=555, help='Seed for ensuring reproducibility')
    parser.add_argument('--temp_dir', '-td', type=str, required=False,
                        default=None, help='TMP directory')
    return parser


def main():
    """
    The main executable function
    """
    
    parser = make_argument_parser()
    args = parser.parse_args()
    filename = args.inputcisTopic_obj
    infile = open(filename, 'rb')
    cisTopic_obj = pickle.load(infile)
    infile.close()
    print('Input cisTopic_object:', filename)
    
    output = args.output
    print('Output file:', output)
    
    n_topics = args.n_topics
    n_topics = list(map(int, n_topics[0].split(',')))
    print('Number of topics:', n_topics)
    
    alpha=args.alpha
    print('Alpha:', alpha)
    
    alpha_by_topic=args.alpha_by_topic
    print('Divide alpha by the number of topics:', alpha_by_topic)
    
    eta=args.eta
    print('Eta:', eta)
    
    eta_by_topic=args.eta_by_topic
    print('Divide eta by the number of topics:', eta_by_topic)
    
    n_iter=args.n_iter
    print('Number of iterations:', n_iter)
    
    n_cpu=args.n_cpu
    print('Number of cores:', n_cpu)
    
    save_path=args.save_path
    print('Path to save intermediate files:', save_path)
    if save_path == 'None':
        save_path = None
    
    random_state=args.seed
    print('Seed:', random_state)
    
    temp_dir=args.temp_dir
    print('TMP directory:', temp_dir)

    # Run models
    print('Running models')
    print('--------------')
    models=run_cgs_models(cisTopic_obj,
                        n_topics=n_topics,
                        n_cpu=n_cpu,
                        n_iter=n_iter,
                        random_state=random_state,
                        alpha=alpha,
                        alpha_by_topic=alpha_by_topic,
                        eta=eta,
                        eta_by_topic=eta_by_topic,
                        save_path=save_path,
                        _temp_dir=temp_dir)

    # Save
    with open(output, 'wb') as f:
        pickle.dump(models, f)

if __name__ == "__main__":
    main()


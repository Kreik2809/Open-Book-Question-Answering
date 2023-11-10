import argparse
import json

from trainer import train

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='../configs/config.json')
    args = parser.parse_args()
    config = json.load(open(args.config, 'r'))  

    #Gather all hyperparameters
    experiement_name = config['experiment_name']
    dataset_id = config['dataset_id']
    model_id = config['model_id']
    device = config['device']
    batch_size = config['batch_size']
    epochs = config['epochs']
    lr = config['lr']
    max_length = config['max_length']
    stride = config['stride']
    max_answer_length = config['max_answer_length']
    n_best = config['n_best']
    seed = config['seed']
    
    #Launch training
    train(experiement_name, dataset_id, model_id, device, batch_size, epochs, lr, max_length, stride, max_answer_length, n_best, seed)
    
if __name__ == '__main__':
    main()
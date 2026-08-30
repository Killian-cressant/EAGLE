import argparse
import ast

def get_args_from_input():
	parser = argparse.ArgumentParser(description='modify network parameters', argument_default=argparse.SUPPRESS)


	#optimizer params
	parser.add_argument('--learning_rate', type=float, default=1e-5, help='learning rate')
	parser.add_argument('--max_epochs', type=int, default=90, help='maximum number of epochs for training')
	parser.add_argument('--eval_every', type=int, default=1, help='calculate validation/test accuracy every X epochs')
	parser.add_argument('--stopping_threshold', type=float, default=1.001, help="model perceives no improvement when it does worse than (best loss) * T")
	parser.add_argument('--patience', type=int, default=50, help='model stops training after P epochs with no improvement')
	parser.add_argument('--dropout', type=float, default=0.2, help='layer dropout probability')
	parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay added to loss function')
	parser.add_argument('--batch_size', type=int, default=1, help='number of samples in each training batch')
	parser.add_argument('--num_trials', type=int, default=1, help='number of times the network is trained')
	parser.add_argument('--kl_warmup_epochs', type=int, default=10, help='number of epochs to warm up the KL divergence')


	#model params
	parser.add_argument('--window_size', type=int, default=25, help='size of the sliding window')
	parser.add_argument('--horizon', type=int, default=25, help='stride for sliding window')
	parser.add_argument('--layer_type', default='GCN', help='type of layer in GNN (GCN, GIN, GAT, etc.)')
	parser.add_argument('--input_dim', type=int, default=None, help='input dimension')
	parser.add_argument('--output_dim', type=int, default=8, help='output dimension')
	parser.add_argument('--hidden_dim', type=int, default=16, help='width of hidden layer')
	parser.add_argument('--hidden_layers', type=ast.literal_eval, default=None, help='list containing dimensions of all hidden layers')
	parser.add_argument('--num_layers', type=int, default=5, help='number of hidden layers')
	parser.add_argument('--temporal_window', type=int, default=1, help='size of the temporal window')
	parser.add_argument('--threshold', type=float, default=0.4, help='threshold for edge selection')
	#env params
	parser.add_argument('--device', default=0, type=int, help='the gpu to use')

	

	#experiments
	parser.add_argument('--dataset', type=str, default='smd', help='name of dataset to use')
	parser.add_argument('--experiment_name', type=str, default='FTS', help='can be \'FTS\' (for Full time Series), \'LEED\' (For LEED version) or \'LTS\' (For Latent Time series)')
	parser.add_argument('--graph_type', type=str, default='corr', help='type of graph to build, can be cosi or corr')
	parser.add_argument('--top_k', type=int, default=15, help='top k nodes to be selected for expansion')
	parser.add_argument('--top_k_centrality', type=int, default=10, help='top k nodes to be selected for centrality computation')
	parser.add_argument('--top_center', type=int, default=100, help='top percentage k nodes to be selected for centrality computation')
	parser.add_argument('--metric', type=str, default='all', help='metric to evaluate the model')
	parser.add_argument('--pos_weight_exponent', type=float, default=1.0, help='exponent for positive weight calculation')

	arg_values = parser.parse_args()
	return arg_values
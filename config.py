import argparse

# data_root
#    - ref (contain older refcoco, refcoco+)
#        - refcoco
#        - refcoco+
#        - refcocog
#        - refgta
#    - ref2 (contain new refcoco, refcoco+)
#        - refcoco
#        - refcoco+
#        - refcocog
#        - refgta

# save_dir
#    - prepro
#        - refcoco_unc
#        - refcoco+_unc
#        - refcocog_google
#        - refgta_utokyo
#    - model
#        - refcoco_unc
#        - refcoco+_unc
#        - refcocog_google
#        - refgta_utokyo

# coco_image_root
#    - COCO_train2014_000000000009.jpg
#    - COCO_train2014_0000000000025.jpg
#    - ...

# gta_image_root
#    - dont_specify
#        - final
#           - GTA_CvMod_2018-04-25_18-07-48_final_0000011640.png
#           - ...
#    - black_wearing
#    - white_wearing



def parse_opt():
	parser = argparse.ArgumentParser()
	
	parser.add_argument('--data_root', type=str, default='',
						help='data folder containing four datasets.')
	parser.add_argument('--coco_image_root', type=str, default='',
						help='data folder containing MSCOCO images')
	parser.add_argument('--gta_image_root', type=str, default='',
						help='data folder containing gta images')
	parser.add_argument('--save_dir', default='',
						help='save directory')
    
	parser.add_argument('--data_json', default='data.json', help='converted json file')
	parser.add_argument('--data_h5', default='data.h5', help='converted h5 file')
	parser.add_argument('--old', '-old', action='store_true', help='the each test set of RefCOCO and RefCOCO+ have 2 versions')
	# options
	parser.add_argument('--dataset', '-d', default='refcoco', type=str, help='refcoco/refcoco+/refcocog/refgta')
	parser.add_argument('--splitBy', '-s', default='unc', type=str, help='unc/google/utokyo')
	
	# preprocess options
	parser.add_argument('--max_length', type=int, help='max length of a caption')  # refcoco 10, refclef 10, refcocog 20, refgta 20
	parser.add_argument('--word_count_threshold', default=5, type=int,
						help='only words that occur more than this number of times will be put in vocab')
	# image feature extraction
	parser.add_argument('--image_feats', default='sp_image_feats.npy', help='image, Variable feats file')
	parser.add_argument('--sp_ann_feats', default='sp_ann_feats.npy', help='spatial ann feats file')
	parser.add_argument('--ann_feats', default='ann_feats.npy', help='ann feats file')
	parser.add_argument('--ann_shapes', default='shapes.npy', help='ann shapes file')

	# pretraining settings
	parser.add_argument('--sample_ratio', type=float, default=0.5)
	parser.add_argument('--sample_neg', type=int, default=1)
	parser.add_argument('--grad_clip', type=float, default=0.1)
	parser.add_argument('--ranking_lambda', type=float, default=0.1)
	parser.add_argument('--seq_per_ref', type=int, default=3)
	parser.add_argument('--learning_rate_decay_start', type=int, default=8000)
	parser.add_argument('--learning_rate_decay_every', type=int, default=8000)
	parser.add_argument('--optim_epsilon', type=float, default=1e-8)
	parser.add_argument('--losses_log_every', type=int, default=25)
	parser.add_argument('--max_iter', type=int, default=-1)
	parser.add_argument('--save_checkpoint_every', type=int, default=2000)
	## language encoder
	parser.add_argument('--learning_rate', type=float, default=4e-4)
	parser.add_argument('--optim_alpha', type=float, default=0.8)
	parser.add_argument('--optim_beta', type=float, default=0.999)
	## visual encoder
	parser.add_argument('--ve_learning_rate', type=float, default=4e-5)
	parser.add_argument('--ve_optim_alpha', type=float, default=0.8)
	parser.add_argument('--ve_optim_beta', type=float, default=0.999)
	
	parser.add_argument('--pretrained_w', '-pw', action='store_true')
	parser.add_argument('--word_emb_path', default='word_emb.npy')

	# training settings
	parser.add_argument('--generation_weight', type=float, default=1)
	parser.add_argument('--vis_rank_weight', type=float, default=1)
	parser.add_argument('--lang_rank_weight', type=float, default=0)
	parser.add_argument('--embedding_weight', type=float, default=1)
	parser.add_argument('--rank_lam', type=float, default=0.4)
	parser.add_argument('--lm_margin', type=float, default=1)
	parser.add_argument('--emb_margin', type=float, default=0.1)
	
	parser.add_argument('--id', '-id',default='sp')
	parser.add_argument('--id2', '-id2',default='')
	parser.add_argument('--ranking', '-r', action='store_true', help='use ranking for RefGTA')

	# running settings
	parser.add_argument('--batch_size', type=int, default=32)
	parser.add_argument('--gpu_id', '-g', type=int, default=0)
	
	## test setting
	parser.add_argument('--split', '-split',default='testA')
	parser.add_argument('--mode','-mode',type=int, default=1, help='0: speaker, 1: reinforcer, 2: ensemble')
	parser.add_argument('--lamda','-lam',type=float, default=1)
	parser.add_argument('--beam_width','-beam',type=int, default=10)
	parser.add_argument('--write_result', default=0, type=int)
	args = parser.parse_args()
	return args
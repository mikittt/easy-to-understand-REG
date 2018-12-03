from tokenizer.ptbtokenizer import PTBTokenizer
from bleu.bleu import Bleu
from meteor.meteor import Meteor
from rouge.rouge import Rouge
from cider.cider import Cider
from cider_r.cider_r import CiderR

"""
Input: refer and Res = [{ref_id, sent}]

Things of interest
evalRefs  - list of ['ref_id', 'CIDEr', 'Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'ROUGE_L', 'METEOR']
eval      - dict of {metric: score}
refToEval - dict of {ref_id: ['ref_id', 'CIDEr', 'Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'ROUGE_L', 'METEOR']}
"""

class CrossEvaluation:
	def __init__(self, refer, preds):
		"""
        :param refer: refer class of current dataset
        :param Res: [{'ref_id', 'sent'}]
        """
		self.refer = refer	 			# refer loader
		self.preds = preds  			# [{ref_id, sent}]
		self.Evals = {}  				# id1_id2 -> {scorer: score} 
		self.ref_to_evals = {}

	def make_ref_to_evals(self):
		"""
		We will convert self.Evals = {pair_id: {method: sc}} to
		ref_to_evals = {source_ref_id: {cross_ref_id: {method: sc}}}
		so that ref_to_evals[ref_id1][ref_id2] means ref_id1's prediction 
		on ref_id2's gd sents.
		"""
		ref_to_evals = {}
		for pair_id in self.Evals:

			source_ref_id = int(pair_id[:pair_id.find('_')])
			cross_ref_id  = int(pair_id[pair_id.find('_')+1:])
			method_to_sc = self.Evals[pair_id]

			if source_ref_id not in ref_to_evals:
				ref_to_evals[source_ref_id] = {}
			ref_to_evals[source_ref_id][cross_ref_id] = method_to_sc
		self.ref_to_evals = ref_to_evals


	def Xscore(self, scorer='CIDEr'):
		# compute CIDEr difference
		sc = 0
		n = 0
		for ref_id in self.ref_to_evals:
		    # load eval result
		    evals = self.ref_to_evals[ref_id] # cross_ref_id: {method: sc}
		    # check self_sc, max_cross_sc
		    self_sc  = evals[ref_id][scorer]
		    cross_ref_ids = [cross_ref_id for cross_ref_id in evals.keys() if cross_ref_id != ref_id] 
		    cross_scs = [evals[cross_ref_id][scorer] for cross_ref_id in cross_ref_ids]
		    if len(cross_scs) > 0:
		        max_cross_sc = max(cross_scs)
		    else:
		        max_cross_sc = 0
		    # compute 
		    if self_sc > max_cross_sc: 
		    	n += 1
		    sc += (self_sc - max_cross_sc)

		sc /= len(self.ref_to_evals)
		n = n*1.0/len(self.ref_to_evals)
		print ('average (self_sc - max_cross_sc) = %.3f' % sc)
		print ('%.2f%% genenerated sentence has higher %s using groud-truth expressions' % (n*100.0, scorer))


	def cross_evaluate(self):
		"""
		We will evaluate how relevant is the generated expression to the ground-truth expressions,
		and how different it is to the expressions of the other objects within the same image.
		Thus, the prerequisite is the dataset is split by image_id, and each ann has multiple
		expressions, e.g., our new RefCOCO dataset whose tesing object has ~10 expressions.
		We first compute score on sc_ii = (sent_i, gd_sents_i), then compute score on 
		sc_ij = (sent_i, gd_sents_j), the margin of max(0, sc_ii - sc_ij) will be considered
		as final score.
		Speficically, we choose METEOR and CIDEr for this kind of evaluation.

		For doing so, we need to prepare ref_to_gts and ref_to_res. As we want to do cross evaluation,
		our key would be paird_id, i.e., "ref_id1_to_ref_id2", e.g, '123_456', then 
		input:
		- Gts[123_456] = [456's gd sents]
		- Res[123_456] = [123's predicted sents]. 
		return:
		- ref_to_eval[123_456] = {method: score}, which measures 123's generation over 456's gd-sents
		Note, we also compute score of 123_123
		
		We will use "sids" and "cids" to donate source_ref_ids and cross_ref_ids.
		"""
		source_ref_ids = [pred['ref_id'] for pred in self.preds]
		Preds = {pred['ref_id']: pred['sent'] for pred in self.preds }

		# construct pair_id, which is [source_ref_id]_[target_ref_id], i.e, 123_456
		Gts = {}
		Res = {}
		for source_ref_id in source_ref_ids:
			image_id = self.refer.Refs[source_ref_id]['image_id']
			cross_refs = self.refer.imgToRefs[image_id]  # including source_ref itself
			for cross_ref in cross_refs:
				pair_id = str(source_ref_id)+'_'+str(cross_ref['ref_id'])
				Res[pair_id] = [Preds[source_ref_id]]
				Gts[pair_id] = [sent['sent'] for sent in cross_ref['sentences']]

		# tokenize
		print ('tokenization...')
		tokenizer = PTBTokenizer()
		Gts = tokenizer.tokenize(Gts)	
		Res = tokenizer.tokenize(Res)

		# set up scorers
		print( 'setting up scorers...')
		scorers = [
			(Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
			(Meteor(),"METEOR"),
			(Rouge(), "ROUGE_L"),
			(Cider(), "CIDEr"),
		]

		# compute scores
		for scorer, method in scorers:
			print( 'computing %s score...'%(scorer.method()))
			score, scores = scorer.compute_score(Gts, Res)
			if type(method) == list:
				for sc, scs, m in zip(score, scores, method):
					self.setEvals(scs, Gts.keys(), m)
					print ("%s: %0.3f"%(m, sc))
			else:
				self.setEvals(scores, Gts.keys(), method)
				print ("%s: %0.3f"%(method, score))


	def setEvals(self, scores, pair_ids, method):
		for pair_id, score in zip(pair_ids, scores):
			if not pair_id in self.Evals.keys():
				self.Evals[pair_id] = {}
				self.Evals[pair_id]['pair_id'] = pair_id
			self.Evals[pair_id][method] = score


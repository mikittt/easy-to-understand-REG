import chainer.functions as F

def emb_crits(emb_flows, margin, vlamda=1, llamda=1):
    vis_loss = F.mean(F.relu(margin+emb_flows['vis'][1]-emb_flows['vis'][0]))
    lang_loss = F.mean(F.relu(margin+emb_flows['lang'][1]-emb_flows['lang'][0]))
    return vlamda*vis_loss + llamda*lang_loss
    
def lm_crits(lm_flows, num_labels, margin, vlamda=1, llamda=0, langWeight=1):
    total_loss = 0
    n = 0
    lang_loss = 0
    Tprob = lm_flows['T']
    lang_num = num_labels['T']
    total_loss -= F.sum(Tprob)/(sum(lang_num)+len(lang_num))*langWeight
    if vlamda==0 and llamda==0:
        return total_loss
    
    def triplet_loss(flow, num_label):
        pairGenP = flow[0]
        unpairGenP = flow[1]
        pairSentProbs = F.sum(pairGenP,axis=0)/(num_label+1)
        unpairSentProbs = F.sum(unpairGenP,axis=0)/(num_label+1)
        trip_loss = F.mean(F.relu(margin+unpairSentProbs-pairSentProbs))
        return trip_loss
    
    if vlamda!=0:
        vloss = triplet_loss(lm_flows['visF'], num_labels['T'])
        total_loss += vlamda*vloss
    if llamda!=0:
        lloss = triplet_loss(lm_flows['langF'], num_labels['F'])
        total_loss += llamda*lloss
    return total_loss
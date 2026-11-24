"""Online updates to four pretrained decoder rows, episodic baselines and active probing."""
import numpy as np

METHODS = ('uniform','frozen','neighbor','kernel','weights')
CUTS = (0,8,32,96)
BUDGETS = (1,3,6)
COUNT = 16


def softmax(x):
    p = np.exp(x-x.max())
    return p/p.sum()


class History:
    def __init__(self, key, settings):
        self.key = key
        self.settings = settings
        self.delta = np.zeros((4,129))
        self.items = []
        self.targets = []

    def prior(self, vector, logits):
        if self.key == 'uniform':
            return np.full(4,.25)
        if self.key == 'weights':
            return softmax(logits + self.delta @ vector)
        base = softmax(logits)
        if self.key == 'frozen' or not self.items:
            return base
        sim = np.asarray(self.items) @ (vector[:-1]/np.linalg.norm(vector[:-1]))
        if self.key == 'neighbor':
            votes = self.targets[np.argmax(sim)]
        else:
            votes = softmax(sim/self.settings['temperature']) @ np.asarray(self.targets)
        mix = self.settings['mix']
        return (1-mix)*base + mix*votes

    def save(self, vector, logits, target):
        if self.key == 'weights':
            error = target-self.prior(vector,logits)
            self.delta += self.settings['rate']*np.outer(error,vector)
        elif self.key in ('kernel','neighbor'):
            self.items.append(vector[:-1]/np.linalg.norm(vector[:-1]))
            self.targets.append(target.copy())

    def bytes(self):
        if self.key=='weights':
            return self.delta.nbytes
        return sum(a.nbytes for a in self.items+self.targets)


def features(dataset):
    # A fixed affine reparameterization makes the effective decoder update depend on
    # the name rather than the shared prompt direction. Center is practice-only.
    mean = dataset['practice_h'].mean(0).astype(float)
    scale = np.linalg.norm(dataset['practice_h']-mean,axis=1).mean()
    result = {}
    for domain in ('practice','dev','number','text'):
        h = dataset[domain+'_h'].astype(float)-mean
        v = np.column_stack((h/scale, np.full(len(h),.25)))
        result[domain] = v, dataset[domain+'_logits'].astype(float)
    return result


def posterior(prior, likelihood, active):
    probs = prior*likelihood*active
    return probs/probs.sum()


def probe(weights, bank, used):
    entropy_distance = abs(weights @ bank - .5)
    entropy_distance[used] = np.inf
    return int(np.argmin(entropy_distance))


def observe(likelihood, bank, idx, truth):
    return likelihood*np.where(bank[:,idx]==truth,.95,.05)


def teacher(target, matrix, practice=False):
    bank = matrix[:,:COUNT]
    like = np.ones(16)
    ids = []
    work = 0
    for step in range(6):
        p = like/like.sum()
        field = p.reshape(4,4).sum(1)
        if practice and len(ids)>=2 and field.max()>=.85:
            return ids,field,work
        if practice and step==5:
            break
        idx = probe(p,bank,ids)
        ids.append(idx)
        like = observe(like,bank,idx,bank[target,idx])
        work += 16*(COUNT+1)
    return ids,None,work


def correct(weights, checks, truth):
    return int(np.sum((weights @ checks >= .5)==truth))


def replay(prior, target, matrix, trace, corrupt=False):
    bank,checks = matrix[:,:COUNT],matrix[:,COUNT:]
    p = np.repeat(prior/4,4)
    scores = [correct(p,checks,checks[target])]
    for i,idx in enumerate(trace):
        label = int(bank[target,idx]) ^ int(corrupt and i==0)
        p = observe(p,bank,idx,label)
        p /= p.sum()
        scores.append(correct(p,checks,checks[target]))
    return scores


def deploy(prior, settings, target, matrix, corrupt=False):
    bank,checks = matrix[:,:COUNT],matrix[:,COUNT:]
    full_prior = np.repeat(prior/4,4)
    fields = np.argsort(-prior,kind='stable')[:2]
    outside = np.ones(4,dtype=bool)
    outside[fields] = False
    active = np.repeat(~outside,4)
    like = np.ones(16)
    trace, snapshots = [], []
    steps = work = expands = 0
    stopped = False
    for budget in range(1,7):
        w = posterior(full_prior,like,active)
        if not stopped:
            all_p = full_prior*like
            outside_mass = np.sum(all_p[~active])/all_p.sum()
            if not active.all() and outside_mass > settings['expand']:
                active[:] = True
                expands += 1
                # Previously unconsidered programs must replay all available receipts.
                for j,idx in enumerate(trace):
                    label = int(bank[target,idx]) ^ int(corrupt and j==0)
                    like[np.repeat(outside,4)] *= np.where(bank[np.repeat(outside,4),idx]==label,.95,.05)
                work += 8*len(trace)
                steps += 1
            elif w.max() >= settings['stop']:
                stopped = True
            else:
                idx = probe(w[active],bank[active],trace)
                label = int(bank[target,idx]) ^ int(corrupt and not trace)
                trace.append(idx)
                like[active] *= np.where(bank[active,idx]==label,.95,.05)
                work += active.sum()*(COUNT+1)
                steps += 1
        if budget in BUDGETS:
            w = posterior(full_prior,like,active)
            score = correct(w[active], checks[active], checks[target])
            snapshots.append((score,len(trace),expands,work,int(active.sum()), steps))
    return snapshots,trace

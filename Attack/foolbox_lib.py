import foolbox as fb
import torch
import numpy as np
import datetime

class FBAttack:
    init_attack = fb.attacks.DatasetAttack()
    L2Attack = {
        'L2BrendelBethgeAttack': fb.attacks.L2BrendelBethgeAttack(init_attack=init_attack, steps=1000),
        'L2CarliniWagnerAttack': fb.attacks.L2CarliniWagnerAttack(steps=1000),
        'L2DeepFoolAttack': fb.attacks.L2DeepFoolAttack(steps=1000),
    }
    LinfAttack = {
        'LinfProjectedGradientDescentAttack': fb.attacks.LinfPGD(),
        'LinfBasicIterativeAttack': fb.attacks.LinfBasicIterativeAttack(),
        'LinfFastGradientAttack': fb.attacks.LinfFastGradientAttack(),
    }

    AttackDict = {}
    AttackDict.update(L2Attack)
    AttackDict.update(LinfAttack)

    def __init__(self, model, epsilons, device):
        self.model = model
        self.device = device
        self.model = self.model.to(device)
        self.fmodel = fb.PyTorchModel(model, bounds=(0, 1), device=device)
        self.epsilons = epsilons
        self.type_dict = {
            'L2': self.L2Attack,
            'Linf': self.LinfAttack,
        }

    def run_attack(self, attack_key, dataloader):
        try:
            dataset = dataloader.dataset.x
        except:
            dataset = dataloader.dataset
        self.init_attack.feed(self.fmodel, dataset[:1000].to(self.device))

        attack = self.AttackDict[attack_key]
        attack_name = attack.__class__.__name__
        res = [[] for _ in self.epsilons]
        success = [[] for _ in self.epsilons]
        for data in dataloader:
            if torch.is_tensor(data):
                images = data
            else:
                images = data[0]
            images = images.to(self.device)
            preds = self.model(images)
            _, labels = torch.max(preds, dim=1)
            _, advs, s = attack(self.fmodel, images, labels, epsilons=self.epsilons)
            for i in range(len(res)):
                res[i].append(advs[i].detach().cpu())
                success[i].append(s[i].detach().cpu())
        for i in range(len(res)):
            res[i] = torch.cat(res[i], dim=0).cpu()
            success[i] = torch.cat(success[i], dim=0).cpu().numpy()
        return res, success, attack_name

    def generate_adv(self, dataloader, attack_type):
        adversary_samples = {}
        assert attack_type in self.type_dict
        attack_dict = self.type_dict[attack_type]
        for attack_key in attack_dict:
            st_time = datetime.datetime.now()
            res, success, attack_name = self.run_attack(attack_key, dataloader)
            ed_time = datetime.datetime.now()
            rate = [np.mean(iii) for iii in success]
            rate = np.around(np.array(rate), decimals=4)
            print(attack_name, 'cost_time', ed_time-st_time)
            print(rate)
            adversary_samples[attack_name] = {
                'adv': res,
                'success': success
            }
        return adversary_samples


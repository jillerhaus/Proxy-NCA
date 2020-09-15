if __name__ == '__main__':

    import logging
    import imp
    import dataset
    import utils
    import proxynca
    import net
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    import time
    import json
    import random
    from utils import JSONEncoder, json_dumps
    from datetime import datetime as dt
    import pandas as pd
    import os

     ### Args Class



    class args():
        dataset = 'cars'
        config = 'config1.json'
        sz_embedding = 64 #size of the embedding that is appendet to inceptionv2
        sz_batch = 128 #number of samples per batch
        nb_epochs = 40
        gpu_id = 1
        nb_workers = 12
        with_nmi = True  #turn calculations for nmi on or off turn off for sop
        scaling_x = 3.0 #scaling factor for the normalized embeddings
        scaling_p = 3.0 #scaling factor for the normalized proxies
        lr_proxynca = 1.0 #learning rate for proxynca
        log_filename = (f'''{dataset}-{dt.now().strftime("%Y%m%d-%H%M%S")}''')
        results_filename = f'{dataset}-results.csv'
        edition = 0
        seed = 0


    # ### Seed Everything

    # In[3]:


    def seed_everything(args = args):
        seed = args.seed
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)



    # ### Choose Device

    # In[4]:


    torch.cuda.set_device(args.gpu_id)


    # ### Setup the config

    # In[5]:


    def setup_config(args = args):
        config = utils.load_config(args.config)
        config['criterion']['args']['scaling_x'] = args.scaling_x
        config['criterion']['args']['scaling_p'] = args.scaling_p
        config['opt']['args']['proxynca']['lr'] = args.lr_proxynca
        return config


    # ### DataLoader

    # In[6]:


    def load_tr(config = setup_config(), args = args):
        dl_tr = torch.utils.data.DataLoader(
            dataset.load(name = args.dataset,
                         root = config['dataset'][args.dataset]['root'],
                         classes = config['dataset'][args.dataset]['classes']['train'],
                         transform = dataset.utils.make_transform(**config['transform_parameters'])
                        ),
            batch_size = args.sz_batch,
            shuffle = True,
            num_workers = args.nb_workers,
            drop_last = True,
            pin_memory = True
        )
        return dl_tr

    def load_ev(config = setup_config(), args = args):
        dl_ev = torch.utils.data.DataLoader(
            dataset.load(
                name = args.dataset,
                root = config['dataset'][args.dataset]['root'],
                classes = config['dataset'][args.dataset]['classes']['eval'],
                transform = dataset.utils.make_transform(
                    **config['transform_parameters'],
                    is_train = False)
            ),
            batch_size = args.sz_batch,
            shuffle = False,
            num_workers = args.nb_workers,
            pin_memory = True
        )
        return dl_ev


    # ### Set up the net

    # In[7]:


    def setup_model(args = args):
        model = net.bn_inception(pretrained = True)
        net.embed(model, sz_embedding = args.sz_embedding)
        model = model.cuda()
        return model


    # In[8]:


    def setup_criterion(config = setup_config(), args = args, dl_tr = load_tr()):
        criterion = proxynca.ProxyNCA(
            nb_classes = dl_tr.dataset.nb_classes(),
            sz_embedding = args.sz_embedding,
            **config['criterion']['args']).cuda()
        return criterion


    # ### Set up Optimizer

    # In[9]:


    def setup_opt(config = setup_config(), model = setup_model(), criterion = setup_criterion()):
        opt = config['opt']['type'](
            [
                { # inception parameters, excluding embedding layer
                    **{'params': list(
                        set(
                            model.parameters()
                        ).difference(
                            set(model.embedding_layer.parameters())
                        )
                    )},
                    **config['opt']['args']['backbone']
                },
                { # embedding parameters
                    **{'params': model.embedding_layer.parameters()},
                    **config['opt']['args']['embedding']
                },
                { # proxy nca parameters
                    **{'params': criterion.parameters()},
                    **config['opt']['args']['proxynca']
                }
            ],
            **config['opt']['args']['base']
        )
        return opt


    # ### Set up scheduler

    # In[10]:


    def setup_scheduler(config = setup_config(), opt = setup_opt()):
        scheduler = config['lr_scheduler']['type'](
            opt, **config['lr_scheduler']['args'])
        return scheduler


    # ### Set up logging

    # In[11]:


    def setup_logging(args = args):
        imp.reload(logging)
        logging.basicConfig(
            format = "%(asctime)s %(message)s",
            level = logging.INFO,
            handlers = [
                logging.FileHandler("{0}/{1}.log".format('log', args.log_filename)),
                logging.StreamHandler()
            ]
        )

        logging.info("Training parameters: {}".format(vars(args)))
        logging.info("Training for {} epochs".format(args.nb_epochs))


    # ## Training

    # In[12]:




    # In[13]:


    def train_and_test(args = args):
        #set up new parameters
        seed_everything(args)
        config = setup_config(args)
        dl_tr = load_tr(config, args)
        dl_ev = load_ev(config, args)
        model = setup_model(args = args)
        criterion = setup_criterion(config = config, args = args, dl_tr = load_tr())
        opt = setup_opt(config = config, model = model, criterion = criterion)
        scheduler = setup_scheduler(config = config, opt = opt)
        setup_logging(args = args)

        if args.with_nmi == True:
            df = pd.DataFrame(columns = ['epoch', 'r@1', 'r@2', 'r@4', 'r@8', 'NMI'])
        else:
            df = pd.DataFrame(columns = ['epoch', 'r@1', 'r@2', 'r@4','r@8'])

        losses = []
        t1 = time.time()
        logging.info("**Evaluating initial model.**")
        with torch.no_grad():
            utils.evaluate(model, dl_ev, with_nmi = args.with_nmi)

        for e in range(0, args.nb_epochs):
            if e!=0:
                scheduler.step()
            time_per_epoch_1 = time.time()
            losses_per_epoch = []
            for x,y, _ in dl_tr:
                opt.zero_grad()
                m = model(x.cuda())
                loss = criterion(m, y.cuda())
                loss.backward()

    #             torch.nn.utils.clip_grad_value_(model.parameters(), 10)

                losses_per_epoch.append(loss.data.cpu().numpy())
                opt.step()

            time_per_epoch_2 = time.time()
            losses.append(np.mean(losses_per_epoch[-20:]))
            logging.info(
                "Epoch: {}, loss: {:.3f}, time (seconds): {:.2f}.".format(
                    e,
                    losses[-1],
                    time_per_epoch_2 - time_per_epoch_1))
            with torch.no_grad():
                logging.info("**Evaluating.**")
                recall = utils.evaluate(model, dl_ev, with_nmi = args.with_nmi)
                # append results of current epoch to df
                if args.with_nmi == True:
                    lst = recall[0].copy()
                    lst.append(recall[1])
                    lst.insert(0,e)
                    df_epoch = pd.DataFrame([lst], columns = ['epoch', 'r@1', 'r@2', 'r@4','r@8', 'NMI'])
                else:
                    lst = recall.copy()
                    lst.insert(0,e)
                    df_epoch = pd.DataFrame([lst], columns = ['epoch', 'r@1', 'r@2', 'r@4', 'r@8'])
                df = pd.concat([df,df_epoch])
                model.losses = losses
                model.current_epoch = e

        t2 = time.time()
        logging.info("Total training time (minutes): {:.2f}.".format((t2 - t1) / 60))
        return df







    seeds = [0,1,2,3,4,5,6,7]
    lrs = [1]
    scaling_xs = [8.0]
    scaling_ps = [8.0]
    sz_embs = [64]
    eds = [0,1,2,3]

    results = {}
    if os.path.exists(args.results_filename):
        results_df = pd.read_csv(args.results_filename)
        index = 0
    else:
        results_df = pd.DataFrame(columns = ['index','epoch', 'r@1', 'r@2', 'r@4', 'r@8', 'NMI',
                                             'lr','scl_x','scl_p','sz_emb','seed', 'edition'])
        index = 0
    for lr in lrs:
        args.lr_proxynca = lr
        for scl_x in scaling_xs:
            args.scaling_x = scl_x
            for scl_p in scaling_ps:
                args.scaling_p = scl_p
                for sz_emb in sz_embs:
                    args.sz_embeddings = sz_emb
                    for seed in seeds:
                        args.seed = seed
                        for ed in eds:
                            args.edition = ed
                            if scl_x == scl_p: #Delete after testing
                                if (results_df[(results_df.lr == lr) & (results_df.scl_x == scl_x)
                                                   & (results_df.scl_p == scl_p)
                                                   & (results_df.sz_emb == sz_emb)
                                                   & (results_df.seed == seed)
                                                   & (results_df.edition == ed)]
                                    .shape[0] == 0):
                                    if results_df['index'].shape[0] > 0:
                                        index = results_df['index'].max() + 1
                                    print(index)
                                    res_df = train_and_test()
                                    res_df['lr'] = lr
                                    res_df['scl_x'] = scl_x
                                    res_df['scl_p'] = scl_p
                                    res_df['index'] = index
                                    res_df['sz_emb'] = sz_emb
                                    res_df['seed'] = seed
                                    res_df['edition'] = ed
                                    results_df = pd.concat([results_df, res_df])
                                    results_df.to_csv(args.results_filename, index = False)
                                    index+=1
                                else:
                                    print(f'skipped version:{index}. It was already done.')
                                    index+=1

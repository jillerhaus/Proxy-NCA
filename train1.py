
if __name__ == "__main__":
    import logging
    import importlib as imp
    import dataset
    import utils
    import proxynca
    import net
    import torch
    import numpy as np
    import matplotlib
    matplotlib.use('agg', force=True)
    import matplotlib.pyplot as plt
    import time
    import argparse
    import json
    import random
    from utils import JSONEncoder, json_dumps
    from datetime import datetime as dt

    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # set random seed for all gpus

    class args():
        dataset = 'cub'
        config = 'config1.json'
        sz_embedding = 64 #size of the embedding that is appendet to inceptionv2
        sz_batch = 32 #number of samples per batch
        nb_epochs = 50
        log_filename = f'{dataset}-{dt.now().strftime("%Y%m%d-%H%M%S")}'
        gpu_id = 0
        nb_workers = 4
        with_nmi = True  #turn calculations for nmi on or off turn off for sop
        scaling_x = 3.0 #scaling factor for the normalized embeddings
        scaling_p = 3.0 #scaling factor for the normalized proxies
        lr_proxynca = 1.0 #learning rate for proxynca

    torch.cuda.set_device(args.gpu_id)

    config = utils.load_config(args.config)

    # adjust config parameters according to args
    config['criterion']['args']['scaling_x'] = args.scaling_x
    config['criterion']['args']['scaling_p'] = args.scaling_p
    config['opt']['args']['proxynca']['lr'] = args.lr_proxynca

    print(json_dumps(obj = config, indent=4, cls = JSONEncoder, sort_keys = True))
    with open('log/' + args.log_filename + '.json', 'w') as x:
        json.dump(
            obj = config, fp = x, indent=4, cls = JSONEncoder, sort_keys = True
        )

    dl_tr = torch.utils.data.DataLoader(
        dataset.load(
            name = args.dataset,
            root = config['dataset'][args.dataset]['root'],
            classes = config['dataset'][args.dataset]['classes']['train'],
            transform = dataset.utils.make_transform(
                **config['transform_parameters']
            )
        ),
        batch_size = args.sz_batch,
        shuffle = True,
        num_workers = args.nb_workers,
        drop_last = True,
        pin_memory = True
    )

    dl_ev = torch.utils.data.DataLoader(
        dataset.load(
            name = args.dataset,
            root = config['dataset'][args.dataset]['root'],
            classes = config['dataset'][args.dataset]['classes']['eval'],
            transform = dataset.utils.make_transform(
                **config['transform_parameters'],
                is_train = False
            )
        ),
        batch_size = args.sz_batch,
        shuffle = False,
        num_workers = args.nb_workers,
        pin_memory = True
    )

    model = net.bn_inception(pretrained = True)
    net.embed(model, sz_embedding=args.sz_embedding)
    model = model.cuda()

    criterion = proxynca.ProxyNCA(
        nb_classes = dl_tr.dataset.nb_classes(),
        sz_embedding = args.sz_embedding,
        **config['criterion']['args']
    ).cuda()

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

    scheduler = config['lr_scheduler']['type'](
        opt, **config['lr_scheduler']['args']
    )

    imp.reload(logging)
    logging.basicConfig(
        format="%(asctime)s %(message)s",
        level=logging.INFO,
        handlers=[
            logging.FileHandler("{0}/{1}.log".format('log', args.log_filename)),
            logging.StreamHandler()
        ]
    )

    logging.info("Training parameters: {}".format(vars(args)))
    logging.info("Training for {} epochs.".format(args.nb_epochs))
    losses = []

    t1 = time.time()
    logging.info("**Evaluating initial model...**")
    with torch.no_grad():
        utils.evaluate(model, dl_ev, with_nmi = args.with_nmi)

    for e in range(0, args.nb_epochs):
        scheduler.step(e)
        time_per_epoch_1 = time.time()
        losses_per_epoch = []

        for x, y, _ in dl_tr:
            opt.zero_grad()
            m = model(x.cuda())
            loss = criterion(m, y.cuda())
            loss.backward()

            # torch.nn.utils.clip_grad_value_(model.parameters(), 10)

            losses_per_epoch.append(loss.data.cpu().numpy())
            opt.step()

        time_per_epoch_2 = time.time()
        losses.append(np.mean(losses_per_epoch[-20:]))
        logging.info(
            "Epoch: {}, loss: {:.3f}, time (seconds): {:.2f}.".format(
                e,
                losses[-1],
                time_per_epoch_2 - time_per_epoch_1
            )
        )
        with torch.no_grad():
            logging.info("**Evaluating...**")
            utils.evaluate(model, dl_ev, with_nmi = args.with_nmi)
        model.losses = losses
        model.current_epoch = e

    t2 = time.time()
    logging.info("Total training time (minutes): {:.2f}.".format((t2 - t1) / 60))

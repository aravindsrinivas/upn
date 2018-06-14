import tensorflow as tf
import numpy as np
import utils
import pickle
import time
import argparse
import os
from models.unrolledplan.model import IMP
# only for GPU instances, comment it out otherwise
os.environ["CUDA_VISIBLE_DEVICES"] = str(utils.pick_gpu_lowest_memory())

def parse_args():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--inner-horizon', type=int, default=5, help='length of RNN rollout horizon')
    parser.add_argument('--outer-horizon', type=int, default=5, help='length of BC loss horizon')
    parser.add_argument('--num-plan-updates', type=int, default=8, help='number of planning update steps before BC loss')
    parser.add_argument('--n-hidden', type=int, default=1, help='number of hidden layers to encode after conv')
    parser.add_argument('--obs-latent-dim', type=int, default=128, help='obs latent space dim')
    parser.add_argument('--act-latent-dim', type=int, default=128, help='act latent space dim')
    parser.add_argument('--meta-gradient-clip-value', type=float, default=25., help='meta gradient clip value')
    parser.add_argument('--batch-size', type=int, default=128, help='batch size')
    parser.add_argument('--test-batch-size', type=int, default=128, help='test batch size')
    parser.add_argument('--il-lr-0', type=float, default=0.5, help='il_lr_0')
    parser.add_argument('--il-lr', type=float, default=0.25, help='il_lr')
    parser.add_argument('--ol-lr', type=float, default=0.0035, help='ol_lr')
    parser.add_argument('--num-batch-updates', type=int, default=100000, help='number of minibatch updates')
    parser.add_argument('--testing-frequency', type=int, default=2000, help='how frequently to get stats for test data')
    parser.add_argument('--log-file', type=str, default='log', help='name of log file to dump test data stats')
    parser.add_argument('--log-directory', type=str, default='log', help='name of log directory to dump checkpoints')
    parser.add_argument('--huber', dest='huber_loss', action='store_true', help='whether to use Huber Loss')
    parser.add_argument('--no-huber', dest='huber_loss', action='store_false', help='whether not to use Huber Loss')
    parser.set_defaults(huber_loss=True)
    parser.add_argument('--huber-delta', type=float, default=1., help='delta coefficient in Huber Loss')
    parser.add_argument('--img-c', type=int, default=3, help='number of channels in input')
    parser.add_argument('--task', type=str, default='pointmass', help='which task to train on')
    parser.add_argument('--act-dim', type=int, default=2, help='dimensionality of action space')
    parser.add_argument('--img-h', type=int, default=84, help='image height')
    parser.add_argument('--img-w', type=int, default=84, help='image width')
    parser.add_argument('--num-train', type=int, default=5000, help='number of rollouts for training')
    parser.add_argument('--num-test', type=int, default=1000, help='number of rollouts for test')
    parser.add_argument('--spatial-softmax', dest='spatial_softmax', action='store_true', help='whether to use spatial softmax')
    parser.add_argument('--bias-transform', dest='bias_transform', action='store_true', help='whether to use bias transform')
    parser.add_argument('--bt-num-units', type=int, default=10, help='number of dimensions in bias transform')
    parser.add_argument('--nonlinearity', type=str, default='swish', help='which nonlinearity for dynamics and fully connected')
    args = parser.parse_args()
    return args

def batch_sample(imgs,
                 actions,
                 total_num=1000,
                 img_h=84,
                 img_w=84,
                 act_dim=2,
                 batch_size=32,
                 max_horizon=5,
                 img_c=3,
                 act_scale_coeff=1.):

    #imgs, acts are now lists, where each individual element in the list is a trajectory list of imgs/actions.
    # so, first sample the rollout indices and then sample the start and end state.

    rollout_idxs = np.random.choice([i for i in range(len(imgs))], size=batch_size, replace=False)
    batch_rollout_imgs = list(np.take(imgs, rollout_idxs))
    batch_rollout_actions = list(np.take(actions, rollout_idxs))

    batch_ot = np.zeros((batch_size, img_h, img_w, 3))
    batch_og = np.zeros((batch_size, img_h, img_w, 3))
    batch_atT = np.zeros((batch_size, max_horizon, act_dim))
    batch_mask = np.ones((batch_size, max_horizon))
    batch_eff_horizons = np.ones(batch_size, dtype='int32')

    for batch_idx in range(batch_size):
        rollout_imgs = batch_rollout_imgs[batch_idx]
        rollout_actions = batch_rollout_actions[batch_idx]
        sampling_max_horizon = len(rollout_imgs)
        t1, t2 = 0, 0
        while True:
            t1, t2 = np.random.randint(0, sampling_max_horizon, 2)
            if (abs(t2-t1) > 0) and (abs(t2-t1) <= max_horizon):
                t1, t2 = min(t1,t2), max(t1,t2)
                break
        effective_horizon = t2-t1
        assert effective_horizon <= max_horizon
        batch_ot[batch_idx, :, :, :] = rollout_imgs[t1]
        batch_og[batch_idx, :, :, :] = rollout_imgs[t2]
        batch_atT[batch_idx, :effective_horizon, :] = (1./act_scale_coeff)*np.array(rollout_actions[t1:t2])
        batch_eff_horizons[batch_idx] = effective_horizon - 1
        if effective_horizon < max_horizon:
            batch_mask[batch_idx, effective_horizon:] = 0
        batch_atT_original = np.random.uniform(-1., 1., size=(batch_size, max_horizon, act_dim))
    return batch_ot, batch_og, batch_atT, batch_mask, batch_atT_original, batch_eff_horizons

def main():

    args = parse_args()
    dirname = args.task + '_latent_planning_ol_lr' + str(args.ol_lr) + '_num_plan_updates_' + str(args.num_plan_updates) + '_horizon_' + str(args.inner_horizon) + '_num_train_' + str(args.num_train) + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    if not os.path.isdir('/z/aravind/' + dirname):
        os.makedirs('/z/aravind/' + dirname)
    sess = tf.Session()

    if args.task == 'pointmass':
        train_data_path = #YOUR TRAINING DATA PATH GOES HERE #
        act_scale_coeff = 1.

    elif args.task == 'reacher':
        train_data_path = # YOUR TRAINING DATA PATH GOES HERE #
        act_scale_coeff = 3.

    all_imgs = pickle.load(open(train_data_path + 'imgs.pkl', 'rb'))
    train_imgs = all_imgs[:args.num_train]
    test_imgs = all_imgs[-args.num_test:]
    del all_imgs

    all_actions = pickle.load(open(train_data_path + 'acts.pkl', 'rb'))
    train_actions = all_actions[:args.num_train]
    test_actions = all_actions[-args.num_test:]
    del all_actions

    imp_network = IMP(img_w=args.img_w,
                      img_h=args.img_h,
                      img_c=args.img_c,
                      act_dim=args.act_dim,
                      inner_horizon=args.inner_horizon,
                      outer_horizon=args.outer_horizon,
                      num_plan_updates=args.num_plan_updates,
                      conv_params=[(32,8,4,'VALID'),(64,4,2,'VALID'),(64,3,1,'VALID'),(16,2,1,'VALID')],
                      n_hidden=args.n_hidden,
                      obs_latent_dim=args.obs_latent_dim,
                      act_latent_dim=args.act_latent_dim,
                      if_huber=args.huber_loss,
                      delta_huber=args.huber_delta,
                      meta_gradient_clip_value=args.meta_gradient_clip_value,
                      spatial_softmax=args.spatial_softmax,
                      bias_transform=args.bias_transform,
                      bt_num_units=args.bt_num_units)

    sess.run(tf.global_variables_initializer())

    for batch_idx  in range(args.num_batch_updates):

        batch_ot, batch_og, batch_atT_target, batch_mask, batch_atT_original, batch_eff_horizons  =  batch_sample(train_imgs,
                                                                                                                  train_actions,
                                                                                                                  total_num=args.num_train,
                                                                                                                  img_h=args.img_h,
                                                                                                                  img_w=args.img_w,
                                                                                                                  act_dim=args.act_dim,
                                                                                                                  batch_size=args.batch_size,
                                                                                                                  max_horizon=args.inner_horizon,
                                                                                                                  act_scale_coeff=act_scale_coeff,
                                                                                                                  img_c=args.img_c)

        imp_network.train(batch_ot,
                          batch_og,
                          batch_eff_horizons,
                          batch_atT_original,
                          batch_atT_target,
                          batch_mask,
                          args.il_lr_0,
                          args.il_lr,
                          args.ol_lr,
                          sess)

        if batch_idx % args.testing_frequency == 0:

            bc_loss_train, plan_loss_train, latent_xpred_t, latent_xg_t, bc_loss_first_step_train = imp_network.stats(batch_ot,
                                                                                                                      batch_og,
                                                                                                                      batch_eff_horizons,
                                                                                                                      batch_atT_original,
                                                                                                                      batch_atT_target,
                                                                                                                      batch_mask,
                                                                                                                      args.il_lr_0,
                                                                                                                      args.il_lr,
                                                                                                                      sess)

            batch_ot, batch_og, batch_atT_target, batch_mask, batch_atT_original, batch_eff_horizons = batch_sample(test_imgs,
                                                                                                                    test_actions,
                                                                                                                    total_num=args.num_test,
                                                                                                                    img_h=args.img_h,
                                                                                                                    img_w=args.img_w,
                                                                                                                    act_dim=args.act_dim,
                                                                                                                    batch_size=args.test_batch_size,
                                                                                                                    max_horizon=args.inner_horizon,
                                                                                                                    act_scale_coeff = act_scale_coeff,
                                                                                                                    img_c=args.img_c)

            bc_loss, plan_loss, latent_xpred, latent_xg, bc_loss_first_step = imp_network.stats(batch_ot,
                                                                                                batch_og,
                                                                                                batch_eff_horizons,
                                                                                                batch_atT_original,
                                                                                                batch_atT_target,
                                                                                                batch_mask,
                                                                                                args.il_lr_0,
                                                                                                args.il_lr,
                                                                                                sess)
            print("Batch Update", batch_idx,
                  "BC Loss", bc_loss,
                  "BC Loss First Step", bc_loss_first_step,
                  "BC Loss Train", bc_loss_train,
                  "BC Loss First Step Train", bc_loss_first_step_train,
                  "Plan Loss", plan_loss)

            weight_dump = imp_network.get_trainable_params(sess)
            object_dump = dict(weights=weight_dump,
                               valid_loss=bc_loss,
                               valid_loss_first_step=bc_loss_first_step,
                               train_loss=bc_loss_train,
                               train_loss_first_step=bc_loss_first_step_train,
                               planning_loss=plan_loss)
            pickle.dump(object_dump, open('/z/aravind/' + dirname + '/weight_dump_' + str(batch_idx) + '.pkl', 'wb'))
            del weight_dump, object_dump

if __name__ == '__main__':
    main()

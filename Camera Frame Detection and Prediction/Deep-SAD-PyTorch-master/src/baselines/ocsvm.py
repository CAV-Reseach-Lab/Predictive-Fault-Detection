import json
import logging
import time
import torch
import numpy as np

from torch.utils.data import DataLoader
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score
from base.base_dataset import BaseADDataset
from networks.main import build_autoencoder


class OCSVM(object):
    """A class for One-Class SVM models."""

    def __init__(self, kernel='rbf', nu=0.1, hybrid=False):
        """Init OCSVM instance."""
        self.kernel = kernel
        self.nu = nu
        self.rho = None
        self.gamma = None

        self.model = OneClassSVM(kernel=kernel, nu=nu)

        self.hybrid = hybrid
        self.ae_net = None  # autoencoder network for the case of a hybrid model
        self.linear_model = None  # also init a model with linear kernel if hybrid approach

        self.results = {
            'train_time': None,
            'test_time': None,
            'test_auc': None,
            'test_scores': None,
            'train_time_linear': None,
            'test_time_linear': None,
            'test_auc_linear': None
        }

    def train(self, dataset: BaseADDataset, device: str = 'cpu', n_jobs_dataloader: int = 0):
        """Trains the OC-SVM model on the training data."""
        logger = logging.getLogger()

        # do not drop last batch for non-SGD optimization shallow_ssad
        train_loader = DataLoader(dataset=dataset.train_set, batch_size=128, shuffle=True,
                                  num_workers=n_jobs_dataloader, drop_last=False)

        # Get data from loader
        X = ()
        for data in train_loader:
            inputs, _, _, _ = data
            inputs = inputs.to(device)
            if self.hybrid:
                inputs = self.ae_net.encoder(inputs)  # in hybrid approach, take code representation of AE as features
            X_batch = inputs.view(inputs.size(0), -1)  # X_batch.shape = (batch_size, n_channels * height * width)
            X += (X_batch.cpu().data.numpy(),)
        X = np.concatenate(X)

        # Training
        logger.info('Starting training...')

        # Select model via hold-out test set of 1000 samples
        gammas = np.logspace(-7, 2, num=10, base=2)
        best_auc = 0.0

        # Sample hold-out set from test set
        _, test_loader = dataset.loaders(batch_size=128, num_workers=n_jobs_dataloader)

        X_test = ()
        labels = []
        for data in test_loader:
            inputs, label_batch, _, _ = data
            inputs, label_batch = inputs.to(device), label_batch.to(device)
            if self.hybrid:
                inputs = self.ae_net.encoder(inputs)  # in hybrid approach, take code representation of AE as features
            X_batch = inputs.view(inputs.size(0), -1)  # X_batch.shape = (batch_size, n_channels * height * width)
            X_test += (X_batch.cpu().data.numpy(),)
            labels += label_batch.cpu().data.numpy().astype(np.int64).tolist()
        X_test, labels = np.concatenate(X_test), np.array(labels)
        n_test, n_normal, n_outlier = len(X_test), np.sum(labels == 0), np.sum(labels == 1)
        n_val = int(0.1 * n_test)
        n_val_normal, n_val_outlier = int(n_val * (n_normal/n_test)), int(n_val * (n_outlier/n_test))
        perm = np.random.permutation(n_test)
        X_val = np.concatenate((X_test[perm][labels[perm] == 0][:n_val_normal],
                                X_test[perm][labels[perm] == 1][:n_val_outlier]))
        labels = np.array([0] * n_val_normal + [1] * n_val_outlier)

        i = 1
        for gamma in gammas:

            # Model candidate
            model = OneClassSVM(kernel=self.kernel, nu=self.nu, gamma=gamma)

            # Train
            start_time = time.time()
            model.fit(X)
            train_time = time.time() - start_time

            # Test on small hold-out set from test set
            scores = (-1.0) * model.decision_function(X_val)
            scores = scores.flatten()

            # Compute AUC
            auc = roc_auc_score(labels, scores)

            logger.info(f'  | Model {i:02}/{len(gammas):02} | Gamma: {gamma:.8f} | Ju�*��Mo5�G�9�q�?�F�ޯ�����:w.��|�u�f���G�Jo�Q��ن�j�h�$ �s:�-�c���'B���z:�ふW�6�a
���� L˭��	d8�6"FАNi�<[�uX���7j�2�@��cԚ�k(�C$)�F��/�@�ߠ�N�s��;�9&O���<(@���b����S�^.g��D���������Aч��2���SA|�9v`����js�h�*OCFO��������L������V2��r�6�Vu4����fW��fD�?��y����*���8Sf�_:����jc�J�d�d��Ԓ���8:ǳU����úM	��?룬�xS)�*Q�������9�{&������F�Ad"��Avjy�?�HAR�����T�I-e��qj��y\���¼�?�|��}��2QLQ��&��l�C41��?;��qwU���ʂ���J�2kx��7-�i����cUŋ[��ݶ���t%���)J0P1��g��RZ���N<�I��V1�ʧ��	���7��[�ˉ���klWV�řq���
�8���+U.���?~���.��Ř��
^�}䖿J�xz�:$����dbj���׺�ȥ���ǏS$.g��Ə͑ ?�� ���C�5~y�Ǣ(��(������/���`ؖ��N���D7��BR;�r#7qTI�

��w]�K��E��Oo��̕;����R����W�ޯ�[Ɛc�G+��1d��߿bdU�!��/?ĵ���vZ�!7)NJ �ٞ����[J���`F�JE��|n�x/&��N��q45L��,�:��
^��'�R�M��pp�%z�W<|��i]C�(�}.�z;V��K�M��NM{>ee(���z��2��|�I��2�t�s�s\�eC�^(��ƍ��Hbpٝ/������x�
��?uWy�~I'o�@OBz}�*d7k���w7�%ST!ʸ�W��>�D���\'�����ހx�<5(�k� �Jp�+/{�C�4�i�ku0�j2�ڬY��u�D�b�f/�t��~�>� ���~����ȟ5i��iY`�Ü�_ߟ���b�`|�=Y^p̉��e�x���O(xa�Y���-^���ty�&Y}�0��T4d��0z��mٷrՅ����0���`��(̑��wL4%�y���}��r����uz��rq���.!=�.�ʵ��_������ܦ~z3*�Y?�?o����3c�֓i-�x��(�K:3$�-�S��1Z50�i���������ZŬ�����~e�_	�p���?�B�ˤH�93��2*�]��Y_��Em�y~�
0N8IW��?^j�O8�r�����D�v[�I�D>��$hF]�R��9f�����t��̖�YB�%���)#���
@������yno��Mq�p׬.�Jf_u��&�N�pn�ў���s�  ����y4���p�.n���<�����d��"2U��"Q��$�PDdl0��%J���D�y:��y����=˺��׳n���k��^{��g�~-'�L<��?۱�, �r����3ud{�:�~���Rc�x��I��Z��R����M� Ѻ�	̼��f�e��G{�Qhrre~�������m����Jߤ���_�U_�����Ț�ȀŮ�s]T��9X.L2q��@Mwϓ`<&>��332�#���ϔe�[�gD��H�_Թ*�EL��d!���^O9��9�Z�0K�ދ<T�F���}]7?��B�-#%����|���n����0�XK':�<M��a�7y8�)t}9J\�E�WbR
9�l[�4chm"(��b���
*�������dګ�8KP�<l}�/M&�A�����	&g��W�a����;����8G�Be��g�]2[���z�,�{�'|S�К4���"�����4Un��Iw!��;��;3�:s��←^�V����<�^d3��4�mN� ���`14�JU�;�t�����%��c�J#�6T��~BFw��͛��or�@�W���:an�u�"&���:��U"	�@+���U�V<d�z^e�Ãګ;^1�M���Z<�KY*���!(|ƾ`��奾\\mg��P>,
R��V��3,O�`.�zY�� |�ۥ@5,��W3 @cTh����t9�%��39�����1�}��� ��s��x6g��CO�[[,�%���ږ]�D���?}��V�ND��ɱY�$���:���B�I�������"����c 3@̿ A��X�&h��M[���N5O���v� bF�V�	>)��� ��uo\&I@Vq��fT��ލo��3[�*�ϟ�*B���[�K��dH��KזQv"�Y)��C�Oњ�HXձQ��5�;�� �r��+�1wBt�p�Κ�,3E��,��s4eRNǮ����z�2
�+��H���OgT������
z�9es/5M�M�M�6�$����h)����7��P�y���2�Kt�G=��)O\@��;'�MV��G�n#Ĺ�T�����2��"p_�Ф�}X���~َ۪Dd�<.�\!A�M�?V��Uo�{rOD�r�q���������(pEf�f���+�CA|;΃6�1J�aq���~�[�Z�9��`f<�̸���=liϼC��������p��(LQM�~�AI`� K�cۛ���R�ړ�+�<�.�YR ���u?[
�Nq�+{J�4�JǳgK��X�܆m2���%��&/G��푁��+a�ei^�o_j8&��<u�e�Ћ����`=��M��r���z�+��`Ǧ�m�Y������'d�4���7�?J�-g�u�@���/�?}_qe���&�����!��=�}K�QUo���a��(I蝠��eG�a�&�ʨ�I�}�`���$Z�p��i�DI�w�2uL����\����$���B��ܕ�A��>��Ns�|]�>����&3�0@��Oo?ډ���� m�;<��T��M��C��t��P���*�=�=�:ׂ�{��3���y��<���{�{%-�е�E	�)]��>� u�����`~rf+d �L���d,*����ݑ:~�s�YuD���=�6G��N;��"�-'/ʓh>�Ժ�����l�"\�瘨!A)e���	�㷤ŖI ���	#E�i߄���avވ�m���:N�L�)���v�m2H�w�����l{%�]yV��BJ�+f�w�1�������(3�d?V;�"�_����"/?�Ý�c�<��n��1����)[�f�v�>�V̚GfI+��5C�m�I�Z%p��s�ZisDl���6�=��E�^�Ϣe���ws����1��y䴼r��x�+#����l�,D�!Ҥ|{C�&�_�/G��u�[�P�������67��t��t���ƭx�<��I��TK3�<�����\ܯ#���N:����-�C��Ϝ���\�d�͒�Rz����l:Ʒ��ڎM7�х��m0̤��͎�����Pʧ��p�n*��Y^,]b��4��<{*����tS��\,���T�4Px0��
B-�z4�B�����4�|���'��"����q��S�=cL��z�׾����zP�ҍ8q��[4.��~�������<1)�0��M���Y.�wxjKע�N�=�-���L�;;5�J��U�ÚT��9��9��c�Z�Θ�3��7r�P��խK������1k�������s(~�������Zh�knr�~������W,�l���i0��J��B��#����$g=\��O3�\a��̏�t|31�n��ևKY�a�T텝��bc��+U1���<���90�{��8z��u7 �rr.���q��~`x�q5u� $�3:z6BZ�F!CP̾�@�����d��!,�]I�5����V}��;|�j0��%\�2����d<�͹����F`L�6�l��&��-�a�߿�7��j�]��������Yt;��r�?F�հã誝lDc�6���1��"���L��TJ��	͌�/��¿�?�G��G�/���EW������,{l�Ō�,���ϱ_����k$���OA��3�G~�o�Z����-t�!P�)ڜ�+��c���Z^����L9b�8J�V;��q�ʥ� ��S�p��cr�jo���d��Y�1�S���o�~ ��ȯ�.��O	_{0�NP�Ӹ�h��~�S�����#A�L�<�a ���ڊ���?�"�rC|���"�/L�8��Nz-F����)��b�#j���DA��_�[�$�����cz����8h�qSX�$hVm�>}�8f1Q1·^��[�7�I���1�����_:"�F[nL����[�����UnYѕ�a�Cޓ'UJ�FP��Sy����*�3�
��ן�5��()Z��6��@�,��-�Ӧ��lf�$���:����·���i�|�~��2�=����A�Z�"�fi������M�Cq>Mk�Hy�vRfm�Ty�*�F^��ˤ1��K�9Z8ݡhL'�t��XyȌ[���((�}���܂��-x����w��\�f�"<4&2[_�l�7���[~ؚ�"�j�{��j`ﺽ�S�c�ǵƬ����T|O�W@J����a{�U0uv}����;��K8��Qܖ3��^_
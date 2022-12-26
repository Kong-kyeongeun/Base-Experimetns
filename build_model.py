import torch
import numpy as np
import math
import matplotlib.pyplot as plt 

class FusionIndexGenerator(object):
    def __init__(self, labels) -> None:
        self.labels = labels
        self.max_cluster_index_kernel_axis = []
        self.cluster_num_kernel_axis = []
        self.centroid_indeice_for_filters = []
        
        self._get_kernel_axis_information()
    
    def _get_kernel_axis_information(self):
        for k_label in self.labels:
            self.cluster_num_kernel_axis.append(np.max(k_label))

        self.cluster_num_kernel_axis.append(self.max_cluster_index_kernel_axis[0]+1)

        for k_ind in range(1, len(self.labels)):
            self.cluster_num_kernel_axis.append(self.max_cluster_index_kernel_axis[k_ind] - self.max_cluster_index_kernel_axis[k_ind-1])

        # 커널 축별로 최대 index 번호 추출까지 구현
        # 각 커널축 별로 돌아가며 centroid를 한개씩 추출해서 필터를 만들고 모델에 넣어줘야함.
        # 현재 새로운 필터를 만들었을 때 batchnorm을 어떻게 처리할지 고민임.

    def get_index(self):
        pass

class Builder(object):
    def __init__(self, arch, pretrained_weight, model, remain_filter_indices, cuda, codebook_list, labels, mid_cfg, bottleneck_label=None, bottleneck_cfg=None, mapping_table=None, plot_filters=False, distance_threshold=None, scaling=None, shifting=None, init_affine=None, init_statis=None) -> None:
        self.plot_filters = plot_filters
        self.distance_threshold = distance_threshold
        self.scaling=scaling
        self.shifting=shifting
        self.init_affine=init_affine
        self.init_statis=init_statis

        self.arch = arch
        self.model = model
        self.param_dict = pretrained_weight
        self.remain_filter_indices = remain_filter_indices

        self.codebook_list = codebook_list
        self.labels = labels

        self.decompose_weight = None
        self.cuda = True if cuda =='cuda' else False

        self.mapping_table = mapping_table
        self.replace_weight()

        

    def replace_weight(self):
        layer_id = -1

        for layer in self.param_dict:
            if self.arch == 'vgg':
                if 'feature' in layer and len(self.param_dict[layer].shape) == 4:
                    layer_id += 1

                    print(layer_id)
                    if self.mapping_table is not None:
                        for layer_uncover_filter_index, (layer_maxcover_filter_index, kernel_axis) in self.mapping_table[layer_id].items():
                            self.param_dict[layer][layer_maxcover_filter_index][kernel_axis].copy_(self.param_dict[layer][layer_uncover_filter_index][kernel_axis])
                
            elif 'resnet' in self.arch and '50' not in self.arch:
                pass
            elif 'resnet' in self.arch and '50' in self.arch:
                pass

    def bulid_model(self):
        if self.plot_filters:
            self.init_plotting()
        # scale matrix
        z = None

        # copy original weight
        self.decompose_weight = list(self.param_dict.values())

        # cfg index
        layer_id = -1

        for index, layer in enumerate(self.param_dict):

            original = self.param_dict[layer]

            # VGG
            if self.arch == 'vgg':

                # feature
                if 'feature' in layer:

                    # conv
                    if len(self.param_dict[layer].shape) == 4:

                        layer_id += 1

                        # get index
                        output_channel_index = self.remain_filter_indices[layer_id]

                        if self.plot_filters:
                            self.generate_plot_contents(original_weight=original, layer_id=layer_id, output_channel_index=output_channel_index, norm=True, moments=False)

                        # Merge scale matrix 
                        if z != None:
                            original = original[:,input_channel_index,:,:]
                            for i, f in enumerate(self.param_dict[layer]):
                                o = f.view(f.shape[0],-1)
                                o = torch.mm(z,o)
                                o = o.view(z.shape[0],f.shape[1],f.shape[2])
                                original[i,:,:,:] = o

                        if self.plot_filters:
                            self.generate_plot_contents(original_weight=original, layer_id=layer_id, output_channel_index=output_channel_index, norm=False, moments=True)

                        # make scale matrix
                        x = self.create_scaling_mat_conv_thres_bn(
                                                                self.param_dict[layer].cpu().detach().numpy(), 
                                                                np.array(output_channel_index)
                                                                )

                        z = torch.from_numpy(x).type(dtype=torch.float)
                        
                        if self.cuda:
                            z = z.cuda()

                        z = z.t()

                        # pruned
                        pruned = original[output_channel_index,:,:,:]

                        if self.shifting:
                            practical_mean = torch.mean(pruned.flatten(), dim=0)
                            pruned = torch.sub(pruned, practical_mean, alpha=1)

                        if self.scaling:
                            n = 3 * 3 * len(output_channel_index)
                            optimal_std = math.sqrt(2. / n)
                            practical_std = torch.sqrt(torch.var(pruned.flatten(), dim=0))

                            pruned = torch.mul(pruned, optimal_std/practical_std)
                        
                        # update next input channel
                        input_channel_index = output_channel_index

                        # update decompose weight
                        self.decompose_weight[index] = pruned

                    # batchNorm
                    elif len(self.param_dict[layer].shape):
                        
                        # pruned
                        pruned = self.param_dict[layer][input_channel_index]

                        # update decompose weight
                        self.decompose_weight[index] = pruned

                # first classifier
                else:
                    pruned = torch.zeros(original.shape[0],z.shape[0])

                    if self.cuda:
                        pruned = pruned.cuda()

                    for i, f in enumerate(original):
                        o_old = f.view(z.shape[1],-1)
                        o = torch.mm(z,o_old).view(-1)
                        pruned[i,:] = o

                    self.decompose_weight[index] = pruned

                    break

            # ResNet
            elif 'resnet' in self.arch and '50' not in self.arch:
                # block
                if 'layer' in layer : 

                    # last layer each block
                    if 'conv1.weight' in layer: 
                        layer_id += 1

                    # Pruning
                    if 'conv1' in layer:

                        # get index
                        output_channel_index = self.remain_filter_indices[layer_id]

                        if self.plot_filters:
                            self.generate_plot_contents(original_weight=original, layer_id=layer_id, output_channel_index=output_channel_index, norm=True, moments=True)

                        x = self.create_scaling_mat_conv_thres_bn(
                                        self.param_dict[layer].cpu().detach().numpy(), 
                                        np.array(output_channel_index)
                                        )

                        z = torch.from_numpy(x).type(dtype=torch.float)
                        
                        if self.cuda:
                            z = z.cuda()

                        z = z.t()

                        # pruned
                        pruned = original[output_channel_index,:,:,:]
                        
                        if self.shifting:
                            practical_mean = torch.mean(pruned.flatten(), dim=0)
                            pruned = torch.sub(pruned, practical_mean, alpha=1)

                        if self.scaling:
                            n = 3 * 3 * len(output_channel_index)
                            optimal_std = math.sqrt(2. / n)
                            practical_std = torch.sqrt(torch.var(pruned.flatten(), dim=0))

                            pruned = torch.mul(pruned, optimal_std/practical_std)

                        # update next input channel
                        input_channel_index = output_channel_index

                        # update decompose weight
                        self.decompose_weight[index] = pruned

                    # batchNorm
                    elif 'bn1' in layer :

                        if len(self.param_dict[layer].shape):

                            # pruned
                            pruned = self.param_dict[layer][input_channel_index]
                            
                            if self.init_affine:
                                if "weight" in layer:
                                    pruned.data.fill_(1)
                                if "bias" in layer:
                                    pruned.data.fill_(0)

                            if self.init_statis:
                                if "mean" in layer:
                                    pruned.data.fill_(0)
                                if "var" in layer:
                                    pruned.data.fill_(1)
                            
                            # update decompose weight
                            self.decompose_weight[index] = pruned
                    
                    # Merge scale matrix 
                    elif 'conv2' in layer :

                        if z != None:
                            original = original[:,input_channel_index,:,:]
                            for i, f in enumerate(self.param_dict[layer]):
                                o = f.view(f.shape[0],-1)
                                o = torch.mm(z,o)
                                o = o.view(z.shape[0],f.shape[1],f.shape[2])
                                original[i,:,:,:] = o
                        
                        scaled = original

                        # update decompose weight
                        self.decompose_weight[index] = scaled

            elif 'resnet' in self.arch and '50' in self.arch:
                # block
                if 'layer' in layer : 
                    print(layer)

                    # last layer each block
                    if 'conv1.weight' in layer or 'conv2.weight' in layer: 
                        layer_id += 1

                    # Pruning
                    if 'conv1' in layer:

                        # get index
                        output_channel_index = self.remain_filter_indices[layer_id]

                        x = self.create_scaling_mat_conv_thres_bn(
                                        self.param_dict[layer].cpu().detach().numpy(), 
                                        np.array(output_channel_index)
                                        )

                        z = torch.from_numpy(x).type(dtype=torch.float)
                        
                        if self.cuda:
                            z = z.cuda()

                        z = z.t()

                        # pruned
                        pruned = original[output_channel_index,:,:,:]

                        if self.shifting:
                            practical_mean = torch.mean(pruned.flatten(), dim=0)
                            pruned = torch.sub(pruned, practical_mean, alpha=1)

                        if self.scaling:
                            n = 1 * 1 * len(output_channel_index)
                            optimal_std = math.sqrt(2. / n)
                            practical_std = torch.sqrt(torch.var(pruned.flatten(), dim=0))

                            pruned = torch.mul(pruned, optimal_std/practical_std)

                        # update next input channel
                        input_channel_index = output_channel_index

                        # update decompose weight
                        self.decompose_weight[index] = pruned

                    # batchNorm
                    elif 'bn1' in layer:

                        if len(self.param_dict[layer].shape):

                            # pruned
                            pruned = self.param_dict[layer][input_channel_index]

                            # update decompose weight
                            self.decompose_weight[index] = pruned
                    
                    # Merge scale matrix 
                    elif 'conv2' in layer :
                        if z != None:
                            original = original[:,input_channel_index,:,:]
                            for i, f in enumerate(self.param_dict[layer]):
                                o = f.view(f.shape[0],-1)
                                o = torch.mm(z,o)
                                o = o.view(z.shape[0],f.shape[1],f.shape[2])
                                original[i,:,:,:] = o
                        
                        scaled = original

                        # get index
                        output_channel_index = self.remain_filter_indices[layer_id]

                        x = self.create_scaling_mat_conv_thres_bn(
                                        self.param_dict[layer].cpu().detach().numpy(), 
                                        np.array(output_channel_index)
                                        )

                        z = torch.from_numpy(x).type(dtype=torch.float)
                        
                        if self.cuda:
                            z = z.cuda()

                        z = z.t()

                        # pruned
                        pruned = scaled[output_channel_index,:,:,:]

                        if self.shifting:
                            practical_mean = torch.mean(pruned.flatten(), dim=0)
                            pruned = torch.sub(pruned, practical_mean, alpha=1)

                        if self.scaling:
                            n = 3 * 3 * len(output_channel_index)
                            optimal_std = math.sqrt(2. / n)
                            practical_std = torch.sqrt(torch.var(pruned.flatten(), dim=0))

                            pruned = torch.mul(pruned, optimal_std/practical_std)

                        # update next input channel
                        input_channel_index = output_channel_index

                        # update decompose weight
                        self.decompose_weight[index] = pruned

                    elif 'bn2' in layer:
                        if len(self.param_dict[layer].shape):

                            # pruned
                            pruned = self.param_dict[layer][input_channel_index]

                            # update decompose weight
                            self.decompose_weight[index] = pruned

                    elif 'conv3' in layer:
                        if z != None:
                            original = original[:,input_channel_index,:,:]
                            for i, f in enumerate(self.param_dict[layer]):
                                o = f.view(f.shape[0],-1)
                                o = torch.mm(z,o)
                                o = o.view(z.shape[0],f.shape[1],f.shape[2])
                                original[i,:,:,:] = o
                        
                        scaled = original

                        # update decompose weight
                        self.decompose_weight[index] = scaled
        
        if self.plot_filters:
            self.plot_contents()

        self._weight_init()
    
    def _weight_init(self):
        for layer in self.model.state_dict():
            decomposed_weight = self.decompose_weight.pop(0)
            self.model.state_dict()[layer].copy_(decomposed_weight)

    def build_model_codebook(self):
        assert len(self.codebook_list) != 0

        layer_num = len(self.labels)
        for l_ind in range(layer_num):
            layer_labels = self.labels[l_ind].T

            layer_labels

        #for layer_codebook in self.codebook_list:
        #    print(len(layer_codebook))

    def create_scaling_mat_conv_thres_bn(self, weight, ind):
        
        weight = weight.reshape(weight.shape[0], -1)

        weight_chosen = weight[ind, :]
        scaling_mat = np.zeros([weight.shape[0], weight_chosen.shape[0]])

        for i in range(weight.shape[0]):
            if i in ind: # chosen
                ind_i, = np.where(ind == i)
                assert(len(ind_i) == 1) # check if only one index is found
                scaling_mat[i, ind_i] = 1
            else: # not chosen
                continue

        return scaling_mat

    def init_plotting(self):
        print("Plot init!")
        layers = {
            'vgg': 13,
            'resnet56': 27,
            'resnet50': 26
        }
        layer_depth = layers[self.arch]
        fig, self.norm_plots = plt.subplots(1, layer_depth, figsize=(10*layer_depth, 5), constrained_layout=True) 

        self.layer_remained_filter_means = []
        self.layer_remained_filter_stds = []

        self.layer_remained_filter_optimal_means = []
        self.layer_remained_filter_optimal_stds = []

    def generate_plot_contents(self, original_weight, layer_id, output_channel_index, norm=True, moments=True):
        sub_plot = self.norm_plots[layer_id]

        whole_filters = original_weight.reshape(original_weight.size(0), -1)
        remained_filters = original_weight[output_channel_index,:,:,:].reshape(len(output_channel_index), -1)

        whole_filters_norm = torch.norm(whole_filters, p=2, dim=1)
        remained_filters_norm = torch.norm(remained_filters, p=2, dim=1)

        remained_filters_all_mean = torch.mean(remained_filters.flatten(), dim=0)
        remained_filters_all_std = torch.sqrt(torch.var(remained_filters.flatten(), dim=0))

        n = 3 * 3 * len(output_channel_index)
        optimal_std = math.sqrt(2. / n)

        if moments:
            self.layer_remained_filter_means.append(remained_filters_all_mean)
            self.layer_remained_filter_stds.append(remained_filters_all_std)

            self.layer_remained_filter_optimal_means.append(0)
            self.layer_remained_filter_optimal_stds.append(optimal_std)

        min_range = 0.0 #torch.min(whole_filters_norm).cpu().numpy()
        max_range = 1.5 #torch.max(whole_filters_norm).cpu().numpy()

        whole_filters_norm = whole_filters_norm.cpu().numpy()
        remained_filters_norm = remained_filters_norm.cpu().numpy()
        whole_filters = whole_filters.cpu().numpy()
        remained_filters = remained_filters.cpu().numpy()
        remained_filters_all_mean = remained_filters_all_mean.cpu().numpy()
        remained_filters_all_std = remained_filters_all_std.cpu().numpy()

        if norm:
            sub_plot.hist(whole_filters_norm, label='Whole filters', histtype="step", range=(min_range, max_range), bins=15, density=True)
            sub_plot.hist(remained_filters_norm, label='Remained filters', histtype="step", range=(min_range, max_range), bins=15, density=True)

            sub_plot.set_xlabel("Norm")
            sub_plot.set_ylabel("Density")
            sub_plot.set_title("Layer "+str(layer_id))
            sub_plot.legend()

    def plot_contents(self):
        distance_threshold = str(self.distance_threshold)

        plt.title("Filters l2 norm histogram ("+self.arch+", dt="+distance_threshold+")")
        plt.savefig(self.arch+"_dt="+distance_threshold+"_filters_norm_hist_density_all_remained.pdf")
        plt.show()

        plt.plot(self.layer_remained_filter_optimal_means, label="Optimal value (He)")
        plt.plot(self.layer_remained_filter_means, label="Practical value")
        plt.legend()
        plt.xlabel("Layer index")
        plt.ylabel("Mean")
        plt.title("Mean of the remained filters ("+ self.arch +", dt="+distance_threshold+")")
        plt.savefig(self.arch+"_dt="+distance_threshold+"_filters_mean_plot.pdf")
        plt.show()

        plt.plot(self.layer_remained_filter_optimal_stds, label="Optimal value (He)")
        plt.plot(self.layer_remained_filter_stds, label="Practical value")
        print(self.layer_remained_filter_stds)
        plt.xlabel("Layer index")
        plt.ylabel("Std")
        plt.ylim((0, 0.5))
        plt.title("Std of the remained filters ("+ self.arch +", dt="+distance_threshold+")")
        plt.legend()
        plt.savefig(self.arch+"_dt="+distance_threshold+"_filters_std_plot.pdf")
        plt.show()

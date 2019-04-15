import torch
import torch.nn.functional as F
from torch.autograd import Variable

class GHMC_Loss:
    def __init__(self, bins=10, momentum=0.0):
        self.bins = bins
        self.momentum = momentum
        self.edges = [float(x) / bins for x in range(bins+1)]
        self.edges[-1] += 1e-6
        if momentum > 0:
            self.acc_sum = [0.0 for _ in range(bins)]

    def calc(self, input, target, mask):
        """ Args:
        input [batch_num, class_num]:
            The direct prediction of classification fc layer.
        target [batch_num, class_num]:
            Binary target (0 or 1) for each sample each class. The value is -1
            when the sample is ignored.
        """
        edges = self.edges
        mmt = self.momentum
        weights = torch.zeros_like(input)

        # gradient length
        self.g = torch.abs(input.sigmoid().detach() - target)

        valid = mask > 0
        tot = max(valid.float().sum().item(), 1.0)
        n = 0  # n valid bins
        for i in range(self.bins):
            inds = (self.g >= edges[i]) & (self.g < edges[i+1]) & valid
            num_in_bin = inds.sum().item()
            if num_in_bin > 0:
                if mmt > 0:
                    self.acc_sum[i] = mmt * self.acc_sum[i] \
                        + (1 - mmt) * num_in_bin
                    weights[inds] = tot / self.acc_sum[i]
                else:
                    weights[inds] = tot / num_in_bin
                n += 1
        if n > 0:
            weights = weights / n

        loss = F.binary_cross_entropy_with_logits(
            input, target, weights, reduction='sum') / tot
        return loss, self.acc_sum,self.g



import tensorflow as tf


class GHMCLoss:
    def __init__(self, bins=10, momentum=0.75):
        self.bins = bins
        self.momentum = momentum
        self.edges_left, self.edges_right = self.get_edges(self.bins)  # edges_left: [bins, 1, 1], edges_right: [bins, 1, 1]
        if momentum > 0:
            self.acc_sum = self.get_acc_sum(self.bins) # [bins]

    def get_edges(self, bins):
        edges_left = [float(x) / bins for x in range(bins)]
        edges_left = tf.constant(edges_left) # [bins]
        edges_left = tf.expand_dims(edges_left, -1) # [bins, 1]
        edges_left = tf.expand_dims(edges_left, -1) # [bins, 1, 1]

        edges_right = [float(x) / bins for x in range(1, bins + 1)]
        edges_right[-1] += 1e-6
        edges_right = tf.constant(edges_right) # [bins]
        edges_right = tf.expand_dims(edges_right, -1) # [bins, 1]
        edges_right = tf.expand_dims(edges_right, -1) # [bins, 1, 1]
        return edges_left, edges_right

    def get_acc_sum(self, bins):
        acc_sum = [0.0 for _ in range(bins)]
        return tf.Variable(acc_sum, trainable=False)

    def calc(self, input, target, mask=None, is_mask=False):
        """ Args:
        input [batch_num, class_num]:
            The direct prediction of classification fc layer.
        target [batch_num, class_num]:
            Binary target (0 or 1) for each sample each class. The value is -1
            when the sample is ignored.
        mask [batch_num, class_num]
        """
        edges_left, edges_right = self.edges_left, self.edges_right
        mmt = self.momentum
        # gradient length
        self.g = tf.abs(tf.sigmoid(input) - target) # [batch_num, class_num]
        g = tf.expand_dims(self.g, axis=0) # [1, batch_num, class_num]
        g_greater_equal_edges_left = tf.greater_equal(g, edges_left)# [bins, batch_num, class_num]
        g_less_edges_right = tf.less(g, edges_right)# [bins, batch_num, class_num]
        zero_matrix = tf.cast(tf.zeros_like(g_greater_equal_edges_left), dtype=tf.float32) # [bins, batch_num, class_num]
        if is_mask:
            mask_greater_zero = tf.greater(mask, 0)
            inds = tf.cast(tf.logical_and(tf.logical_and(g_greater_equal_edges_left, g_less_edges_right),
                                          mask_greater_zero), dtype=tf.float32)  # [bins, batch_num, class_num]
            tot = tf.maximum(tf.reduce_sum(tf.cast(mask_greater_zero, dtype=tf.float32)), 1.0)
        else:
            inds = tf.cast(tf.logical_and(g_greater_equal_edges_left, g_less_edges_right),
                           dtype=tf.float32)  # [bins, batch_num, class_num]
            input_shape = tf.shape(input)
            tot = tf.maximum(tf.cast(input_shape[0] * input_shape[1], dtype=tf.float32), 1.0)
        num_in_bin = tf.reduce_sum(inds, axis=[1, 2]) # [bins]
        num_in_bin_greater_zero = tf.greater(num_in_bin, 0) # [bins]
        num_valid_bin = tf.reduce_sum(tf.cast(num_in_bin_greater_zero, dtype=tf.float32))

        # num_in_bin = num_in_bin + 1e-12
        if mmt > 0:
            update = tf.assign(self.acc_sum, tf.where(num_in_bin_greater_zero, mmt * self.acc_sum \
                                  + (1 - mmt) * num_in_bin, self.acc_sum))
            with tf.control_dependencies([update]):
                self.acc_sum_tmp = tf.identity(self.acc_sum, name='updated_accsum')
                acc_sum = tf.expand_dims(self.acc_sum_tmp, -1)  # [bins, 1]
                acc_sum = tf.expand_dims(acc_sum, -1)  # [bins, 1, 1]
                acc_sum = acc_sum + zero_matrix # [bins, batch_num, class_num]
                weights = tf.where(tf.equal(inds, 1), tot / acc_sum, zero_matrix)
                weights = tf.reduce_sum(weights, axis=0)
        else:
            num_in_bin = tf.expand_dims(num_in_bin, -1)  # [bins, 1]
            num_in_bin = tf.expand_dims(num_in_bin, -1)  # [bins, 1, 1]
            num_in_bin = num_in_bin + zero_matrix # [bins, batch_num, class_num]
            weights = tf.where(tf.equal(inds, 1), tot / num_in_bin, zero_matrix)
            weights = tf.reduce_sum(weights, axis=0)
        weights = weights / num_valid_bin
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=input)
        loss = tf.reduce_sum(loss * weights) / tot
        return loss

if __name__ == '__main__':
    ghm = GHMCLoss(momentum=0.75)
    input_1 = tf.constant([[0.05, 0.25],[0.15, 0.65]], dtype=tf.float32) #
    target_1 = tf.constant([[1.0, 0.0], [0.0, 1.0]], dtype=tf.float32)

    input_2 = tf.constant([[0.75, 0.65], [0.85, 0.05]], dtype=tf.float32)
    target_2 = tf.constant([[1.0, 0.0], [0.0, 0.0]], dtype=tf.float32)
    with tf.Session() as sess:
        init = tf.initialize_all_variables()
        sess.run(init)
        loss = ghm.calc(input_1, target_1)
        print(sess.run([loss,ghm.g,ghm.acc_sum_tmp]))
        loss = ghm.calc(input_2, target_2)
        print(sess.run([loss,ghm.g,ghm.acc_sum_tmp]))
        loss = ghm.calc(input_2, target_2)
        print(sess.run([loss,ghm.g,ghm.acc_sum_tmp]))
        loss = ghm.calc(input_1, target_1)
        print(sess.run([loss,ghm.g,ghm.acc_sum_tmp]))
        loss = ghm.calc(input_1, target_1)
        print(sess.run([loss,ghm.g,ghm.acc_sum_tmp]))

        # loss = ghm.calc(input_1, target_1)
        # print(sess.run([loss,ghm.g]))
        # loss = ghm.calc(input_2, target_2)
        # print(sess.run([loss,ghm.g]))
        # loss = ghm.calc(input_2, target_2)
        # print(sess.run([loss,ghm.g]))
        # loss = ghm.calc(input_1, target_1)
        # print(sess.run([loss,ghm.g]))
        # loss = ghm.calc(input_1, target_1)
        # print(sess.run([loss,ghm.g]))

    ghm_ = GHMC_Loss(momentum=0.75)
    input_1 = Variable(torch.torch.Tensor([[0.05,0.25],[0.15,0.65]]))
    target_1 = torch.Tensor([[1.0,0.0], [0.0,1.0]])
    mask_1 = torch.Tensor([[1.0,1.0], [1.0,1.0]])

    input_2 = Variable(torch.Tensor([[0.75,0.65], [0.85,0.05]]))
    target_2 = torch.Tensor([[1.0,0.0], [0.0,0.0]])
    mask_2 = torch.Tensor([[1.0,1.0], [1.0,1.0]])

    loss,acc_sum,g = ghm_.calc(input_1, target_1, mask_1)
    print(loss)
    print(g)
    print(acc_sum)

    loss,acc_sum,g = ghm_.calc(input_2, target_2, mask_2)
    print(loss)
    print(g)
    print(acc_sum)

    loss,acc_sum,g = ghm_.calc(input_2, target_2, mask_2)
    print(loss)
    print(g)
    print(acc_sum)

    loss,acc_sum,g = ghm_.calc(input_1, target_1, mask_1)
    print(loss)
    print(g)
    print(acc_sum)

    loss,acc_sum,g = ghm_.calc(input_1, target_1, mask_1)
    print(loss)
    print(g)
    print(acc_sum)


    # loss = ghm_.calc(input_1, target_1,mask_1)
    # print(loss)
    #
    # loss = ghm_.calc(input_2, target_2,mask_2)
    # print(loss)
    #
    # loss = ghm_.calc(input_2, target_2,mask_2)
    # print(loss)
    #
    # loss = ghm_.calc(input_1, target_1,mask_1)
    # print(loss)
    #
    # loss = ghm_.calc(input_1, target_1,mask_1)
    # print(loss)


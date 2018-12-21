from datetime import datetime
import time
import math
import collections
import tensorflow as tf

slim = tf.contrib.slim

# 使用collections.namedtuple设计ResNet的Block模块
# scope参数是block的名称
# unit_fn是功能单元（如残差单元）
# args是一个列表，如([256, 64, 1]) X 2 + [256, 64, 2]),代表两个（256, 64, 1）单元
# 和一个（256, 64, 2）单元
Block = collections.namedtuple("Block", ['scope', 'unit_fn', 'args'])


# 定义下采样的方法，通过max_pool2d实现
def subsample(inputs, factor, scope=None):
    if factor == 1:
        return inputs
    else:
        return slim.max_pool2d(inputs, [1, 1], stride=factor, scope=scope)


# 定义一个创建卷积层的函数
def conv2d_same(inputs, num_outputs, kernel_size, stride, scope=None):
    if stride == 1:
        return slim.conv2d(inputs, num_outputs, kernel_size, stride=1,
                           padding='SAME', scope=scope)
    else:
        pad_total = kernel_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                 [pad_beg, pad_end], [0, 0]])
        return slim.conv2d(inputs, num_outputs, kernel_size, stride=stride,
                           padding='VALID', scope=scope)


# 定义堆叠的block函数
@slim.add_arg_scope
def stack_blocks_dense(net, blocks, outputs_collections=None,weights_initializer=None):
    for block in blocks:
        with tf.variable_scope(block.scope, 'block', [net]) as sc:
            for i, unit in enumerate(block.args):
                with tf.variable_scope('unit_%d' % (i + 1), values=[net]):
                    unit_depth, unit_depth_bottleneck, unit_stride = unit
                    net = block.unit_fn(net,
                                        depth=unit_depth,
                                        depth_bottleneck=unit_depth_bottleneck,
                                        stride=unit_stride)
            net = slim.utils.collect_named_outputs(outputs_collections, sc.name,
                                                   net)

    return net


# 用于设定默认值
def resnet_arg_scope(is_training=True,
                     weight_decay=0.0001,
                     batch_norm_decay=0.997,
                     batch_norm_epsilon=1e-5,
                     weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                     batch_norm_scale=True):
    batch_norm_params = {
        'is_training': is_training,
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
    }

    with slim.arg_scope(
            [slim.conv2d],
            weights_regularizer=slim.l2_regularizer(weight_decay),
            weights_initializer=slim.variance_scaling_initializer(),
            activation_fn=tf.nn.relu,
            normalizer_fn=slim.batch_norm,
            normalizer_params=batch_norm_params
    ):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
                return arg_sc


# 定义残差学习单元
@slim.add_arg_scope
def bottleneck(inputs, depth, depth_bottleneck, stride,
               outputs_collections=None,weights_initializer=None, scope=None):
    with tf.variable_scope(scope, 'bottleneck_v2', [inputs]) as sc:
        depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
        preact = slim.batch_norm(inputs, activation_fn=tf.nn.relu,
                                 scope='preact')
        # shortcut为直连的X
        if depth == depth_in:
            shortcut = subsample(inputs, stride, 'shortcut')
        else:
            shortcut = slim.conv2d(preact, depth, [1, 1], stride=stride,
                                   normalizer_fn=None, activation_fn=None,
                                   scope='shortcut')

        residual = slim.conv2d(preact, depth_bottleneck, [1, 1], stride=1,
                               scope='conv1')
        residual = conv2d_same(residual, depth_bottleneck, 3, stride,
                               scope='conv2')
        residual = slim.conv2d(residual, depth, [1, 1], stride=1,
                               normalizer_fn=None, activation_fn=None,
                               scope='conv3')

        # 将直连的X加到残差上，得到output
        output = shortcut + residual

        return slim.utils.collect_named_outputs(outputs_collections,
                                                sc.name, output)


# 定义ResNet的主函数
def resnet_v2(inputs,
              blocks,
              num_classes=None,
              global_pool=True,
              include_root_block=True,
              reuse=None,
              scope=None):
    with tf.variable_scope(scope, 'resnet_v2', [inputs], reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d,slim.fully_connected, bottleneck,
                             stack_blocks_dense],
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.005),
                            outputs_collections=end_points_collection):
            net = inputs
            if include_root_block:
                with slim.arg_scope([slim.conv2d], activation_fn=None,
                                    normalizer_fn=None):
                    net = conv2d_same(net, 64, 7, stride=2, scope='conv1')
                net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')
            net = stack_blocks_dense(net, blocks)
            net = slim.batch_norm(net, activation_fn=tf.nn.relu, scope='postnorm')
            if global_pool:
                net = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=True)
            if num_classes is not None:
                net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                                  normalizer_fn=None, scope='logits')
            end_points = slim.utils.convert_collection_to_dict(
                end_points_collection
            )
            net = tf.squeeze(net, [1, 2])
            return net#, end_points


# 定义152层的ResNet
def resnet_v2_152(inputs,
                  num_classes=None,
                  global_pool=True,
                  reuse=None,
                  scope='resnet_v2_152'):
    blocks = [
        Block('block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
        Block('block2', bottleneck, [(512, 128, 1)] * 7 + [(512, 128, 2)]),
        Block('block3', bottleneck, [(1024, 256, 1)] * 35 + [(1024, 256, 2)]),
        Block('block4', bottleneck, [(2048, 512, 1)] * 3)
    ]
    return resnet_v2(inputs, blocks, num_classes, global_pool,
                     include_root_block=True, reuse=reuse, scope=scope)


�	Kvl�ڂ@Kvl�ڂ@!Kvl�ڂ@	�Z�x@�?�Z�x@�?!�Z�x@�?"p
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails/6Kvl�ڂ@a�ri�B@1���_�o@I�+�V݋u@YM�~2Ƈ�?*	���ƃ_@2j
3Iterator::Model::ParallelMap::Zip[1]::ForeverRepeaty\T��b�?!θ��-|<@)���:�?1���08@:Preprocessing2F
Iterator::Model�Y/�r��?!J����F@)�+�,�?1����(7@:Preprocessing2S
Iterator::Model::ParallelMap��i��_�?!���6@)��i��_�?1���6@:Preprocessing2t
=Iterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate���ECƓ?!e��;V�.@)�����?1�tyc;L"@:Preprocessing2X
!Iterator::Model::ParallelMap::Zip�#F�-t�?!� #�
K@)K�ɀ?1�M8�@:Preprocessing2�
MIterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�w���?!qc��5�@)�w���?1qc��5�@:Preprocessing2v
?Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat::FromTensorl��+v?!s���-@)l��+v?1s���-@:Preprocessing2d
-Iterator::Model::ParallelMap::Zip[0]::FlatMap,Ԛ���?!<�o�#3@)�+�j�s?1L;,�;@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"@57.1 % of the total step time sampled is spent on Kernel Launch.*no#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	a�ri�B@a�ri�B@!a�ri�B@      ��!       "	���_�o@���_�o@!���_�o@*      ��!       2      ��!       :	�+�V݋u@�+�V݋u@!�+�V݋u@B      ��!       J	M�~2Ƈ�?M�~2Ƈ�?!M�~2Ƈ�?R      ��!       Z	M�~2Ƈ�?M�~2Ƈ�?!M�~2Ƈ�?JGPU�"]
2gradient_tape/model/conv3d_1/Conv3DBackpropInputV2Conv3DBackpropInputV2G����?!G����?"1
model/conv3d_1/Conv3DConv3D���j>��?!$V7!�^�?"_
3gradient_tape/model/conv3d_1/Conv3DBackpropFilterV2Conv3DBackpropFilterV2��;�:�?!��>�?"]
2gradient_tape/model/conv3d_2/Conv3DBackpropInputV2Conv3DBackpropInputV2��k����?!ʑ��S�?"_
3gradient_tape/model/conv3d_2/Conv3DBackpropFilterV2Conv3DBackpropFilterV2�:��E�?!�9�{G�?"/
model/conv3d/Conv3DConv3D�\�����?!�R���?"1
model/conv3d_2/Conv3DConv3D���R\�?!�[y��\�?"]
1gradient_tape/model/conv3d/Conv3DBackpropFilterV2Conv3DBackpropFilterV2���n.�?!��`�o�?"�
fgradient_tape/model/lstm/while/model/lstm/while_grad/body/_1471/gradients/lstm_cell/MatMul_grad/MatMulMatMul�cE�ΐ?!J���D>�?"�
hgradient_tape/model/lstm/while/model/lstm/while_grad/body/_1471/gradients/lstm_cell/MatMul_1_grad/MatMulMatMul��;_�?!��>��?2blackQ      Y@"�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high@57.1 % of the total step time sampled is spent on Kernel Launch.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQ2"GPU(: 
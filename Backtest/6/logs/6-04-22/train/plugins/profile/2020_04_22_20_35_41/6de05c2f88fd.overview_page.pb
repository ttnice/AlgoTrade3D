�	G�Z�Q�@G�Z�Q�@!G�Z�Q�@	l>�궡?l>�궡?!l>�궡?"p
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails/<G�Z�Q�@�캷"�0@1��CR˚~@Ic��y�v@Y��n�;2�?*	l�t��]@2F
Iterator::Model���]���?!��r�mG@)%#gaO;�?1���z}@@:Preprocessing2j
3Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat{�"0�7�?!]���	p:@)�J�E��?11����58@:Preprocessing2S
Iterator::Model::ParallelMapw��g�?!��һ��+@)w��g�?1��һ��+@:Preprocessing2t
=Iterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate=��S��?!<E��b0@)�x��?1�0S��&@:Preprocessing2X
!Iterator::Model::ParallelMap::Zip�a���L�?!c#�T�J@)f��Os�?1t~xI�@:Preprocessing2�
MIterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlices�4�Bx?!!��#�@)s�4�Bx?1!��#�@:Preprocessing2d
-Iterator::Model::ParallelMap::Zip[0]::FlatMap>B͐*��?!:��F�/3@)���{k?1��j�f@:Preprocessing2v
?Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat::FromTensor� ݗ3�e?!^�J�t�@)� ݗ3�e?1^�J�t�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"@41.5 % of the total step time sampled is spent on Kernel Launch.*no#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�캷"�0@�캷"�0@!�캷"�0@      ��!       "	��CR˚~@��CR˚~@!��CR˚~@*      ��!       2      ��!       :	c��y�v@c��y�v@!c��y�v@B      ��!       J	��n�;2�?��n�;2�?!��n�;2�?R      ��!       Z	��n�;2�?��n�;2�?!��n�;2�?JGPU�	"�
hgradient_tape/model/lstm/while/model/lstm/while_grad/body/_1471/gradients/lstm_cell/MatMul_1_grad/MatMulMatMul�I}�ͫ?!�I}�ͫ?"�
ngradient_tape/model/lstm_1/while/model/lstm_1/while_grad/body/_1301/gradients/lstm_cell_1/MatMul_1_grad/MatMulMatMul�u��Z��?!�_8�n��?"�
lgradient_tape/model/lstm_1/while/model/lstm_1/while_grad/body/_1301/gradients/lstm_cell_1/MatMul_grad/MatMulMatMulxj%��d�?!v��Ij��?"�
jgradient_tape/model/lstm/while/model/lstm/while_grad/body/_1471/gradients/lstm_cell/MatMul_1_grad/MatMul_1MatMul��븿=�?!`x 8���?"�
ngradient_tape/model/lstm_1/while/model/lstm_1/while_grad/body/_1301/gradients/lstm_cell_1/MatMul_grad/MatMul_1MatMulӮ�?��?!&--j��?"�
pgradient_tape/model/lstm_1/while/model/lstm_1/while_grad/body/_1301/gradients/lstm_cell_1/MatMul_1_grad/MatMul_1MatMul�*'�v��?!��*����?"M
1model/lstm_1/while/body/_159/lstm_cell_1/MatMul_1MatMul眠��?!�l��m��?"G
+model/lstm/while/body/_1/lstm_cell/MatMul_1MatMul[����ן?!j�B���?"K
/model/lstm_1/while/body/_159/lstm_cell_1/MatMulMatMul�k.oǟ?!)�4`��?"�
lgradient_tape/model/lstm_2/while/model/lstm_2/while_grad/body/_1131/gradients/lstm_cell_2/MatMul_grad/MatMulMatMulTs(����?!^z��z��?2blackQ      Y@"�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high@41.5 % of the total step time sampled is spent on Kernel Launch.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQ2"GPU(: 
�	��Q%�@��Q%�@!��Q%�@	;��sU'�?;��sU'�?!;��sU'�?"p
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails/6��Q%�@1�Tm7�5@1�<�Ȗ�@I~5��|@Yrjg��R�?*	����sd@2j
3Iterator::Model::ParallelMap::Zip[1]::ForeverRepeatR����?!K����H<@)�+��?1���sɍ8@:Preprocessing2F
Iterator::Modelr���	�?!�n��
�A@)9{�ᯡ?1(o#�5@:Preprocessing2S
Iterator::Model::ParallelMapr�	�OƘ?!Q�V��-@)r�	�OƘ?1Q�V��-@:Preprocessing2d
-Iterator::Model::ParallelMap::Zip[0]::FlatMap�\5��?!���~��=@)�+ٱ��?1��ߺ,@:Preprocessing2t
=Iterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate��ฌ��?!��#��/@)pUj��?1e�kee?$@:Preprocessing2X
!Iterator::Model::ParallelMap::Zip(I�L�ٺ?!�H,�zP@)z����?1-CU��@:Preprocessing2�
MIterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceZ-��DJ�?!��oq�@)Z-��DJ�?1��oq�@:Preprocessing2v
?Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat::FromTensor��� y?!��@��@)��� y?1��@��@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"@28.7 % of the total step time sampled is spent on Kernel Launch.*no#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	1�Tm7�5@1�Tm7�5@!1�Tm7�5@      ��!       "	�<�Ȗ�@�<�Ȗ�@!�<�Ȗ�@*      ��!       2      ��!       :	~5��|@~5��|@!~5��|@B      ��!       J	rjg��R�?rjg��R�?!rjg��R�?R      ��!       Z	rjg��R�?rjg��R�?!rjg��R�?JGPU�"]
2gradient_tape/model/conv3d_1/Conv3DBackpropInputV2Conv3DBackpropInputV2O�f�V�?!O�f�V�?"_
3gradient_tape/model/conv3d_1/Conv3DBackpropFilterV2Conv3DBackpropFilterV2u�P3�?!�{o���?"1
model/conv3d_1/Conv3DConv3D�ܰ��?!��n�#�?"]
2gradient_tape/model/conv3d_2/Conv3DBackpropInputV2Conv3DBackpropInputV2�Q���[�?!
 ���N�?"1
model/conv3d_2/Conv3DConv3D6]@�ޠ?!��H�j�?"_
3gradient_tape/model/conv3d_2/Conv3DBackpropFilterV2Conv3DBackpropFilterV2cs�Y��?!:-4�|�?"/
model/conv3d/Conv3DConv3D��%\��?!c����-�?"�
fgradient_tape/model/lstm/while/model/lstm/while_grad/body/_1471/gradients/lstm_cell/MatMul_grad/MatMulMatMul�z!Gf�?!:�0� ��?"�
ngradient_tape/model/lstm_1/while/model/lstm_1/while_grad/body/_1301/gradients/lstm_cell_1/MatMul_1_grad/MatMulMatMul ���Ň�?![§�^��?"�
lgradient_tape/model/lstm_1/while/model/lstm_1/while_grad/body/_1301/gradients/lstm_cell_1/MatMul_grad/MatMulMatMul�����l�?!qY��h�?2blackQ      Y@"�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high@28.7 % of the total step time sampled is spent on Kernel Launch.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQ2"GPU(: 
	 ?Ȓ???@ ?Ȓ???@! ?Ȓ???@	b?j????b?j????!b?j????"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails: ?Ȓ???@ڭe2ϻ?AϤM?=??@Y???1????rEagerKernelExecute 0*	??"??F?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generatorv?և??<@!c΂???X@)v?և??<@1c΂???X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?????!?z"??n??)?????1?z"??n??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism0???"??!ޭ??)???)I?Ǵ6???1??σ??:Preprocessing2F
Iterator::Model5)?^Ұ?!-J?>??)?鲘?||?1o??w????:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap?B˺?<@![@??y?X@)cb?qm?h?1??>'J??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9c?j????I?Ui??X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	ڭe2ϻ?ڭe2ϻ?!ڭe2ϻ?      ??!       "      ??!       *      ??!       2	ϤM?=??@ϤM?=??@!ϤM?=??@:      ??!       B      ??!       J	???1???????1????!???1????R      ??!       Z	???1???????1????!???1????b      ??!       JCPU_ONLYYc?j????b q?Ui??X@